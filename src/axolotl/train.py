"""Prepare and train a model on a dataset. Can also infer from a model or merge lora"""

import os
import signal
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import transformers.modelcard
from accelerate.logging import get_logger
from datasets import Dataset
from peft import PeftModel
from pkg_resources import get_distribution  # type: ignore
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from axolotl.common.cli import TrainerCliArgs
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
from axolotl.utils.freeze import freeze_layers_except
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.utils.trainer import setup_trainer
from axolotl.utils.metrics import record_metrics_to_finetune_job, upload_files_to_s3, randomid, register_model_to_finetune_job

try:
    from optimum.bettertransformer import BetterTransformer
except ImportError:
    BetterTransformer = None

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

configure_logging()
LOG = get_logger("axolotl.train")


@dataclass
class TrainDatasetMeta:
    """
    dataclass to capture the dataset specific options for training
    """

    train_dataset: Dataset
    eval_dataset: Optional[Dataset] = None
    total_num_steps: Optional[int] = None


def train(
    *, cfg: DictDefault, cli_args: TrainerCliArgs, dataset_meta: TrainDatasetMeta
) -> Tuple[Union[PeftModel, PreTrainedModel], PreTrainedTokenizer]:
    # load the tokenizer first
    LOG.debug(
        f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}",
        main_process_only=True,
    )
    tokenizer = load_tokenizer(cfg)

    train_dataset = dataset_meta.train_dataset
    eval_dataset = dataset_meta.eval_dataset
    total_num_steps = dataset_meta.total_num_steps

    if cfg.resume_from_checkpoint is None and cfg.auto_resume_from_checkpoints:
        possible_checkpoints = [
            str(cp) for cp in Path(cfg.output_dir).glob("checkpoint-*")
        ]
        if len(possible_checkpoints) > 0:
            sorted_paths = sorted(
                possible_checkpoints,
                key=lambda path: int(path.split("-")[-1]),
            )
            cfg.resume_from_checkpoint = sorted_paths[-1]
            LOG.info(
                f"Using Auto-resume functionality to start with checkpoint at {cfg.resume_from_checkpoint}"
            )
    resume_from_checkpoint = cfg.resume_from_checkpoint

    # Load the model and tokenizer
    msg = "loading model"
    if cfg.adapter:
        msg += " and peft_config..."
    LOG.debug(msg)
    model, peft_config = load_model(cfg, tokenizer, inference=cli_args.inference)
    print("Hello1", model.num_parameters(), model.num_parameters(only_trainable=True))
    model.generation_config.do_sample = True

    model_ref = None
    if cfg.rl and cfg.rl != "orpo":
        if cfg.adapter and not cfg.rl_adapter_ref_model:
            # use built-in trl autounwrap
            LOG.debug("Passing model_ref: None to RL trainer")
            model_ref = None  # explicit setting to None
        else:
            # load the model again for model_ref/baseline
            model_ref, _ = load_model(
                cfg, tokenizer, inference=cli_args.inference, reference_model=True
            )

    safe_serialization = cfg.save_safetensors is True

    if cfg.unfrozen_parameters:
        freeze_layers_except(model, cfg.unfrozen_parameters)

    trainer = setup_trainer(
        cfg,
        train_dataset,
        eval_dataset,
        (model, model_ref, peft_config),
        tokenizer,
        total_num_steps,
    )

    # go ahead and presave, so we have the adapter config available to inspect
    if peft_config:
        LOG.info(f"Pre-saving adapter config to {cfg.output_dir}")
        peft_config.save_pretrained(cfg.output_dir)
    # additionally presave the tokenizer and model configs
    if not Path(cfg.output_dir).is_dir():
        os.makedirs(cfg.output_dir, exist_ok=True)
    tokenizer.save_pretrained(str(Path(cfg.output_dir)))
    if hasattr(model, "config"):
        model.config.save_pretrained(str(Path(cfg.output_dir)))

    # In case we want to stop early with ctrl+c, this is a nice to have to save the pretrained model
    if cfg.local_rank == 0:

        def terminate_handler(_, __, model):
            if cfg.flash_optimum and BetterTransformer:
                model = BetterTransformer.reverse(model)
            model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)
            sys.exit(0)

        signal.signal(
            signal.SIGINT, lambda signum, frame: terminate_handler(signum, frame, model)
        )

    badge_markdown = """[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)"""
    transformers.modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n{badge_markdown}"

    if getattr(cfg, "axolotl_config_path"):
        raw_axolotl_cfg = Path(cfg.axolotl_config_path)
        version = get_distribution("axolotl").version
        if raw_axolotl_cfg.is_file():
            transformers.modelcard.AUTOGENERATED_TRAINER_COMMENT += f"\n<details><summary>See axolotl config</summary>\n\naxolotl version: `{version}`\n```yaml\n{raw_axolotl_cfg.read_text(encoding='utf-8')}\n```\n\n</details><br>\n"

    LOG.info("Starting trainer...")
    if cfg.group_by_length:
        LOG.info("hang tight... sorting dataset for group_by_length")

    pretrain_hooks(cfg, trainer)
    if cfg.flash_optimum:
        with torch.backends.cuda.sdp_kernel(
            # TODO configure these from the YAML w/ sdp_kernel_kwargs: ...
            enable_flash=True,
            enable_math=True,
            enable_mem_efficient=True,
        ):
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    post_train_hooks(cfg, trainer)

    LOG.info(f"Training Completed!!! Saving pre-trained model to {cfg.output_dir}")

    # post training
    for name, module in model.named_modules():
        if hasattr(module, "_post_training"):
            module._post_training(model, name)  # pylint: disable=protected-access

    if trainer.is_fsdp_enabled:
        trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        LOG.info("Set FSDP state dict type to FULL_STATE_DICT for saving.")

    if cfg.relora_steps:
        if cfg.adapter == "lora" and not (cfg.load_in_4bit or cfg.load_in_8bit):
            model = model.merge_and_unload()
        else:
            # final model weights have already been saved by `ReLoRACallback.on_train_end`
            return model, tokenizer

    # TODO do we need this fix? https://huggingface.co/docs/accelerate/usage_guides/fsdp#saving-and-loading
    # only save on rank 0, otherwise it corrupts output on multi-GPU when multiple processes attempt to write the same file
    if cfg.fsdp:
        trainer.save_model(cfg.output_dir)
    elif cfg.deepspeed and is_deepspeed_zero3_enabled():
        # Copied over from: https://github.com/huggingface/accelerate/blob/5ae611118057232f441055f7ef9ba0b0f2b8d533/docs/source/usage_guides/deepspeed.md#saving-and-loading
        trainer.accelerator.wait_for_everyone()
        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model_wrapped)

        # Saves the whole/unpartitioned fp16 model when in ZeRO Stage-3 to the output directory if
        # `stage3_gather_16bit_weights_on_model_save` is True in DeepSpeed Config file or
        # `zero3_save_16bit_model` is True in DeepSpeed Plugin.
        # For Zero Stages 1 and 2, models are saved as usual in the output directory.
        # The model name saved is `pytorch_model.bin`
        unwrapped_model.save_pretrained(
            cfg.output_dir,
            is_main_process=trainer.accelerator.is_main_process,
            save_function=trainer.accelerator.save,
            state_dict=trainer.accelerator.get_state_dict(trainer.model_wrapped),
        )
    elif cfg.local_rank == 0:
        if cfg.flash_optimum and BetterTransformer:
            model = BetterTransformer.reverse(model)

        model.save_pretrained(cfg.output_dir, safe_serialization=safe_serialization)

    if not cfg.hub_model_id:
        try:
            trainer.create_model_card(model_name=cfg.output_dir.lstrip("./"))
        except AttributeError:
            pass
    elif cfg.hub_model_id:
        # defensively push to the hub to ensure the model card is updated
        trainer.push_to_hub()
    # update metrics to server
    record_metrics_to_finetune_job('uploading', 'status')
    if cfg.base_model.find("/") != -1:
        generated_model_name = f'ft:{cfg.base_model.split("/").pop()}:nextai::{randomid()}'
    else:
        generated_model_name = f'ft:{cfg.base_model}:nextai::{randomid()}'
    try:
        upload_files_to_s3(cfg.output_dir, generated_model_name)
        size = (model.num_parameters() + model.num_parameters(only_trainable=True)) / (10**9)
        while True:
            stats = register_model_to_finetune_job(name=generated_model_name, path=f's3://nextai-production/models/{generated_model_name}/', parameters=f"{size:.4f}")
            if stats == 200:
                break
            elif stats == 409:
                if cfg.base_model.find("/") != -1:
                    generated_model_name = f'ft:{cfg.base_model.split("/").pop()}:nextai::{randomid()}'
                else:
                    generated_model_name = f'ft:{cfg.base_model}:nextai::{randomid()}'
            else:
                break
        record_metrics_to_finetune_job('succeeded', 'status')
    except Exception as e:
        record_metrics_to_finetune_job('failed', 'status')
        record_metrics_to_finetune_job({
            "code": "upload_failed",
            "err": e.__str__()
        }, 'metadata')
    return model, tokenizer


def pretrain_hooks(_cfg, _trainer):
    """
    Run hooks right before kicking off the training
    :param cfg:
    :param trainer:
    :return:
    """


def post_train_hooks(_cfg, _trainer):
    """
    Run hooks right after training completes
    :param cfg:
    :param trainer:
    :return:
    """
