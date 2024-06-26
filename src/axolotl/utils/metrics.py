import requests;
import logging;
import os;
from datetime import datetime;
from axolotl.logging_config import configure_logging
from typing import Annotated
from tqdm import tqdm
import boto3
import string
from pathlib import Path
import random

configure_logging()
NEXTAI_LOG = logging.getLogger("nextai.callbacks")
FINETUNE_JOB = os.getenv('FINETUNE_JOB')

def record_metrics_to_finetune_job(metrics, key: Annotated[str, None] = None):
        API_HOST = os.getenv('API_HOST', 'https://api.nextai.co.in')
        FINETUNE_KEY = os.getenv('FINETUNE_KEY')
        if FINETUNE_JOB is not None and FINETUNE_KEY is not None:
            payload = {}
            if not isinstance(metrics, dict):
                payload[key] = metrics
            else:
                if 'train_runtime' in metrics:
                    payload = {
                        'final_metrics': {
                            **metrics,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                elif 'eval_loss' in metrics:
                    payload = {
                        'eval_metrics': {
                            **metrics,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                elif 'loss' in metrics:
                    payload = {
                        'training_metrics': {
                            **metrics,
                            'timestamp': datetime.now().isoformat()
                        }
                    }
                else:
                    payload[key] = metrics
            try:
                request = requests.patch(f'{API_HOST}/fine-tune/jobs/{FINETUNE_JOB}', json=payload, headers={
                    'x-api-key': FINETUNE_KEY
                })
                response = request.json()
                if request.status_code == 200:
                    if isinstance(metrics, dict):
                        NEXTAI_LOG.info(f"Metrics saved for step: {metrics.get('step')}")
                    else:
                        NEXTAI_LOG.info(f"Metrics saved {key}")
                else:
                    if isinstance(metrics, dict):
                        NEXTAI_LOG.warn(f"Failed to save metrics for step: {metrics.get('step')}. Error: {response['message']}")
                    else:
                        NEXTAI_LOG.warn(f"Failed to save metrics for {key}. Error: {response['message']}")
            except Exception as e:
                NEXTAI_LOG.error(e.__str__())
        else:
            NEXTAI_LOG.warn("Finetuning job id / Finetuning job key missing. Please check your environment")

def register_model_to_finetune_job(name, path, parameters = None):
        API_HOST = os.getenv('API_HOST', 'https://api.nextai.co.in')
        FINETUNE_KEY = os.getenv('FINETUNE_KEY')
        if FINETUNE_JOB is not None and FINETUNE_KEY is not None:
            payload = {
                "name": name,
                "path": path,
                "parameters": parameters,
                "job": FINETUNE_JOB
            }
            try:
                request = requests.post(f'{API_HOST}/models/', json=payload, headers={
                    'x-api-key': FINETUNE_KEY
                })
                response = request.json()
                if request.status_code == 200:
                    NEXTAI_LOG.info(f"Model added to registry: {name}")
                else:
                    NEXTAI_LOG.warn(f"Failed to register model {name}. Error: {response['message']}")
                return request.status_code
            except Exception as e:
                NEXTAI_LOG.error(e.__str__())
        else:
            NEXTAI_LOG.warn("Finetuning job id / Finetuning job key missing. Please check your environment")

def randomid(length = 8):
    characters = "";
    characters += string.ascii_letters
    characters += string.digits;
    id = ""
    for i in range(length):
        randomchar = random.Random(os.urandom(4)).choice(characters)
        id += randomchar
    return id

def upload_files_to_s3(output_dir, generated_folder_name):
    AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
    AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
    AWS_REGION = os.getenv('AWS_REGION')
    if AWS_ACCESS_KEY_ID is not None and AWS_SECRET_KEY is not None:
        session = boto3.Session(
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_KEY,
            region_name=AWS_REGION
        )
        s3 = session.client(service_name="s3")
        for file in os.listdir(output_dir):
            path = Path(output_dir, file)
            if os.path.isfile(path):
                size = os.stat(path).st_size;
                with tqdm(total=size, unit="B", unit_scale=True, desc=file) as bar:
                    s3.upload_file(
                        Filename=path,
                        Bucket='nextai-production',
                        Key=f'models/{generated_folder_name}/{file}',
                        Callback=lambda bytes_transferred: bar.update(bytes_transferred),
                    )
    else:
        NEXTAI_LOG.warn("AWS keys missing. Please check your environment")