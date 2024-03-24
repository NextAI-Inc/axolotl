import requests;
import logging;
import os;
from datetime import datetime;
from axolotl.logging_config import configure_logging
from typing import Annotated

configure_logging()
NEXTAI_LOG = logging.getLogger("nextai.callbacks")

def record_metrics_to_finetune_job(metrics: Annotated[dict, str, int, float], key: Annotated[str, None]):
        API_HOST = os.getenv('API_HOST', 'https://api.nextai.co.in')
        FINETUNE_JOB = os.getenv('FINETUNE_JOB')
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
                    'X-API-KEY': FINETUNE_KEY
                })
                response = request.json()
                if request.status_code == 200:
                    NEXTAI_LOG.info(f"Metrics saved for step: {metrics['step']}")
                else:
                    NEXTAI_LOG.warn(f"Failed to save metrics for step: {metrics['step']}. Error: {response['message']}")
            except Exception as e:
                NEXTAI_LOG.error(e.__str__())
        else:
            NEXTAI_LOG.warn("Finetuning job id / Finetuning job key missing. Please check your environment")