import torch
import numpy as np

import subprocess


def evaluate_model(model_path, peft_path=None, task=None):

    if peft_path:
        run_command = f"cd lm-evaluation-harness/lm_eval/ && python __main__.py --model hf \
        --model_args pretrained={model_path},load_in_4bit=True,peft=../../{peft_path} \
        --tasks {task}"
    else:
        run_command = f"cd lm-evaluation-harness/lm_eval/ && python __main__.py --model hf \
        --model_args pretrained={model_path} \
        --tasks {task}"

    subprocess.run(run_command, shell=True)

    
