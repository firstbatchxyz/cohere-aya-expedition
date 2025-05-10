import run_model
import lm_eval

from datasets import load_dataset

# from huggingface_hub import login

import json


def prep_data(config):

    ds = load_dataset("json", data_files=config["datapath"])["train"]

    return ds


def pipeline(config):

    if config["ft"]:
        ds = prep_data(config)
        run_model.train(config["model_path"], ds, config["name"])

        lm_eval.evaluate_model(config["model_path"], config["name"], config["task"])

    else:
        lm_eval.evaluate_model(config["model_path"], None, config["task"])


config_dria = {
    "datapath": "dria_dataset_v1.json",
    "ft": True,
    "task": "tinyGSM8k",
    "model_path": "microsoft/Phi-4-mini-instruct",
    "name": "dria_math_v1",
}
pipeline(config_dria)
