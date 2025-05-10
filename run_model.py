import torch
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import get_peft_model, LoraConfig
from datasets import load_dataset
from trl import SFTTrainer


def load_model(model_path, lora_path=None):

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", trust_remote_code=True, load_in_4bit=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if lora_path:
        model.load_adapter(lora_path)

    return model, tokenizer


def train(model_path, dataset, name):

    model, tokenizer = load_model(model_path)

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=8,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, peft_config)

    train_args = TrainingArguments(
        output_dir=name,
        num_train_epochs=3,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        optim="adamw_torch_fused",
        fp16=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=train_args,
    )

    trainer.train()
    trainer.model.save_pretrained("./" + name)
