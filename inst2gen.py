from transformers import AutoTokenizer, AutoModelForCausalLM
from node import LLMNode
import json


def getNode(base_model="microsoft/Phi-4-mini-instruct"):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto"
    )
    return LLMNode(
        model,
        tokenizer,
        seed=42,
        node_id="i2g",
        end_tokens=["<|end|>", "<|endoftext|>"],
    )


def clean_aug(txt):

    if "Instruction" in txt:
        txt = txt.split("Instruction")[-1]
    elif "instruction:" in txt:
        txt = txt.split("instruction")[-1]
    elif "Augmentation:" in txt:
        txt = txt.split("Augmentation")[-1]
    elif "augmentation:" in txt:
        txt = txt.split("augmentation")[-1]

    return txt


def genDataset(base_model="microsoft/Phi-4-mini-instruct", inst_path="dria.json"):

    formatter = "\n Provide your answer as Question: <Question here> Answer: <Answer here>. Output only the question-answer pair."

    with open(inst_path, "r") as file:
        data = json.load(file)
    node = getNode(base_model)
    gen = []
    for inst in data:
        prompt = clean_aug(inst)
        gen.append({"text": node.generate(prompt + formatter, 300)})

    with open(inst_path.split(".")[0] + "_dataset.json", "w") as f:
        json.dump(gen, f)


genDataset(inst_path="dria.json")
genDataset(inst_path="bl.json")
