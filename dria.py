import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from node import LLMNode
from controller import Controller
from tqdm import tqdm

from prompts import get_prompt

def fill_prompt(base, generation):
    classify_prompt = get_prompt("classify")
    return classify_prompt.format(BASE=base, OUTPUT=generation)

class Dria:
    def __init__(self, num_nodes, base_instruction, base_model="microsoft/Phi-4-mini-instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
        self.model.eval()
        self.nodes: List[LLMNode] = []
        self.data = []
        self.controller = Controller()
        self.max_steps, self.guidance_scale, self.temp, self.top_p = 150, 2.0, 1.0, 0.9
        self.base_instruction = base_instruction

        for _ in range(num_nodes):
            self.nodes.append(LLMNode(self.model, self.tokenizer, 
                                seed=len(self.nodes), node_id="n" + str(len(self.nodes)), end_tokens=["<|end|>", "<|endoftext|>"]))

    def bootstrap(self):
        augmentation_prompt = get_prompt("aug_v2")
        for node in self.nodes:
            node.set_seed()
            augmentation = node.generate(augmentation_prompt + self.base_instruction, self.max_steps)
            self.data.append(augmentation) # store generated data
            # label = node.generate(fill_prompt(self.base_instruction, generation), self.max_steps)
            self.controller.add_concept(augmentation)

    def guided_generation(self):
        augmentation_prompt = get_prompt("aug_v2")

        for node in tqdm(self.nodes, desc="Guided generation"):
            negative_instructions = self.controller.select(k=3, knn=5)
            guidance_scale = 1.0; iterations = 0;
            node.set_seed()

            augmentation = node.generate_with_guidance(augmentation_prompt + self.base_instruction, negative_instructions, 150, guidance_scale, 1.0, 0.9)
            #label = node.generate(fill_prompt(self.base_instruction, generation), self.max_steps)

            while self.controller.similar(augmentation) and iterations < 5:
                guidance_scale += 0.5
                augmentation = node.generate_with_guidance(augmentation_prompt + self.base_instruction, negative_instructions, 150, guidance_scale, 1.0, 0.9)
                #label = node.generate(fill_prompt(self.base_instruction, generation), self.max_steps)
                iterations += 1
            
            self.data.append(augmentation) # store generated data
            self.controller.add_concept(augmentation)
    
    def run(self, iters=1):
        print("Started running")
        self.bootstrap()
        print("Bootstrap done")
        for _ in range(iters):
            self.guided_generation()

if __name__ == "__main__":
    base_instruction =  "Write a math problem."
    dria = Dria(num_nodes=1, base_instruction=base_instruction)
    dria.run()

    for instruction in dria.data:
        print(instruction)
