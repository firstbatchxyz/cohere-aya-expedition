import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
import time

from node import LLMNode
from controller import Controller

classify_prompt = """
You are a domain classifier.  
Task: From a (Base instruction) and its (Output), return the most specific sub-domain / concept that describes the Output relative to the Base, using ≤ 5 words.

Format:  
Category: <your label>

Example  
Base: Generate a math question  
Output: Solve for x: x⁴ - 5x² + 4 = 0  
Category: Quartic polynomial algebra

Now classify:  
Base: {BASE}  
Output: {OUTPUT}  
Category:
"""


def fill_prompt(base, generation):
    return classify_prompt.format(BASE=base, OUTPUT=generation)

class Dria:
    def __init__(self, num_nodes, base_instruction, base_model="Qwen/Qwen3-4B"):
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
        for node in self.nodes:
            generation = node.generate(self.base_instruction, self.max_steps)
            self.data.append(generation) # store generated data
            label = node.generate(fill_prompt(self.base_instruction, generation), self.max_steps)
            self.controller.add_concept(label)

    def guided_generation(self):
        negatives = self.controller.select(k=10)
        for node in self.nodes:
            guidance_scale = 2.0; iterations = 0;
            generation = node.generate_with_guidance(self.base_instruction, negatives, 150, guidance_scale, 1.0, 0.9)
            label = node.generate(fill_prompt(self.base_instruction, generation), self.max_steps)

            while self.controller.similar(label) and iterations < 5:
                generation = node.generate_with_guidance(self.base_instruction, negatives, 150, guidance_scale + 0.5, 1.0, 0.9)
                label = node.generate(fill_prompt(self.base_instruction, generation), self.max_steps)
                iterations += 1
            
            self.data.append(generation) # store generated data
            self.controller.add_concept(label)

if __name__ == "__main__":
    start = time.time()

    BASE_INSTRUCTION = "Write a random tweet."
    NUM_NODES = 8
    dria = Dria(num_nodes=NUM_NODES, base_instruction=BASE_INSTRUCTION)
    dria.guided_generation()
    data = dria.data
    print(data)

    end = time.time()
    print(end - start)
