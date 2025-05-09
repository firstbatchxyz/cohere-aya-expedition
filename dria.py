import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from node import LLMNode
from controller import Controller
from tqdm import tqdm

from metrics import compute_diversity

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

augmentation_prompt = """
You are an expert in instruction augmentation. Each turn, you are given a base instruction and you respond in an augmented version of that instruction that steers the prompt in a new semantic direction. 
You should:
- Only respond with the augmented instruction, nothing else
- Make sure the new prompt is different enough

    """

def fill_prompt(base, generation):
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
        self.num_nodes = num_nodes

        for _ in range(num_nodes):
            self.nodes.append(LLMNode(self.model, self.tokenizer, 
                                seed=len(self.nodes), node_id="n" + str(len(self.nodes)), end_tokens=["<|end|>", "<|endoftext|>"]))

    def bootstrap(self):
        for node in self.nodes:
            node.set_seed()
            augmentation = node.generate(augmentation_prompt + self.base_instruction, self.max_steps)
            self.data.append(augmentation) # store generated data
            # label = node.generate(fill_prompt(self.base_instruction, generation), self.max_steps)
            self.controller.add_concept(augmentation)

    def guided_generation(self):

        for node in tqdm(self.nodes, desc="Guided generation"):
            negative_instructions = self.controller.select(self.num_nodes-2, self.num_nodes) # just an heuristic
            print("selected negatives ", negative_instructions)

            node.set_seed()
            guidance_scale = 1.0; iterations = 0;

            augmentation = node.generate_with_guidance(augmentation_prompt + self.base_instruction, negative_instructions, 150, guidance_scale, 1.0, 0.9)
            #label = node.generate(fill_prompt(self.base_instruction, generation), self.max_steps)
            print("first augmentation ", augmentation)
            print("\n")

            while self.controller.similar(augmentation) and iterations < 5:
                print("retry")
                guidance_scale += 0.5
                augmentation = node.generate_with_guidance(augmentation_prompt + self.base_instruction, negative_instructions, 150, guidance_scale, 1.0, 0.9)
                #label = node.generate(fill_prompt(self.base_instruction, generation), self.max_steps)
                print("secondary ", augmentation)
                print("\n")
                iterations += 1
                
            
            print("***"*20)
            print("\n")
            self.data.append(augmentation) # store generated data
            self.controller.add_concept(augmentation)
            
    def generate_regular(self):
        generations = []
        for node in self.nodes:
            node.set_seed()
            generation = node.generate(self.base_instruction, self.max_steps)
            generations.append(generation) # store generated data

        return generations

    def generate_based_on_guided(self):
        self.guided_generation()

        generations = []
        instructions = self.data
        
        # Disperse each instruction into a node in self.nodes
        for i, instruction in enumerate(instructions):
            node = self.nodes[i]
            node.set_seed()
            generation = node.generate(instruction, self.max_steps)
            generations.append(generation) # store generated data

        return generations
    
    def run(self):
        print("Started running")
        self.bootstrap()
        print("Bootstrap done")
        for _ in range(1):
            self.guided_generation()

if __name__ == "__main__":
    base_instruction =  "Write a short poem"
    dria = Dria(num_nodes=5, base_instruction=base_instruction)
    regular_generations = dria.generate_regular()
    guided_generations = dria.generate_based_on_guided()

    regular_embeddings = dria.controller.embed_texts(regular_generations)
    guided_embeddings = dria.controller.embed_texts(guided_generations)

    regular_metrics = compute_diversity(regular_embeddings)
    guided_metrics = compute_diversity(guided_embeddings)

    print(f"Regular metrics: {regular_metrics}")
    print(f"Guided metrics: {guided_metrics}")