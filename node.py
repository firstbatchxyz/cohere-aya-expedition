import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List

from transformers.models.auto.modeling_auto import AutoModelForDocumentQuestionAnswering

class LLMNode:
    def __init__(self, model, tokenizer, seed: int, node_id: str, end_tokens: List[str] = None):
        self.node_id = node_id
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        self.model, self.tokenizer = model, tokenizer
        self.sequence_ids, self._prompt_len = None, 0
        # Add stop ids explicitly 
        self.stop_ids = {self.tokenizer.eos_token_id}
        if end_tokens:
            for t in end_tokens:
                tid = self.tokenizer.convert_tokens_to_ids(t)
                if tid and tid != self.tokenizer.unk_token_id:
                    self.stop_ids.add(tid)

    def is_ready(self): return self.model is not None

    def _tokenize_prompt(self, prompt: str):
        txt = self.tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        return self.tokenizer([txt], return_tensors="pt").input_ids.to(self.model.device)

    def apply_instruction(self, prompt: str):
        self.sequence_ids = self._tokenize_prompt(prompt)
        self._prompt_len = self.sequence_ids.shape[-1] if self.sequence_ids is not None else 0

    def get_logits_for_sequence(self, ids: torch.Tensor):
        with torch.no_grad():
            return self.model(input_ids=ids).logits[:, -1, :].cpu()

    def sample_next_token_and_append(self, logits: torch.Tensor, temperature=1.0, top_p=1.0):
        logits = logits.to(self.model.device) if temperature <= 0 else (logits / temperature).to(self.model.device)
        if top_p < 1.0:
            p = F.softmax(logits, -1)
            sp, si = torch.sort(p, -1, True)
            cp = torch.cumsum(sp, -1)
            m = (cp > top_p)
            m[..., 1:], m[..., 0] = m[..., :-1].clone(), False
            mask = torch.scatter(torch.ones_like(p, dtype=torch.bool), -1, si, m)
            logits = logits.masked_fill(mask, -float('inf'))
        #token = torch.multinomial(F.softmax(logits, -1), 1, generator=torch.Generator(device=self.model.device).manual_seed(torch.seed())).squeeze()
        token = torch.multinomial(F.softmax(logits, -1), 1,
                          generator=torch.Generator(device=self.model.device).manual_seed(torch.seed())).squeeze(-1)
        self.sequence_ids = torch.cat([self.sequence_ids, token.unsqueeze(-1)], -1)
        return token.item()

    def get_generated_sequence(self):
        if self.sequence_ids is None: return ""
        ids = self.sequence_ids[0, self._prompt_len:].cpu().tolist()
        return self.tokenizer.decode(ids, skip_special_tokens=False).strip()

    def get_sequence_length(self): return 0 if self.sequence_ids is None else self.sequence_ids.shape[-1]
    def get_prompt_length(self): return self._prompt_len
    def is_end_of_sequence(self):
        if self.sequence_ids is None or self.sequence_ids.shape[-1] <= self._prompt_len:
             return False
        return self.sequence_ids[0, -1].item() in self.stop_ids

    def generate_with_guidance(self,
                               positive_prompt: str,
                               negative_prompts: List[str],
                               max_steps: int,
                               guidance_scale: float,
                               temperature: float,
                               top_p: float) -> str:
        """
        Generates text using Classifier-Free Guidance (CFG) like logic
        with positive and negative prompts.
        """
        self.apply_instruction(positive_prompt)
        if self.sequence_ids is None: # Should not happen if apply_instruction works
             raise ValueError("Failed to initialize sequence with positive prompt.")

        neg_ids_list = [self._tokenize_prompt(n) for n in negative_prompts]

        # Generation loop
        for _ in range(max_steps):
            if self.is_end_of_sequence():
                break
            
            # Logits for the positive prompt
            lp = self.get_logits_for_sequence(self.sequence_ids) # Logits are on CPU

            # Suffix of generated tokens so far
            # Ensure generated_suffix_ids is on the correct device for concatenation later
            generated_suffix_ids = self.sequence_ids[:, self.get_prompt_length():].to(self.model.device)

            if negative_prompts: # Only compute negative logits if there are negative prompts
                logits_negative_conds = []
                for n_prompt_ids in neg_ids_list:
                    # n_prompt_ids is already on model.device from _tokenize_prompt
                    # Ensure generated_suffix_ids is also on model.device
                    input_ids_negative_pass = torch.cat(
                        [n_prompt_ids, generated_suffix_ids], dim=-1
                    )
                    logits_negative_cond = self.get_logits_for_sequence(input_ids_negative_pass) # Logits on CPU
                    logits_negative_conds.append(logits_negative_cond)
                
                if logits_negative_conds: # If list is not empty
                    all_negative_logits = torch.stack(logits_negative_conds, dim=0) # Stack on CPU
                    ln = torch.min(all_negative_logits, dim=0).values # Min on CPU
                    
                    # Guidance calculation (all tensors lp, ln are on CPU here)
                    guided_logits = ln + guidance_scale * (lp - ln)
                else: # Should not happen if negative_prompts is not empty, but as a fallback
                    guided_logits = lp
            else: # No negative prompts, standard generation
                guided_logits = lp
            
            # Sample and append (moves guided_logits to device internally if temp > 0)
            self.sample_next_token_and_append(guided_logits, temperature, top_p)

        return self.get_generated_sequence()

    def generate(self,
                 instruction: str,
                 max_steps: int,
                 temperature: float = 1.0,
                 top_p: float = 1.0) -> str:
        """
        Generates text based on a given instruction using standard sampling.
        """
        self.apply_instruction(instruction)
        if self.sequence_ids is None: # Should not happen if apply_instruction works
             raise ValueError("Failed to initialize sequence with the instruction.")

        # Generation loop
        for _ in range(max_steps):
            if self.is_end_of_sequence():
                break
            
            # Get logits for the current sequence
            # self.sequence_ids is already on the correct device (from apply_instruction or previous append)
            # get_logits_for_sequence returns logits on CPU
            current_logits = self.get_logits_for_sequence(self.sequence_ids)
            
            # Sample the next token and append it to the sequence
            # sample_next_token_and_append handles moving logits to device if temp > 0
            self.sample_next_token_and_append(current_logits, temperature, top_p)

        return self.get_generated_sequence()


if __name__ == "__main__":
    
    augmentation_prompt = """
You are an expert in instruction augmentation. Each turn, you are given a base instruction and you respond in an augmented version of that instruction that steers the prompt in a new semantic direction. 
You should:
- Only respond with the augmented instruction, nothing else
- Make sure the new prompt is different enough

    """

    classify_prompt = """
    You are a domain classifier.  
    Task: From a (Base instruction) and its (Augmentation), return the most specific sub-domain / concept that describes the Augmented Instruction relative to the Base, with a short sentence.

    Format:  
    <your label>

    Now classify:  
    Base: {BASE}  
    Augmentation: {AUGMENTATION}
    """


    base_model = "microsoft/Phi-4-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype="auto", device_map="auto")
    node = LLMNode(model, tokenizer, seed=50, node_id="bombardino", end_tokens=["<|end|>", "<|endoftext|>"])


    pos = "Write down a math question"
    negs = ["Write a short tweet. Topic:Football"]
    gen = node.generate(augmentation_prompt + " " + pos, 150, 1.0, 0.9)
    label = node.generate(classify_prompt.format(BASE=pos, AUGMENTATION=gen), 150, 1.0, 0.9)
    print(gen, label)

    #node.generate_with_guidance()