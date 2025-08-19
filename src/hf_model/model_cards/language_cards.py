from .base_card import BaseModelCard
from typing import LiteralString

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Solidity(BaseModelCard):
    def __init__(self,name, model, tokenizer)->None:
        super().__init__(name, model, tokenizer)

    def generate(self, message, max_new_tokens=1400, batch_size=64)->LiteralString:
        inputs = self.tokenizer(message, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)[len(message):]
        return decoded_output

class Solidity_6b_LLM(Solidity):
    def __init__(self):
        name = "Chain-GPT/Solidity-LLM"
        model = AutoModelForCausalLM.from_pretrained("Chain-GPT/Solidity-LLM", trust_remote_code=True, torch_dtype=torch.float16).cuda()
        tokenizer = AutoTokenizer.from_pretrained("Chain-GPT/Solidity-LLM", trust_remote_code=True)
        super().__init__(name, model, tokenizer)
