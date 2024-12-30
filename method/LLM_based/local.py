from typing import Union
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Local_model:
    def __init__(self, model_name, model_path):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.__load_model(model_path=self.model_path)
        self.__load_tokenizer(model_path=self.model_path)
    
    def get_response(self, user_prompt:Union[str,list]):
        if isinstance(user_prompt,str):
            messages = [{"role": "user", "content": user_prompt}]
        elif isinstance(user_prompt,list):
            messages = user_prompt
        else:
            raise ValueError("user_prompt must be str or list")
        return self.__generate(messages)
    
    def __generate(self, messages, max_new_tokens=512):
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
    
    def __load_model(self, model_path):
        if self.device == "cuda":
            # Use 'auto' for automatic device placement across available GPUs
            device_map = "auto"
        else:
            device_map = None  # Defaults to CPU
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32, 
            device_map=device_map,
            low_cpu_mem_usage=True
        )
    
    def __load_tokenizer(self,model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)