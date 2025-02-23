import json
import openai
import requests
from openai import OpenAI
from PIL import Image
from io import BytesIO
import torch
import numpy as np

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def extract_json_strings(text):
    json_strings = []
    brace_level = 0
    json_str = ''
    in_json = False
    
    for char in text:
        if char == '{':
            brace_level += 1
            in_json = True
        if in_json:
            json_str += char
        if char == '}':
            brace_level -= 1
        if in_json and brace_level == 0:
            json_strings.append(json_str)
            json_str = ''
            in_json = False

    return json_strings[0] if len(json_strings)>0 else "{}"

class Pandai_DSK_Node:
    """集成文本生成、翻译、润色、图像生成和多API支持的综合节点"""
    
    def __init__(self):
        self.session_history = []
        self.system_content = "You are a helpful assistant."

    @classmethod
    def INPUT_TYPES(cls):
        llm_apis = [
            {"value": "https://api.deepseek.com/v1", "label": "DeepSeek"},
            {"value": "https://api.openai.com/v1", "label": "OpenAI"},
            {"value": "https://api.moonshot.cn/v1", "label": "Kimi"}
        ]
        llm_apis_dict = {api["label"]: api["value"] for api in llm_apis}
        
        return {
            "required": {
                "api_key": ("STRING", {"default": "your_api_key_here", "multiline": False}),
                "model": (["deepseek-chat", "gpt-4", "glm-4", "moonshot-v1-128k"], {"default": "deepseek-chat"}),
                "mode": (["text_generation", "translation", "polishing", "image_generation"], {"default": "text_generation"}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0, "max": 2}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "api_provider": (list(llm_apis_dict.keys()), {"default": "DeepSeek"}),
                "image_width": ("INT", {"default": 512, "min": 256}),
                "image_height": ("INT", {"default": 512, "min": 256})
            },
            "optional": {
                "history": ("DEEPSEEK_HISTORY",),
                "json_input": ("STRING", {"forceInput": True})
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "DEEPSEEK_HISTORY", "STRING")
    RETURN_NAMES = ("text", "image", "history", "json_output")
    FUNCTION = "process"
    CATEGORY = "Pandai Nodes"

    def process(self, api_key, model, mode, max_tokens, temperature, system_prompt, user_prompt,
                api_provider, image_width, image_height, history=None, json_input=None):
        # 初始化客户端
        client = self._init_client(api_key, api_provider)
        
        # 处理不同模式
        if mode == "image_generation":
            image = self._generate_image(client, user_prompt, image_width, image_height)
            return ("", image, {}, "")
            
        text_output = self._handle_text(client, model, max_tokens, temperature, system_prompt, user_prompt, history)
        json_output = self._process_json(json_input) if json_input else ""
        
        return (text_output, pil2tensor(Image.new('RGB', (1, 1))), {"messages": []}, json_output)

    def _init_client(self, api_key, api_provider):
        base_urls = {
            "DeepSeek": "https://api.deepseek.com",
            "OpenAI": "https://api.openai.com/v1",
            "Kimi": "https://api.moonshot.cn/v1"
        }
        return OpenAI(api_key=api_key, base_url=base_urls.get(api_provider))

    def _handle_text(self, client, model, max_tokens, temperature, system_prompt, user_prompt, history):
        messages = history["messages"] if history else [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_prompt})
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content

    def _generate_image(self, client, prompt, width, height):
        try:
            response = client.images.generate(
                prompt=prompt,
                size=f"{width}x{height}",
                quality="hd",
                n=1
            )
            image_url = response.data[0].url
            image_data = requests.get(image_url).content
            return pil2tensor(Image.open(BytesIO(image_data)))
        except Exception as e:
            print(f"Image generation failed: {e}")
            return pil2tensor(Image.new('RGB', (1, 1)))

    def _process_json(self, json_input):
        try:
            repaired = extract_json_strings(json_input)
            return json.dumps(json.loads(repaired), ensure_ascii=False)
        except Exception as e:
            print(f"JSON processing failed: {e}")
            return "{}"

NODE_CLASS_MAPPINGS = {"pandai_dsk_node": Pandai_DSK_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"pandai_dsk_node": "Pandai DSK Super Node"}