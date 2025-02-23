import json
import openai
import requests
import torch
import numpy as np
from openai import OpenAI
from PIL import Image
from io import BytesIO
from typing import Optional
import uuid

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ContentSafetyError(Exception):
    """自定义内容安全异常"""
    pass

@torch.no_grad()
@torch.inference_mode()
class Pandai_DSK_Node:
    """集成多平台AI能力的超级节点"""
    
    SAVE_HISTORY = True  # 启用历史记录保存
    
    def __init__(self):
        self.session_history = []
        self.last_response = None

    @classmethod
    def INPUT_TYPES(cls):
        llm_apis = [
            {"value": "https://api.deepseek.com/v1", "label": "DeepSeek"},
            {"value": "https://api.openai.com/v1", "label": "OpenAI"},
            {"value": "https://api.moonshot.cn/v1", "label": "Kimi"},
            {"value": "https://open.volcengineapi.com", "label": "Volcano"}
        ]
        llm_apis_dict = {api["label"]: api["value"] for api in llm_apis}
        
        return {
            "required": {
                "api_key": ("STRING", {"default": "sk-your-key-here", "multiline": False}),
                "model": (["deepseek-chat", "gpt-4", "glm-4", "moonshot-v1-128k", "volcano-llm-7b"], {"default": "deepseek-chat"}),
                "mode": (["text_generation", "translation", "polishing", "image_generation"], {"default": "text_generation"}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0, "max": 2}),
                "system_prompt": ("STRING", {"default": "你是一个有帮助的助手", "multiline": True}),
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

    def process(self, api_key: str, model: str, mode: str, max_tokens: int, temperature: float,
                system_prompt: str, user_prompt: str, api_provider: str, image_width: int,
                image_height: int, history: Optional[dict] = None, json_input: Optional[str] = None):
        
        # 初始化历史记录
        history = history or {"messages": []}
        
        # 火山引擎处理分支
        if api_provider == "Volcano":
            if mode == "image_generation":
                image = self._volcano_generate_image(api_key, user_prompt, image_width, image_height)
                return ("", image, history, "")
            else:
                text = self._volcano_text_generation(api_key, model, system_prompt, user_prompt)
                return (text, pil2tensor(Image.new('RGB', (1, 1))), history, "")
        
        # 其他平台处理
        client = self._init_client(api_key, api_provider)
        
        if mode == "image_generation":
            image = self._generate_image(client, user_prompt, image_width, image_height)
            return ("", image, history, "")
            
        text = self._handle_text(client, model, max_tokens, temperature, system_prompt, user_prompt, history)
        json_output = self._process_json(json_input) if json_input else ""
        
        return (text, pil2tensor(Image.new('RGB', (1, 1))), history, json_output)

    def _volcano_text_generation(self, api_key: str, model: str, system_prompt: str, user_prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Request-ID": str(uuid.uuid4())
        }
        
        payload = {
            "model_name": model,
            "parameters": {
                "system_prompt": system_prompt,
                "max_new_tokens": 4096,
                "temperature": 1.0,
                "language": "zh-CN"
            },
            "messages": [{
                "role": "user",
                "content": user_prompt,
                "i18n_config": {"lang": "zh-HK"}
            }]
        }
        
        try:
            response = requests.post(
                "https://open.volcengineapi.com/api/v1/llm/chat",
                headers=headers,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            return response.json()["data"]["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"[Volcano Error] {str(e)}")
            return "请求失败，请检查API配置"

    def _volcano_generate_image(self, api_key: str, prompt: str, width: int, height: int) -> torch.Tensor:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        valid_resolutions = ["512x512", "1024x1024", "1024x768"]
        if f"{width}x{height}" not in valid_resolutions:
            raise ValueError(f"分辨率错误，可选值: {valid_resolutions}")
        
        payload = {
            "text": prompt,
            "resolution": f"{width}x{height}",
            "style": "realistic",
            "num_images": 1,
            "safety_check": True
        }
        
        try:
            response = requests.post(
                "https://open.volcengineapi.com/api/v1/image_generation",
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            
            if response.json().get("safe_check_score", 1) < 0.8:
                raise ContentSafetyError("内容安全审核未通过")
                
            image_url = response.json()["data"]["images"][0]["url"]
            image_data = requests.get(image_url, timeout=10).content
            return pil2tensor(Image.open(BytesIO(image_data)))
        except ContentSafetyError as e:
            print(f"[安全拦截] {str(e)}")
            return pil2tensor(Image.new('RGB', (1, 1)))
        except Exception as e:
            print(f"[火山图像错误] {str(e)}")
            return pil2tensor(Image.new('RGB', (1, 1)))

NODE_CLASS_MAPPINGS = {"Pandai_DSK": Pandai_DSK_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"Pandai_DSK": "Pandai DSK Super Node"}
