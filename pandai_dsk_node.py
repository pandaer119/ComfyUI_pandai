# pandai_dsk_node.py

import openai
import time
import json
import re
import os
import torch
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from swarm import Swarm, Agent

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class DSKNode:
    def __init__(self):
        self.session_history = []
        self.system_content = "You are a helpful AI assistant."

    @classmethod
    def INPUT_TYPES(cls):
        # 保持原有命名结构
        deepseek_models = [
            "deepseek-chat",
            "deepseek-ai/DeepSeek-R1",
            "Pro/deepseek-ai/DeepSeek-R1",
            "deepseek-ai/DeepSeek-V3",
            "Pro/deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ]
        
        openai_models = [
            "gpt-4o", "gpt-4", "gpt-3.5-turbo",
            "qwen-turbo", "qwen-plus", "qwen-long",
            "glm-4", "glm-3-turbo"
        ]
        
        llm_apis = [
            {"value": "https://api.openai.com/v1", "label": "OpenAI"},
            {"value": "https://api.deepseek.com/v1", "label": "DeepSeek"},
            {"value": "https://api.siliconflow.cn/v1", "label": "SiliconFlow"}
        ]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (sorted(deepseek_models + openai_models), 
                         {"default": "deepseek-ai/DeepSeek-R1"}),
                "mode": (["chat", "translation", "polish", "multi-agent"], 
                        {"default": "chat"}),
                "api_provider": (sorted([api["label"] for api in llm_apis]), 
                                {"default": "DeepSeek"}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 131072}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1}),
                "context_window": ("INT", {"default": 8, "min": 1, "max": 64})
            },
            "optional": {
                "api_key": ("STRING", {"default": ""}),
                "custom_endpoint": ("STRING", {}),
                "system_prompt": ("STRING", {"default": "你是一个专业的AI助手"}),
                "image_prompt": ("STRING", {})
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("text", "image", "history")
    FUNCTION = "process"
    CATEGORY = "PandAI/Integrated"

    # 以下保留原有方法名
    def process(self, prompt, model, mode, api_provider, max_tokens, temperature, 
               top_p, context_window, api_key="", custom_endpoint=None, 
               system_prompt=None, image_prompt=None):
        
        # 初始化和处理逻辑（同之前的UnifiedAINode实现）
        client = self._init_client(api_provider, model, api_key, custom_endpoint)
        model_config = self._model_configurations(model)
        
        messages = self._build_message_history(
            prompt, 
            system_prompt or self.system_content,
            self.session_history[-context_window*2:] if context_window > 0 else []
        )

        if image_prompt:
            messages.append({
                "role": "user",
                "content": {"type": "image_url", "image_url": image_prompt}
            })

        try:
            response = self._execute_model(
                client=client,
                model=model,
                messages=messages,
                max_tokens=min(max_tokens, model_config["max_tokens"]),
                temperature=model_config.get("temperature", temperature),
                top_p=model_config.get("top_p", top_p),
                mode=mode
            )
        except Exception as e:
            return self._handle_error(e)

        image_output = self._generate_image(response) if "IMAGE" in model else pil2tensor(Image.new('RGB', (512, 512), (255, 255, 255)))

        return (response, image_output, json.dumps(messages))

    # 保持原有私有方法结构
    def _init_client(self, provider, model, api_key, custom_endpoint):
        endpoints = {
            "OpenAI": "https://api.openai.com/v1",
            "DeepSeek": self._deepseek_endpoints(model),
            "SiliconFlow": "https://api.siliconflow.cn/v1"
        }

        base_url = custom_endpoint or endpoints.get(provider, "https://api.deepseek.com/v1")
        
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0
        )

    def _deepseek_endpoints(self, model):
        endpoint_map = {
            "deepseek-ai/DeepSeek-R1": "/r1",
            "Pro/deepseek-ai/DeepSeek-R1": "/r1-pro",
            "deepseek-ai/DeepSeek-V3": "/v3",
            "Pro/deepseek-ai/DeepSeek-V3": "/v3-pro",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "/llama-distill"
        }
        return f"https://api.deepseek.com/v1{endpoint_map.get(model, '')}"

    def _model_configurations(self, model):
        configs = {
            "default": {"max_tokens": 4096, "temperature": 0.7, "top_p": 0.95},
            "Pro/deepseek-ai/DeepSeek-R1": {"max_tokens": 16000, "temperature": 0.5},
            "deepseek-ai/DeepSeek-V3": {"max_tokens": 32768},
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": {"max_tokens": 12288, "top_p": 0.85},
            "gpt-4o": {"max_tokens": 131072}
        }
        return configs.get(model, configs["default"])

    # ... (其他方法实现与之前的UnifiedAINode相同)

NODE_CLASS_MAPPINGS = {
    "DSKNode": DSKNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DSKNode": "🔮 PandAI DeepSeek Node"
}
