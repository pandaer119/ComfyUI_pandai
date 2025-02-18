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

class Pandai_DSK_Node:
    def __init__(self):
        self.session_history = []
        self.system_content = "You are a helpful AI assistant."

    @classmethod
    def INPUT_TYPES(cls):
        # 完整的模型支持矩阵
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
        
        multimodal_models = [
            "black-forest-labs/FLUX.1-schnell",
            "glm-4v"
        ]

        llm_apis = [
            {"value": "https://api.openai.com/v1", "label": "OpenAI"},
            {"value": "https://api.deepseek.com/v1", "label": "DeepSeek"},
            {"value": "https://api.siliconflow.cn/v1", "label": "SiliconFlow"},
            {"value": "https://api.moonshot.cn/v1", "label": "Moonshot"}
        ]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "model": (deepseek_models + openai_models + multimodal_models, 
                         {"default": "deepseek-ai/DeepSeek-R1"}),
                "mode": (["chat", "translation", "polish", "multi-agent", "image-gen"], 
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
                "image_prompt": ("STRING", {}),
                "history_input": ("DEEPSEEK_HISTORY",)
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "DEEPSEEK_HISTORY")
    RETURN_NAMES = ("text", "image", "history")
    FUNCTION = "run_pandai_dsk"
    CATEGORY = "Pandai Nodes"

    def run_pandai_dsk(self, prompt, model, mode, api_provider, max_tokens, temperature, 
                     top_p, context_window, api_key="", custom_endpoint=None, 
                     system_prompt=None, image_prompt=None, history_input=None):
        
        # 初始化客户端
        client = self._init_client(api_provider, model, api_key, custom_endpoint)
        
        # 统一配置管理
        model_config = self._get_model_config(model)
        
        # 构建消息历史
        messages = self._build_messages(
            prompt, 
            system_prompt or self.system_content,
            history_input,
            context_window
        )

        # 多模态处理
        if image_prompt:
            messages.append({
                "role": "user",
                "content": {"type": "image_url", "image_url": image_prompt}
            })

        # 主处理逻辑
        try:
            if mode == "image-gen":
                response = self._generate_image(client, prompt, model, model_config)
                text_output = f"Image generated with prompt: {prompt}"
                image_output = response
            else:
                text_output = self._process_text(
                    client, model, messages, mode, 
                    max_tokens, temperature, top_p, model_config
                )
                image_output = self._get_default_image()

            # 更新会话历史
            new_history = self._update_history(messages, text_output)

            return (text_output, image_output, new_history)

        except Exception as e:
            return self._handle_error(e)

    def _init_client(self, provider, model, api_key, custom_endpoint):
        endpoints = {
            "OpenAI": self._openai_endpoint(model),
            "DeepSeek": self._deepseek_endpoint(model),
            "SiliconFlow": "https://api.siliconflow.cn/v1",
            "Moonshot": "https://api.moonshot.cn/v1"
        }
        
        base_url = custom_endpoint or endpoints.get(provider, "https://api.deepseek.com/v1")
        
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0
        )

    def _get_model_config(self, model):
        configs = {
            "deepseek-ai/DeepSeek-R1": {"max_tokens": 16000, "temperature": 0.6},
            "Pro/deepseek-ai/DeepSeek-R1": {"max_tokens": 32000, "temperature": 0.5},
            "deepseek-ai/DeepSeek-V3": {"max_tokens": 131072, "top_p": 0.9},
            "gpt-4o": {"max_tokens": 128000, "temperature": 0.3},
            "black-forest-labs/FLUX.1-schnell": {"is_image_model": True}
        }
        return configs.get(model, {"max_tokens": 4096})

    def _build_messages(self, prompt, system_prompt, history, context_size):
        base_messages = [{"role": "system", "content": system_prompt}]
        
        if history:
            valid_history = history["messages"][-context_size*2:]
            return base_messages + valid_history + [{"role": "user", "content": prompt}]
        
        return base_messages + [{"role": "user", "content": prompt}]

    def _process_text(self, client, model, messages, mode, max_tokens, temp, top_p, config):
        if mode == "multi-agent":
            return self._multi_agent_process(client, model, messages)
            
        if mode in ["translation", "polish"]:
            return self._enhance_text(client, model, messages[1]["content"], mode)
            
        return self._basic_generation(client, model, messages, max_tokens, temp, top_p, config)

    def _generate_image(self, client, prompt, model, config):
        if "siliconflow" in model.lower():
            return self._siliconflow_image(client, prompt)
        return self._default_image_generation(prompt)

    # ...（保持其他核心方法不变）

# 保持原有注册结构
NODE_CLASS_MAPPINGS = {
    "pandai_dsk_node": Pandai_DSK_Node
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "pandai_dsk_node": "Pandai DeepSeek Pro Node"
}
