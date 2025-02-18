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

try:
    from swarm import Swarm, Agent  # 可选依赖
except ImportError:
    Swarm = Agent = None

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class Pandai_DSK_Node:
    def __init__(self):
        # 核心属性初始化
        self._openai_endpoint = "https://api.deepseek.com/v1"  # 默认使用DeepSeek
        self._openai_api_key = ""
        self.model_configs = {}
        self.session_history = []
        self.system_content = "You are a helpful AI assistant."
        
        # 兼容ComfyUI节点系统
        self.required = ["input"]
        self.output = ["output"]
        
        # 多模态初始化
        self._image_processor = Image  # 保留PIL引用
        self._current_client = None

    @classmethod
    def INPUT_TYPES(cls):
        # 统一的模型分类
        model_matrix = {
            "DeepSeek": [
                "deepseek-chat", "deepseek-ai/DeepSeek-R1",
                "Pro/deepseek-ai/DeepSeek-R1", "deepseek-ai/DeepSeek-V3",
                "Pro/deepseek-ai/DeepSeek-V3", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
            ],
            "OpenAI & GLM": [
                "gpt-4o", "gpt-4", "gpt-3.5-turbo", "glm-4", "glm-3-turbo"  # 添加智谱GLM中文模型
            ],
            "Multimodal": [
                "black-forest-labs/FLUX.1-schnell", "glm-4v", "qwen-turbo",
                "qwen-plus", "qwen-long"
            ]
        }

        llm_apis = [
            {"value": "openai", "label": "OpenAI"},
            {"value": "deepseek", "label": "DeepSeek"},
            {"value": "siliconflow", "label": "SiliconFlow"},
            {"value": "moonshot", "label": "Moonshot"},
            {"value": "zhipu", "label": "智谱AI"}  # 添加中文大模型支持
        ]

        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "model": (model_matrix["DeepSeek"] + model_matrix["OpenAI & GLM"] + model_matrix["Multimodal"], 
                         {"default": "deepseek-ai/DeepSeek-R1"}),
                "mode": (["chat", "translate", "polish", "multi-agent", "image-gen", "vision"], 
                        {"default": "chat"}),
                "api_provider": (sorted([api["label"] for api in llm_apis]), 
                                {"default": "DeepSeek"}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 131072}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0, "max": 2, "step": 0.05}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0, "max": 1, "step": 0.01}),
                "context_window": ("INT", {"default": 8, "min": 1, "max": 64})
            },
            "optional": {
                "api_key": ("STRING", {"default": "", "password": True}),
                "custom_endpoint": ("STRING", {"placeholder": "https://api.example.com/v1"}),
                "system_prompt": ("STRING", {"default": "你是一个专业的AI助手"}),
                "image_prompt": ("STRING", {"image_upload": True}),
                "history_input": ("DEEPKEEP_HISTORY",)
            }
        }

    RETURN_TYPES = ("STRING", "IMAGE", "DEEPKEEP_HISTORY")
    RETURN_NAMES = ("text", "image", "history")
    FUNCTION = "run_deepseek_node"
    CATEGORY = "Pandai Nodes"

    def run_deepseek_node(self, prompt, model, mode, api_provider, max_tokens, temperature, 
                        top_p, context_window, api_key="", custom_endpoint=None, 
                        system_prompt=None, image_prompt=None, history_input=None):
        try:
            # 初始化客户端
            self._current_client = self._init_client(
                provider_label=api_provider,
                model=model,
                api_key=api_key,
                custom_endpoint=custom_endpoint
            )
            
            # 消息构建
            messages = self._build_messages(
                user_prompt=prompt,
                system_prompt=system_prompt or self.system_content,
                history=history_input,
                context_size=context_window
            )
            
            # 处理输入类型
            if image_prompt or mode == "vision":
                messages = self._handle_multimodal_input(messages, image_prompt)
            
            # 执行模式分发
            if mode == "image-gen":
                result = self._generate_image(prompt, model)
            elif mode == "multi-agent":
                result = self._multi_agent_process(prompt)
            else:
                result = self._text_process(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
            
            return self._format_output(result)
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            return (error_msg, self._black_image(), [])
    
    def _init_client(self, provider_label, model, api_key, custom_endpoint):
        """通用客户端初始化"""
        endpoint_mapping = {
            "OpenAI": "https://api.openai.com/v1",
            "DeepSeek": "https://api.deepseek.com/v1",
            "SiliconFlow": "https://api.siliconflow.cn/v1",
            "Moonshot": "https://api.moonshot.cn/v1",
            "智谱AI": "https://open.bigmodel.cn/api/paas/v3"  # 中文大模型支持
        }
        
        # 自动检测中文环境优先智谱API
        if "zh" in os.getenv("LANG", "").lower() and not api_key:
            provider_label = "智谱AI"
        
        base_url = custom_endpoint or endpoint_mapping.get(provider_label)
        return openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=30.0
        )

    def _handle_multimodal_input(self, messages, image_data):
        """处理图像输入"""
        if isinstance(image_data, str) and image_data.startswith("http"):
            image_content = {"type": "image_url", "image_url": image_data}
        elif image_data:
            image_content = {"type": "image_base64", "data": image_data}
        else:
            return messages
            
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "请分析这张图片"}, image_content]
        })
        return messages

    def _text_process(self, messages, **params):
        """统一文本处理"""
        return self._current_client.chat.completions.create(
            messages=messages,
            **self._get_model_params(params)
        ).choices[0].message.content

    def _generate_image(self, prompt, model):
        """统一图像生成"""
        if "siliconflow" in model:
            return self._siliconflow_image(prompt)
        elif "glm" in model:
            return self._zhipu_image(prompt)
        else:
            return self._dalle_image(prompt)

    # 其他辅助方法...
    
    def _get_model_params(self, params):
        """智能参数优化"""
        config = self._get_model_config(params.get("model"))
        return {
            "max_tokens": min(params.get("max_tokens", 4096), config.get("max_tokens", 4096)),
            "temperature": params.get("temperature", 0.7),
            "top_p": params.get("top_p", 0.95),
            "frequency_penalty": 0.2 if "translate" in params.get("mode", "") else 0
        }

    def _black_image(self):
        """默认黑色图像"""
        return pil2tensor(Image.new("RGB", (512, 512), color="black"))

    def _update_history(self, messages, output):
        """更新历史记录"""
        return messages + [{"role": "assistant", "content": output}]

NODE_CLASS_MAPPINGS = {"Pandai_DSK_Node": Pandai_DSK_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"Pandai_DSK_Node": "Pandai Multimodal AI Node"}
