import json
import openai
import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from typing import Optional, Dict, Any, List, Tuple
import uuid
import time

def pil2tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ContentSafetyError(Exception):
    """内容安全检测异常"""
    pass

@torch.no_grad()
@torch.inference_mode()
class Pandai_DSK_Node:
    """多模型兼容的超级节点"""
    
    # 保持原有端点不变
    VOLCANO_TEXT_API = "https://open.volcengineapi.com/api/v1/llm/chat/completions" 
    VOLCANO_IMAGE_API = "https://open.volcengineapi.com/api/v1/images/generations"
    
    # 新增DeepSeek-R1专用端点（与原有服务共存）
    DEEPSEEK_R1_API = "https://open.volcengineapi.com/api/v1/deepseek-r1/chat/completions"
    
    def __init__(self):
        self.session_history = []
        self.last_response = None

    @classmethod
    def INPUT_TYPES(cls):
        # 保留全部原有输入参数
        original_params = {
            "required": {
                "api_key": ("STRING", {"default": "volc-your-key-here", "multiline": False}),
                "model": ([
                    "volcano-llm-7b", 
                    "volcano-llm-13b",
                    # 新增模型选项（原有列表保持不变）
                    "deepseek-r1",
                    "deepseek-r1-distill"
                ], {"default": "volcano-llm-7b"}),
                "mode": (["text_generation", "image_generation"], {"default": "text_generation"}),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192}),
                "temperature": ("FLOAT", {"default": 1.0, "min": 0, "max": 2}),
                "system_prompt": ("STRING", {"default": "你是一个专业助手", "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "image_width": ("INT", {"default": 1024, "min": 512}),
                "image_height": ("INT", {"default": 1024, "min": 512})
            },
            "optional": {
                "history": ("VOLCANO_HISTORY",),
                "model_id_override": ("STRING", {"default": ""})
            }
        }
        return original_params

    RETURN_TYPES = ("STRING", "IMAGE", "VOLCANO_HISTORY", "STRING")
    RETURN_NAMES = ("text", "image", "history", "raw_json")
    FUNCTION = "process"
    CATEGORY = "Pandai Nodes"

    def process(self, api_key: str, model: str, mode: str, max_tokens: int, temperature: float,
                system_prompt: str, user_prompt: str, image_width: int, image_height: int,
                history: Optional[dict] = None, model_id_override: str = "") -> Tuple[str, torch.Tensor, dict, str]:
        
        # 历史记录处理保持原样
        history = history or {"messages": []}
        
        # 模型选择逻辑增强
        final_model = self._resolve_model(model, model_id_override)
        
        if mode == "image_generation":
            image = self._generate_image(api_key, user_prompt, image_width, image_height)
            return ("", image, history, "")
            
        text, raw_response = self._handle_text(api_key, final_model, max_tokens, temperature, 
                                             system_prompt, user_prompt, history)
        return (text, pil2tensor(Image.new('RGB', (1, 1))), history, json.dumps(raw_response))

    def _resolve_model(self, selected_model: str, override_id: str) -> str:
        """模型解析逻辑（新增处理不影响原有流程）"""
        if override_id:
            return override_id
            
        # 保持原有模型标识不变
        model_mapping = {
            "deepseek-r1": "ep-xxxx-deepseek-r1",
            "deepseek-r1-distill": "ep-xxxx-deepseek-r1-distill"
        }
        return model_mapping.get(selected_model, selected_model)

    def _generate_image(self, api_key: str, prompt: str, width: int, height: int) -> torch.Tensor:
        """图像生成接口完全保持原有实现"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "X-Request-ID": str(uuid.uuid4())
            }
            
            response = requests.post(
                self.VOLCANO_IMAGE_API,
                headers=headers,
                json={
                    "prompt": prompt,
                    "size": f"{width}x{height}",
                    "n": 1,
                    "steps": 50
                },
                timeout=30
            )
            
            # 保持原有错误处理逻辑
            if response.status_code == 200:
                result = response.json()
                if result.get("code") != 0:
                    raise ContentSafetyError(result.get("message", "内容安全审核未通过"))
                
                # 原有下载逻辑不变
                image_url = result["data"][0]["url"]
                for _ in range(3):
                    try:
                        image_data = requests.get(image_url, timeout=15).content
                        return pil2tensor(Image.open(BytesIO(image_data)))
                    except Exception as e:
                        print(f"[图像下载重试 {_+1}/3] {str(e)}")
                        time.sleep(1)
                raise TimeoutError("图像下载失败")
                
            else:
                raise Exception(f"图像API错误: {response.text}")
                
        except ContentSafetyError as e:
            print(f"[安全拦截] {str(e)}")
            return pil2tensor(Image.new('RGB', (1, 1)))
        except Exception as e:
            print(f"[图像生成失败] {str(e)}")
            return pil2tensor(Image.new('RGB', (1, 1)))

    def _handle_text(self, api_key: str, model: str, max_tokens: int, temperature: float,
                    system_prompt: str, user_prompt: str, history: dict) -> Tuple[str, dict]:
        """文本生成接口升级（兼容新旧模型）"""
        try:
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "X-Request-ID": str(uuid.uuid4())
            }
            
            # 消息构建逻辑保持原有结构
            messages = [{"role": "system", "content": system_prompt}]
            messages += history["messages"][-5:] 
            messages.append({"role": "user", "content": user_prompt})
            
            # 动态选择API端点（新增逻辑）
            endpoint = self.VOLCANO_TEXT_API
            if "deepseek" in model.lower():
                endpoint = self.DEEPSEEK_R1_API
                # DeepSeek-R1专用参数（不影响其他模型）
                headers["X-Volc-Engine"] = "deepseek-r1"
            
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": False
            }
            
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=15
            )
            
            # 响应处理保持原有逻辑
            if response.status_code == 200:
                result = response.json()
                if "error" in result:
                    raise Exception(result["error"]["message"])
                
                # 历史记录更新方式不变
                history["messages"].append({"role": "user", "content": user_prompt})
                history["messages"].append({"role": "assistant", "content": result["choices"][0]["message"]["content"]})
                
                return result["choices"][0]["message"]["content"], result
            else:
                error_msg = f"API请求失败: {response.status_code}"
                if response.text:
                    error_detail = response.json().get("error", {}).get("message", "")
                    error_msg += f" | 详细信息: {error_detail}"
                raise Exception(error_msg)
                
        except Exception as e:
            print(f"[文本生成错误] {str(e)}")
            return "请求失败，请检查API Key和模型配置", {}

NODE_CLASS_MAPPINGS = {"Pandai_DSK": Pandai_DSK_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"Pandai_DSK": "Pandai 多模型集成节点"}
