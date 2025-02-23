import os
import time
import hmac
import hashlib
import json
from enum import Enum
from typing import Optional, Dict, List
from loguru import logger
from dotenv import load_dotenv
from volcengine.ApiInfo import ApiInfo
from volcengine.Credentials import Credentials
from volcengine.ServiceInfo import ServiceInfo
from volcengine.base.Service import Service

# 环境加载
load_dotenv()

class APIType(Enum):
    OPENAI = "openai"
    VOLCANO = "volcano"

class PandaAIProcessor:
    def __init__(self):
        # 初始化火山引擎服务
        if os.getenv("VOLC_ACCESS_KEY"):
            self.volc_service = self._init_volc_service()
            
    def _init_volc_service(self):
        service_info = ServiceInfo(
            "open.volcengineapi.com",
            {'Content-Type': 'application/json'},
            Credentials(
                os.getenv("VOLC_ACCESS_KEY"),
                os.getenv("VOLC_SECRET_KEY"),
                "ark",
                "cn-north-1"
            ),
            5, 5
        )
        api_info = {
            "chat": ApiInfo("POST", "/", {"Action": "Chat", "Version": "2023-08-01"}, {}, [])
        }
        return Service(service_info, api_info)

    @logger.catch
    def _call_volcano_api(self, messages: List[Dict], model: str = "deepseek-r1"):
        """火山引擎专用调用方法"""
        params = {
            "model": model,
            "messages": messages,
            "parameters": {
                "max_new_tokens": self.max_tokens,
                "temperature": self.temperature
            }
        }
        
        try:
            response = self.volc_service.json("chat", {}, json.dumps(params))
            if response["code"] != 0:
                logger.error(f"火山API错误: {response}")
                return None
            return response["data"]
        except Exception as e:
            logger.error(f"火山调用异常: {str(e)}")
            return None
