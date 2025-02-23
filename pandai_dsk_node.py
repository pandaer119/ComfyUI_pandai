# 省略import部分与工具函数，保留核心修改

class Pandai_DSK_Node:
    def _volcano_text_generation(self, api_key, model, system_prompt, user_prompt):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Security-Token": "your_temp_token"  # 实际部署时移除
        }
        
        # 新增粤语参数适配
        payload = {
            "model_name": model,
            "parameters": {
                "system_prompt": system_prompt,
                "max_new_tokens": 4096,
                "temperature": 1.0,
                "language": "yue"  # 粤语标识
            },
            "messages": [{
                "role": "user", 
                "content": user_prompt,
                "i18n_config": {"lang": "zh-HK"}  # 香港地区适配
            }]
        }
        
        try:
            response = requests.post(
                "https://open.volcengineapi.com/api/v1/llm/chat",
                headers=headers,
                json=payload,
                timeout=10  # 增加超时控制
            )
            response.raise_for_status()
            return response.json()["data"]["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Volcano API Error: {str(e)}")
            return "【系统提示】服务暂时不可用，请稍后重试"

    def _volcano_generate_image(self, api_key, prompt, width, height):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "X-Request-ID": str(uuid.uuid4())  # 增加请求追踪
        }
        
        # 优化图像参数验证
        valid_resolutions = ["512x512", "1024x1024", "1024x768"]
        if f"{width}x{height}" not in valid_resolutions:
            raise ValueError(f"Invalid resolution. Valid options: {valid_resolutions}")
        
        payload = {
            "text": prompt,
            "resolution": f"{width}x{height}",
            "style": "realistic",
            "num_images": 1,
            "safety_check": True  # 开启内容安全审核
        }
        
        try:
            response = requests.post(
                "https://open.volcengineapi.com/api/v1/image_generation",
                headers=headers,
                json=payload,
                timeout=15
            )
            response.raise_for_status()
            
            # 增加内容安全校验
            if response.json().get("safe_check_score", 1) < 0.8:
                raise ContentSafetyError("内容未通过安全审核")
                
            image_url = response.json()["data"]["images"][0]["url"]
            image_data = requests.get(image_url, timeout=10).content
            return pil2tensor(Image.open(BytesIO(image_data)))
        except ContentSafetyError as e:
            print(f"内容安全拦截: {str(e)}")
            return pil2tensor(Image.new('RGB', (1, 1)))
        except Exception as e:
            print(f"Volcano Image Error: {str(e)}")
            return pil2tensor(Image.new('RGB', (1, 1)))
