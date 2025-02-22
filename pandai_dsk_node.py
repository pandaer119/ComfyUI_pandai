import json
from openai import OpenAI
from langdetect import detect

class Pandai_DSK_Node:
    """
    A custom ComfyUI node for interacting with DeepSeek API, supporting text generation, translation, and polishing.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"multiline": False, "default": "your_api_key_here"}),
                "model": (["deepseek-chat"],),
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.1}),
                "presence_penalty": ("FLOAT", {"default": 0, "min": -2, "max": 2, "step": 0.1}),
                "frequency_penalty": ("FLOAT", {"default": 0, "min": -2, "max": 2, "step": 0.1}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "enable_translation": (["enable", "disable"], {"default": "disable"}),
                "enable_polish": (["enable", "disable"], {"default": "disable"}),
            },
            "optional": {
                "history": ("DEEPSEEK_HISTORY",),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "DEEPSEEK_HISTORY")
    RETURN_NAMES = ("generated_text", "translated_text", "polished_text", "history")
    FUNCTION = "run_pandai_dsk"
    CATEGORY = "Pandai Nodes"

    def run_pandai_dsk(self, api_key, model, max_tokens, temperature, top_p, presence_penalty, frequency_penalty,
                       system_prompt, user_prompt, enable_translation, enable_polish, history=None):
        # Initialize client
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # Build messages with proper JSON structure
        messages = self._build_messages(system_prompt, user_prompt, history)
        
        # Generate text
        generated_text = self._generate_text(client, messages, model, max_tokens, temperature, top_p, 
                                           presence_penalty, frequency_penalty)
        
        # Translation logic
        translated_text = self._handle_translation(client, generated_text, enable_translation)
        
        # Polish logic
        polished_text = self._handle_polish(client, translated_text, enable_polish)
        
        # Update history
        new_history = self._update_history(messages, generated_text)
        
        return (generated_text, translated_text, polished_text, new_history)

    def _build_messages(self, system_prompt, user_prompt, history):
        """确保消息结构符合API要求"""
        if history and "messages" in history:
            messages = history["messages"].copy()
        else:
            messages = [{"role": "system", "content": system_prompt}]
        
        # 确保user_prompt是合法的content结构
        messages.append({
            "role": "user",
            "content": user_prompt if isinstance(user_prompt, (str, dict)) else str(user_prompt)
        })
        return messages

    def _generate_text(self, client, messages, model, max_tokens, temperature, top_p, presence_penalty, frequency_penalty):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            raise RuntimeError(f"生成文本失败: {str(e)}")

    def _handle_translation(self, client, text, enable_translation):
        if enable_translation == "disable":
            return ""
            
        src_lang = detect(text)
        if src_lang == "zh-cn":
            prompt = f"Translate to English: {text}"
        else:
            prompt = f"翻译成中文: {text}"
            
        return self._call_api_safely(client, prompt, "translation")

    def _handle_polish(self, client, text, enable_polish):
        if enable_polish == "disable":
            return ""
            
        polish_prompt = (
            f"Expand and polish this text for Flux model input. "
            f"Add details about lighting, textures, colors and atmosphere. "
            f"Enhance with vivid adjectives. Input: {text}"
        )
        return self._call_api_safely(client, polish_prompt, "polishing")

    def _call_api_safely(self, client, prompt, operation_name):
        try:
            return client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                stream=False
            ).choices[0].message.content
        except Exception as e:
            print(f"{operation_name}操作失败: {str(e)}")
            return ""

    def _update_history(self, messages, new_response):
        messages.append({"role": "assistant", "content": new_response})
        return {"messages": messages}

NODE_CLASS_MAPPINGS = {"pandai_dsk_node": Pandai_DSK_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"pandai_dsk_node": "Pandai DSK Node"}