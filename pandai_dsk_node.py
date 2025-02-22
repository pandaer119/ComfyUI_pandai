import json
from openai import OpenAI
from langdetect import detect
import copy

class Pandai_DSK_Node:
    # ... INPUT_TYPES和其他方法保持不变 ...

    RETURN_TYPES = ("STRING", "STRING", "STRING", "DEEPSEEK_HISTORY")
    RETURN_NAMES = ("generated_text", "translated_text", "polished_text", "history")
    FUNCTION = "run_pandai_dsk"
    CATEGORY = "Pandai Nodes"

    def run_pandai_dsk(self, api_key, model, max_tokens, temperature, top_p, presence_penalty, frequency_penalty,
                       system_prompt, user_prompt, enable_translation, enable_polish, history=None):
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        
        try:
            # 1. 构建消息历史（深拷贝防止污染原始数据）
            messages = self._build_messages(system_prompt, user_prompt, history)
            
            # 2. 生成文本（添加详细日志）
            print(f"[DEBUG] 发送消息结构: {json.dumps(messages, indent=2, ensure_ascii=False)}")
            generated_text = self._generate_text(client, messages, model, max_tokens, temperature, top_p,
                                               presence_penalty, frequency_penalty)
            print(f"[DEBUG] 原始生成内容: {generated_text}")
            
            # 3. 处理翻译
            translated_text = generated_text  # 默认使用原始文本
            if enable_translation == "enable":
                translated_text = self._handle_translation(client, generated_text)
                print(f"[DEBUG] 翻译后内容: {translated_text}")
                
            # 4. 处理润色
            polished_text = translated_text  # 默认继承翻译结果
            if enable_polish == "enable":
                polished_text = self._handle_polish(client, translated_text)
                print(f"[DEBUG] 润色后内容: {polished_text}")
                
            # 5. 更新历史记录（确保深拷贝）
            new_history = {"messages": copy.deepcopy(messages)}
            new_history["messages"].append({"role": "assistant", "content": generated_text})
            
            return (generated_text, translated_text, polished_text, new_history)
            
        except Exception as e:
            error_msg = f"节点执行失败: {str(e)}"
            print(error_msg)
            return (error_msg, error_msg, error_msg, {"messages": []})

    def _build_messages(self, system_prompt, user_prompt, history):
        """严格验证消息结构"""
        messages = []
        
        # 处理历史记录
        if history and "messages" in history:
            try:
                messages = copy.deepcopy(history["messages"])
                # 验证历史消息格式
                for msg in messages:
                    if "role" not in msg or "content" not in msg:
                        raise ValueError("无效的历史消息格式")
            except Exception as e:
                print(f"历史记录错误: {str(e)}，将使用新对话")
                messages = []
        
        # 初始化系统提示
        if not any(msg["role"] == "system" for msg in messages):
            messages.insert(0, {"role": "system", "content": str(system_prompt)})
        
        # 添加用户输入（强制内容为字符串）
        user_content = str(user_prompt) if user_prompt else ""
        messages.append({
            "role": "user",
            "content": user_content
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
            # 严格解析响应结构
            if not response.choices:
                raise ValueError("API响应中缺少choices字段")
                
            first_choice = response.choices[0]
            if not first_choice.message or not first_choice.message.content:
                raise ValueError("API响应中缺少message.content")
                
            return first_choice.message.content
            
        except Exception as e:
            raise RuntimeError(f"文本生成失败: {str(e)}")

    def _handle_translation(self, client, text):
        try:
            if not text.strip():
                return ""
                
            src_lang = detect(text)
            if src_lang in ["zh-cn", "zh-tw"]:
                prompt = f"将以下内容翻译成英文：{text}"
            else:
                prompt = f"Translate to Chinese：{text}"
            
            translated = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
                temperature=0.7
            ).choices[0].message.content
            
            # 提取实际翻译内容（处理可能的附加说明）
            if "：" in translated:
                return translated.split("：", 1)[-1].strip()
            return translated
            
        except Exception as e:
            print(f"翻译失败: {str(e)}")
            return text  # 失败时返回原始文本

    def _handle_polish(self, client, text):
        try:
            if not text.strip():
                return ""
                
            polish_prompt = (
                "请优化以下文本，添加关于光线、材质、色彩和氛围的细节描述，"
                "使用生动的形容词和副词。保持原意不变。文本："
                f"{text}"
            )
            
            return client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": polish_prompt}],
                max_tokens=2000,
                temperature=0.5
            ).choices[0].message.content
            
        except Exception as e:
            print(f"润色失败: {str(e)}")
            return text  # 失败时返回原始文本

# 注册节点（保持不变）
NODE_CLASS_MAPPINGS = {"pandai_dsk_node": Pandai_DSK_Node}
NODE_DISPLAY_NAME_MAPPINGS = {"pandai_dsk_node": "Pandai DSK Node"}