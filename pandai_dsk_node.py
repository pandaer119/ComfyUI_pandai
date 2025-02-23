import json
from openai import OpenAI

class Pandai_DSK_Node:
    """
    A custom ComfyUI node for interacting with DeepSeek API, supporting text generation, translation, and polishing.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        """
        Define the input fields for the node.
        """
        return {
            "required": {
                "api_key": ("STRING", {
                    "multiline": False,  # Single line input for API key
                    "default": "your_api_key_here",  # Placeholder for API key
                }),
                "model": (["deepseek-chat"],),  # Supported models
                "max_tokens": ("INT", {"default": 4096, "min": 1, "max": 8192, "step": 1}),
                "temperature": ("FLOAT", {"default": 1, "min": 0, "max": 2, "step": 0.1}),
                "top_p": ("FLOAT", {"default": 1, "min": 0, "max": 1, "step": 0.1}),
                "presence_penalty": ("FLOAT", {"default": 0, "min": -2, "max": 2, "step": 0.1}),
                "frequency_penalty": ("FLOAT", {"default": 0, "min": -2, "max": 2, "step": 0.1}),
                "system_prompt": ("STRING", {"default": "You are a helpful assistant.", "multiline": True}),
                "user_prompt": ("STRING", {"default": "", "multiline": True}),
                "enable_translation": (["enable", "disable"], {"default": "disable"}),  # Enable translation
                "enable_polish": (["enable", "disable"], {"default": "disable"}),  # Enable polish
            },
            "optional": {
                "history": ("DEEPSEEK_HISTORY",),  # Optional conversation history
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "DEEPSEEK_HISTORY")  # Output types: generated_text, polished_text, history
    RETURN_NAMES = ("generated_text", "polished_text", "history")  # Friendly names for outputs
    FUNCTION = "run_pandai_dsk"  # Entry-point method
    CATEGORY = "Pandai Nodes"  # Category in the UI

    def run_pandai_dsk(self, api_key, model, max_tokens, temperature, top_p, presence_penalty, frequency_penalty,
                       system_prompt, user_prompt, enable_translation, enable_polish, history=None):
        """
        Main function to generate text, translate, and polish using DeepSeek API.
        """
        # Initialize OpenAI client with the provided API key
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        # Prepare messages for the API call
        if history is not None:
            messages = history["messages"]
        else:
            messages = [{"role": "system", "content": system_prompt}]
        messages.append({"role": "user", "content": user_prompt})

        # Step 1: Generate text using DeepSeek API
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
        generated_text = response.choices[0].message.content
        messages.append({"role": "assistant", "content": generated_text})

        # Step 2: Translate the text if enabled
        translated_text = generated_text
        if enable_translation == "enable":
            language = self.detect_language(generated_text)
            if language == "en":
                translated_text = self.call_deepseek_api(
                    client=client,
                    prompt=f"Translate the following text to Chinese: {generated_text}",
                    field_name="translated_text"
                )
            else:
                translated_text = self.call_deepseek_api(
                    client=client,
                    prompt=f"Translate the following text to English: {generated_text}",
                    field_name="translated_text"
                )

        # Step 3: Polish the text if enabled
        polished_text = translated_text
        if enable_polish == "enable" and enable_translation == "enable":
            polished_text = self.polish_text(client, translated_text)

        return (generated_text, polished_text, {"messages": messages})

    def detect_language(self, text):
        """
        Detect the language of the input text.
        """
        if any("\u4e00" <= char <= "\u9fff" for char in text):
            return "zh"  # Chinese
        else:
            return "en"  # English

    def polish_text(self, client, text):
        """
        Expand and polish the text to make it more suitable for Flux model input.
        """
        polish_prompt = (
            f"Expand and polish the following text to make it more suitable for Flux model input. "
            f"Add details about the scene, such as lighting, textures, colors, and atmosphere. "
            f"Enhance the description with vivid adjectives and adverbs. "
            f"Ensure the output is concise and directly related to the input. "
            f"Do not add unrelated content. Input: {text}"
        )
        polished_text = self.call_deepseek_api(
            client=client,
            prompt=polish_prompt,
            field_name="polished_text"
        )
        return polished_text

    def call_deepseek_api(self, client, prompt, field_name):
        """
        Helper method to call DeepSeek API and handle the response.
        """
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt},
                ],
                stream=False
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"API call failed: {e}")

# Register the node
NODE_CLASS_MAPPINGS = {
    "pandai_dsk_node": Pandai_DSK_Node  # 节点名称改为 pandai_dsk_node
}

# Friendly name for the node
NODE_DISPLAY_NAME_MAPPINGS = {
    "pandai_dsk_node": "Pandai DSK Node"  # 节点显示名称
}