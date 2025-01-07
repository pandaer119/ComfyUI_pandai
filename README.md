# ComfyUI_pandai
ComfyUI_pandai Node Usage Guide
![workflow1](https://github.com/user-attachments/assets/c501bd3f-c905-41d5-8405-8ae82d9ffe80)
Introduction
The ComfyUI_pandai node is a custom ComfyUI node designed to interact with the DeepSeek API. It supports text generation, translation, and text polishing. With this node, users can easily generate text, translate content, and refine the generated text for better quality.

Installation Steps
Download the Node:

Place the ComfyUI_pandai folder in the ComfyUI_windows_portable/ComfyUI/custom_nodes/ directory.

Ensure the folder contains the following files:

__init__.py

pandai_dsk_node.py

README.md (optional)

requirements.txt (optional)

Install Dependencies:

Open a terminal, navigate to the ComfyUI_pandai folder, and run the following command to install dependencies:

bash
复制
pip install -r requirements.txt
If there is no requirements.txt file, manually install the following dependencies:

bash
复制
pip install openai langdetect
Restart ComfyUI:

Restart ComfyUI to ensure the new node is loaded.

Node Features
The ComfyUI_pandai node provides the following features:

Text Generation: Generates text based on user input prompts.

Text Translation: Translates the generated text into English (or other languages).

Text Polishing: Refines the generated text to make it more fluent and natural.

Node Parameters
In ComfyUI, the Pandai DSK Node includes the following input parameters:

Required Parameters
API Key: The API key for DeepSeek. The default value is your_api_key_here. Replace it with your actual API key.

Model: The model to use. Currently, only deepseek-chat is supported.

Max Tokens: The maximum length of the generated text. The default value is 4096.

Temperature: Controls the randomness of the generated text. The default value is 1.

Top P: Controls the diversity of the generated text. The default value is 1.

Presence Penalty: Controls the avoidance of repetitive content. The default value is 0.

Frequency Penalty: Controls the avoidance of high-frequency words. The default value is 0.

System Prompt: A system prompt to guide the model. The default value is "You are a helpful assistant."

User Prompt: The user's input prompt for generating content.

Optional Parameters
Enable Translation: Whether to enable translation. The default value is disable.

Enable Polish: Whether to enable text polishing. The default value is disable.

History: Conversation history (optional).

Output
Generated Text: The original generated text.

Translated Text: The translated text (if translation is enabled).

Polished Text: The polished text (if polishing is enabled).

Polished ZH Text: The polished Chinese text (if polishing is enabled).

History: Updated conversation history.

Usage Examples
Text Generation:

Enter a prompt in the User Prompt field, e.g., "Write a short essay about artificial intelligence."

Set Max Tokens to 500 and Temperature to 0.8.

Run the node and view the generated text.

Text Translation:

Enable the Enable Translation option.

Enter a prompt, e.g., "Write a short essay about artificial intelligence."

Run the node and view the generated text along with the translated English text.

Text Polishing:

Enable the Enable Polish option.

Enter a prompt, e.g., "Write a short essay about artificial intelligence."

Run the node and view the generated text, translated text, and polished text.

Notes
API Key: Ensure you use a valid DeepSeek API key.

Dependencies: If the node does not work, check if all dependencies are installed.

Model Limitation: Currently, only the deepseek-chat model is supported.

Troubleshooting
Node Not Loading:

Ensure the ComfyUI_pandai folder is placed in the custom_nodes directory.

Ensure the __init__.py file exists and correctly exports NODE_CLASS_MAPPINGS.

API Call Failure:

Verify that the API key is correct.

Ensure your internet connection is stable.

Missing Dependencies:

Run pip install -r requirements.txt to install all dependencies.

Contribution and Feedback
If you encounter any issues or have suggestions for improvement, feel free to submit an Issue or Pull Request to the GitHub repository: ComfyUI_pandai.

I hope this guide helps! If you need further assistance or additional features, let me know! 😊
