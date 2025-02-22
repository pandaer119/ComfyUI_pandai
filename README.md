# ComfyUI_pandai

![workflow1](https://github.com/user-attachments/assets/c501bd3f-c905-41d5-8405-8ae82d9ffe80)

## 核心改进说明

1. **JSON结构验证**  
   - 新增`_build_messages`方法确保消息结构符合API规范
   - 支持字符串和字典两种content格式
   - 自动处理历史消息的深拷贝

2. **错误处理优化**  
   - 所有API调用添加try-catch保护
   - 分离生成、翻译、润色逻辑到独立方法
   - 新增安全调用方法`_call_api_safely`

3. **输出改进**  
   - 明确返回四个输出端口
   - 新增translated_text独立输出
   - 优化polish提示词模板

## 安装指南（保持不变）
（原有安装步骤保持不变，此处省略以节省篇幅）