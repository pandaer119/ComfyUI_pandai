# ComfyUI_pandai 多协议版

## 新增特性
✅ 双协议支持：同时兼容OpenAI官方API和火山引擎DeepSeek-R1接口  
✅ 智能路由：自动选择响应最快的服务节点  
✅ 安全签名：符合火山引擎V4签名算法规范  

## 火山引擎专属配置
```bash
# 在.env文件中添加
VOLC_ACCESS_KEY=your_ak_from_console
VOLC_SECRET_KEY=your_sk_from_console
MODEL_NAME=deepseek-r1 # 可选r1或r1-turbo
REGION=cn-north-1 # 必须与控制台区域一致
