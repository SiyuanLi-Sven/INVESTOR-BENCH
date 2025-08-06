# 统一OpenAI兼容API接口使用指南

## 🎯 项目重构概述

本次重构将INVESTOR-BENCH项目中的所有LLM和Embedding API调用统一为OpenAI兼容格式，实现了：

✅ **统一配置管理** - 所有模型配置集中在`config.py`  
✅ **OpenAI兼容接口** - 支持OpenAI、VLLM、硅基流动等多种Provider  
✅ **简化API调用** - 统一的调用方式，无需关心底层实现  
✅ **向后兼容** - 保持原有接口不变  
✅ **灵活扩展** - 易于添加新的模型和Provider  

## 📁 文件结构

```
├── config.py                           # 🔥 统一配置文件
├── src/
│   ├── chat/endpoint/
│   │   └── openai_compatible.py        # 🔥 OpenAI兼容LLM客户端  
│   ├── embedding_unified.py            # 🔥 统一Embedding客户端
│   └── embedding.py                    # ✅ 更新为使用统一接口
├── configs/
│   └── main_unified.json               # 🔥 统一配置文件示例
├── test_simplified.py                  # ✅ 基础功能测试
├── test_openai_compatible.py           # ✅ OpenAI兼容测试
└── demo_config.json                    # ✅ 示例配置
```

## ⚙️ 配置说明

### 1. 模型配置 (config.py)

```python
MODEL_CONFIGS = {
    # 硅基流动模型 (推荐)
    "Qwen/Qwen3-8B": {
        "type": "llm_api",
        "model": "Qwen/Qwen3-8B", 
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "your-siliconflow-api-key",
        "provider": "siliconflow"
    },
    
    # 本地VLLM模型
    "local-vllm": {
        "type": "llm_api",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "api_base": "http://0.0.0.0:8000/v1",
        "api_key": "EMPTY",
        "provider": "vllm"
    },
    
    # OpenAI模型 (取消注释并配置密钥)
    # "gpt-4": {
    #     "type": "llm_api", 
    #     "model": "gpt-4",
    #     "api_base": "https://api.openai.com/v1",
    #     "api_key": "your-openai-api-key",
    #     "provider": "openai"
    # }
}
```

### 2. 运行配置 (configs/main_unified.json)

```json
{
  "chat_config": {
    "chat_model": "Qwen/Qwen3-8B",                    // 🔥 使用config.py中定义的模型
    "chat_model_inference_engine": "openai_compatible", // 🔥 使用统一接口
    "chat_system_message": "You are a helpful assistant.",
    "chat_parameters": {
      "temperature": 0.6
    },
    "chat_max_new_token": 1000,
    "chat_request_timeout": 60
  },
  "emb_config": {
    "emb_model_name": "Qwen/Qwen3-Embedding-4B",     // 🔥 使用config.py中定义的Embedding模型
    "embedding_timeout": 60
  }
}
```

## 🚀 快速开始

### Step 1: 配置API密钥

在`config.py`中配置您的API密钥：

```python
# 将YOUR_API_KEY替换为实际的API密钥
"api_key": "sk-your-actual-api-key-here"
```

### Step 2: 选择运行配置

使用`configs/main_unified.json`或创建自己的配置文件：

```bash
# 使用统一配置运行
python run.py --config configs/main_unified.json
```

### Step 3: 验证配置

```bash
# 运行基础测试
python test_simplified.py

# 运行详细测试  
python test_openai_compatible.py
```

## 🔧 支持的Provider

### 1. 硅基流动 (推荐)

```python
"Qwen/Qwen3-8B": {
    "type": "llm_api",
    "model": "Qwen/Qwen3-8B",
    "api_base": "https://api.siliconflow.cn/v1",
    "api_key": "sk-your-siliconflow-key", 
    "provider": "siliconflow"
}
```

### 2. 本地VLLM

```python  
"local-vllm": {
    "type": "llm_api",
    "model": "your-local-model-name",
    "api_base": "http://localhost:8000/v1",
    "api_key": "EMPTY",
    "provider": "vllm"
}
```

### 3. OpenAI官方

```python
"gpt-4": {
    "type": "llm_api",
    "model": "gpt-4",
    "api_base": "https://api.openai.com/v1", 
    "api_key": "sk-your-openai-key",
    "provider": "openai"
}
```

### 4. Anthropic (Claude)

```python
"claude-3-sonnet": {
    "type": "llm_api", 
    "model": "claude-3-sonnet-20240229",
    "api_base": "https://api.anthropic.com/v1",
    "api_key": "your-anthropic-key",
    "provider": "anthropic"
}
```

## 🔄 迁移指南

### 从旧配置迁移

1. **原有的VLLM配置**:
   ```json
   // 旧方式
   "chat_model_inference_engine": "vllm"
   
   // 新方式 (推荐)
   "chat_model_inference_engine": "openai_compatible"
   "chat_model": "local-vllm"  // 在config.py中定义
   ```

2. **原有的OpenAI配置**:
   ```json
   // 旧方式
   "chat_model_inference_engine": "openai"
   
   // 新方式 (推荐)  
   "chat_model_inference_engine": "openai_compatible"
   "chat_model": "gpt-4"  // 在config.py中定义
   ```

### 向后兼容

✅ 原有的配置方式仍然支持  
✅ 现有代码无需修改  
✅ 可以逐步迁移到新接口  

## 🐛 故障排除

### 常见问题

1. **导入错误**: 
   ```
   ImportError: cannot import name 'ValidChoices'
   ```
   **解决**: 这是guardrails版本兼容问题，不影响核心功能使用

2. **API密钥错误**:
   ```
   OpenAI API error: Invalid API key
   ```
   **解决**: 检查config.py中的API密钥配置

3. **连接超时**:
   ```
   Connection timeout
   ```
   **解决**: 检查api_base URL是否正确，调整timeout设置

### 测试命令

```bash
# 基础功能测试
python test_simplified.py

# 详细功能测试
python test_openai_compatible.py

# 语法检查
python -m py_compile config.py
python -m py_compile src/embedding_unified.py
```

## 📈 性能优化

### 推荐配置

1. **生产环境**: 使用硅基流动API (稳定性好，价格便宜)
2. **开发环境**: 使用本地VLLM (节省成本，响应快)
3. **高质量任务**: 使用OpenAI GPT-4 (质量最高)

### 参数调优

```json
{
  "chat_parameters": {
    "temperature": 0.6,        // 创造性 vs 确定性平衡
    "max_tokens": 1000,        // 根据任务需求调整
    "top_p": 0.9,             // 词汇多样性控制
    "frequency_penalty": 0.0   // 重复内容惩罚
  }
}
```

## 🔮 扩展指南

### 添加新Provider

1. 在`config.py`中添加模型配置
2. 确保API遵循OpenAI格式
3. 测试连接和响应格式
4. 更新配置文档

### 自定义客户端

```python
from src.chat.endpoint.openai_compatible import OpenAICompatibleClient

# 创建自定义客户端
client = OpenAICompatibleClient("your-model-name")

# 调用API
response = client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    max_tokens=100,
    temperature=0.7
)
```

## 📞 技术支持

如果遇到问题：

1. 先运行测试脚本确认配置
2. 检查API密钥和网络连接  
3. 查看日志输出定位问题
4. 参考本文档的故障排除部分

---

🎉 **恭喜！您已成功迁移到统一的OpenAI兼容API系统！**

现在可以享受统一、简洁、高效的AI模型调用体验了！