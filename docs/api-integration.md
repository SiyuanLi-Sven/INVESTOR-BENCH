# API集成指南

完整的OpenAI兼容API集成指南，支持多种LLM服务提供商。

## 🌐 支持的API提供商

### 已验证兼容的服务商

| 提供商 | Chat API | Embedding API | 模型示例 | 状态 |
|--------|----------|---------------|----------|------|
| **SiliconFlow** | ✅ | ✅ | Qwen3-8B, Qwen3-Embedding-4B | 已测试 |
| **OpenAI** | ✅ | ✅ | GPT-4, text-embedding-3-large | 已测试 |
| **Anthropic** | ✅ | ❌ | Claude-3.5-Sonnet | 需外部embedding |
| **本地部署** | ✅ | ✅ | 通过vLLM/Ollama等 | 支持 |

## 🔧 配置方法

### 1. SiliconFlow API配置

SiliconFlow提供完整的OpenAI兼容API，是推荐的选择。

#### 环境变量配置
```bash
# .env文件
OPENAI_API_KEY="sk-your-siliconflow-key"
OPENAI_API_BASE="https://api.siliconflow.cn/v1"
OPENAI_MODEL="Qwen/Qwen3-8B"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B" 
```

#### PKL配置文件
```pkl
// configs/chat_models.pkl 中添加
qwen3_8b_siliconflow: ChatModelConfig = new {
    chat_model = "Qwen/Qwen3-8B"
    chat_model_type = "chat"
    chat_model_inference_engine = "openai"
    chat_endpoint = "https://api.siliconflow.cn/v1/chat/completions"
    chat_system_message = "You are a helpful assistant."
    chat_parameters = new Mapping {
        ["temperature"] = 0.7
        ["max_tokens"] = 1024
    }
    chat_request_sleep = new Mapping {
        ["sleep_time"] = 2
        ["sleep_every_count"] = 10
    }
}
```

```pkl  
// configs/embedding.pkl 中添加
qwen3_embedding_4b_siliconflow: EmbEndpointConfig = new {
    emb_model_name = "Qwen/Qwen3-Embedding-4B"
    request_endpoint = "https://api.siliconflow.cn/v1/embeddings"
    emb_size = 2560  // 实际向量维度
}
```

### 2. OpenAI官方API配置

#### 环境变量
```bash
OPENAI_API_KEY="sk-your-openai-key"
OPENAI_API_BASE="https://api.openai.com/v1"  # 可选，默认值
OPENAI_MODEL="gpt-4o"
EMBEDDING_MODEL="text-embedding-3-large"
```

#### 使用已有配置
```pkl
// 在main.pkl中使用
chat_model = "gpt-4o"
embedding_model = "text-embedding-3-large"
```

### 3. Claude API配置

Claude需要单独的embedding服务：

```pkl
claude_sonnet_35: ChatModelConfig = new {
    chat_model = "claude-3-5-sonnet-20240620"
    chat_model_type = "chat"
    chat_model_inference_engine = "anthropic"
    chat_endpoint = "https://api.anthropic.com/v1/messages"
    chat_parameters = new Mapping {
        ["max_tokens"] = 4096
    }
    chat_request_sleep = new Mapping {
        ["sleep_time"] = 60
        ["sleep_every_count"] = 5
    }
}
```

**注意**: 使用Claude时，embedding需要配置OpenAI或其他服务。

### 4. 本地部署API配置

#### 使用vLLM部署
```bash
# 启动vLLM服务
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8000 \
    --served-model-name Qwen3-8B
```

```pkl
local_qwen: ChatModelConfig = new {
    chat_model = "Qwen3-8B"
    chat_model_type = "chat"  
    chat_model_inference_engine = "openai"
    chat_endpoint = "http://localhost:8000/v1/chat/completions"
    chat_parameters = new Mapping {
        ["temperature"] = 0.7
    }
}
```

## 🧪 API测试方法

### 1. 连通性测试

使用提供的测试脚本：

```bash
# 测试Chat API
python test_api.py

# 测试Embedding API  
python test_embedding.py
```

### 2. 手动测试

#### Chat API测试
```bash
curl -X POST "https://api.siliconflow.cn/v1/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [
      {"role": "user", "content": "分析一下苹果公司的投资价值"}
    ],
    "temperature": 0.7
  }'
```

#### Embedding API测试
```bash
curl -X POST "https://api.siliconflow.cn/v1/embeddings" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-4B",
    "input": "苹果公司股价上涨"
  }'
```

## ⚙️ 高级配置

### 1. 请求参数优化

#### Chat API参数
```pkl
chat_parameters = new Mapping {
    ["temperature"] = 0.6        // 控制随机性，0.6-0.8适合金融分析
    ["max_tokens"] = 1024        // 响应长度，1024足够大多数决策
    ["top_p"] = 0.9             // 核采样参数
    ["frequency_penalty"] = 0.1  // 避免重复内容
    ["presence_penalty"] = 0.1   // 鼓励主题多样性
}
```

#### 请求控制参数
```pkl
chat_request_sleep = new Mapping {
    ["sleep_time"] = 2           // 请求间隔秒数
    ["sleep_every_count"] = 10   // 每N个请求后休息
}
chat_request_timeout = 30        // 请求超时时间
```

### 2. 错误处理和重试

系统内置了智能重试机制：

```python
# 自动重试逻辑
max_retries = 3
retry_delays = [1, 3, 5]  # 递增延迟

# 错误类型处理
- 429 (Rate Limit): 自动等待并重试
- 500 (Server Error): 重试3次
- 401 (Auth Error): 立即失败，检查API key
- Timeout: 重试，增加超时时间
```

### 3. 成本控制

#### 估算API成本
```python
# 估算公式 (以SiliconFlow为例)
daily_decisions = 252  # 一年交易日
tokens_per_decision = 2000  # 每次决策约2K tokens
annual_tokens = daily_decisions * tokens_per_decision * 1000  # 总tokens

# 成本 = annual_tokens * price_per_1k_tokens
```

#### 成本优化策略
1. **批处理**: 合并相似请求
2. **缓存**: 避免重复分析相同内容
3. **精简Prompt**: 减少不必要的上下文
4. **智能重试**: 避免无效重试

### 4. 性能优化

#### 并发控制
```pkl
// 控制并发请求数
chat_parameters = new Mapping {
    ["max_concurrent_requests"] = 5  // 最大并发数
    ["request_queue_size"] = 20      // 请求队列大小
}
```

#### 连接池优化
```python
# HTTP连接池配置
httpx.Client(
    timeout=30,
    limits=httpx.Limits(
        max_keepalive_connections=10,
        max_connections=100
    )
)
```

## 🔍 故障诊断

### 常见问题和解决方案

#### 1. API调用失败
```
错误: OpenAIEmbedding failed with unknown error
```

**诊断步骤**:
```bash
# 1. 检查API key
echo $OPENAI_API_KEY

# 2. 测试连通性
curl -I https://api.siliconflow.cn/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# 3. 检查模型名称
curl https://api.siliconflow.cn/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[].id'
```

#### 2. 向量维度不匹配
```
错误: Expected vector dimension 1536, got 2560
```

**解决方法**:
1. 测试实际向量维度: `python test_embedding.py`
2. 更新配置中的`emb_size`参数
3. 重新生成配置: `pkl eval -f json -o configs/main.json configs/main.pkl`

#### 3. 请求限流
```
错误: 429 Too Many Requests
```

**解决方法**:
```pkl
// 增加请求间隔
chat_request_sleep = new Mapping {
    ["sleep_time"] = 5           // 增加到5秒
    ["sleep_every_count"] = 5    // 减少到5次
}
```

### 4. 模型兼容性问题

#### Guardrails兼容性
某些模型可能不支持严格的JSON输出格式：

```python
# 临时解决方案：使用mock验证器
try:
    from guardrails.hub import ValidChoices
except ImportError:
    from .mock_validators import ValidChoices
```

## 📊 性能基准

### API响应时间基准 (毫秒)

| 提供商 | Chat API | Embedding API | 稳定性 |
|--------|----------|---------------|--------|
| SiliconFlow | 800-1500 | 200-400 | 优秀 |
| OpenAI | 1000-2000 | 100-300 | 优秀 |
| 本地vLLM | 500-1000 | 50-150 | 依赖硬件 |

### 并发性能测试

```bash
# 并发测试脚本
for i in {1..10}; do
    python test_api.py &
done
wait
```

建议并发数：
- SiliconFlow: 5-10个并发
- OpenAI: 10-20个并发 (根据plan)
- 本地部署: 取决于硬件资源

这个配置指南可以帮助你快速集成各种API提供商，确保系统稳定高效运行。