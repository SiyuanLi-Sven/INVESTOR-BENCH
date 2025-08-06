# 故障排除指南

本文档提供了INVESTOR-BENCH项目中常见问题的诊断和解决方案。

## 🚨 常见问题

### 1. API超时错误

#### 问题症状
```
❌ 运行失败: The callable `fn` passed to `Guard(fn, ...)` failed with the following error: `The read operation timed out`.
```

#### 问题原因
- API响应时间过长（通常超过5-16分钟）
- 网络连接不稳定
- API服务端负载过高
- 请求数据过于复杂

#### 解决方案

##### 方案1: 调整超时设置
修改`run_enhanced.py`中的超时配置：
```python
"chat_request_timeout": 120,  # 改为2分钟
"embedding_timeout": 60       # 改为1分钟
```

##### 方案2: 使用重试机制
新版本已内置重试机制，会自动重试失败的请求：
- 最大重试次数：3次
- 重试间隔：5秒
- 学术原则：所有重试失败后直接报错，保持实验真实性

##### 方案3: 检查API服务状态
```bash
# 测试API连通性
curl -I https://api.siliconflow.cn/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# 检查模型可用性
curl https://api.siliconflow.cn/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY" | jq '.data[].id'
```

##### 方案4: 切换API提供商
如果SiliconFlow不稳定，可以切换到其他提供商：
```bash
# 使用OpenAI官方API
python run_enhanced.py both --symbol JNJ \
  --start-date 2020-03-12 --end-date 2020-03-13 \
  --api-base https://api.openai.com/v1 \
  --model gpt-4o-mini

# 使用本地vLLM
python run_enhanced.py both --symbol JNJ \
  --start-date 2020-03-12 --end-date 2020-03-13 \
  --api-base http://localhost:8000/v1 \
  --model Qwen/Qwen3-8B
```

### 2. 网络连接问题

#### 问题症状
```
❌ 网络连接失败: Connection timeout
```

#### 解决方案
1. **检查网络连接**：
   ```bash
   ping api.siliconflow.cn
   ```

2. **配置代理**（如果在企业网络环境）：
   ```bash
   export HTTP_PROXY=http://proxy.company.com:8080
   export HTTPS_PROXY=http://proxy.company.com:8080
   ```

3. **使用本地模型**避免网络依赖

### 3. API密钥问题

#### 问题症状
```
❌ API配置错误: OPENAI_API_KEY environment variable is required
```

#### 解决方案
1. **检查环境变量**：
   ```bash
   echo $OPENAI_API_KEY
   ```

2. **设置API密钥**：
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   export OPENAI_API_BASE="https://api.siliconflow.cn/v1"
   ```

3. **使用.env文件**：
   ```bash
   echo 'OPENAI_API_KEY="sk-your-key-here"' > .env
   echo 'OPENAI_API_BASE="https://api.siliconflow.cn/v1"' >> .env
   ```

### 4. 内存不足问题

#### 问题症状
```
❌ RuntimeError: CUDA out of memory
```

#### 解决方案
1. **减少批处理大小**
2. **使用更小的模型**
3. **清理GPU内存**：
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### 5. 数据文件问题

#### 问题症状
```
❌ 文件未找到: data/jnj.json
```

#### 解决方案
1. **检查文件存在性**：
   ```bash
   ls -la data/
   ```

2. **验证JSON格式**：
   ```bash
   python -m json.tool data/jnj.json
   ```

## 🔧 调试技巧

### 启用详细日志
```bash
python run_enhanced.py both --symbol JNJ \
  --start-date 2020-03-12 --end-date 2020-03-13 \
  --verbose
```

### 检查系统资源
```bash
# 内存使用
free -h

# GPU使用（如果有）
nvidia-smi

# 磁盘空间
df -h
```

### 测试API连接
创建`test_api.py`：
```python
import openai
import os

client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_API_BASE")
)

try:
    response = client.chat.completions.create(
        model="Qwen/Qwen3-8B",
        messages=[{"role": "user", "content": "Hello"}],
        timeout=30
    )
    print("✅ API连接正常")
    print(f"响应: {response.choices[0].message.content}")
except Exception as e:
    print(f"❌ API连接失败: {e}")
```

## 📊 性能优化建议

### 1. 减少数据范围
对于测试，使用较短的时间范围：
```bash
# 只测试2天
python run_enhanced.py both --symbol JNJ \
  --start-date 2020-03-12 --end-date 2020-03-13 \
  --test-start-date 2020-03-16 --test-end-date 2020-03-17
```

### 2. 使用更快的模型
```bash
# 使用更小更快的模型
--model Qwen/Qwen3-4B  # 而不是Qwen3-8B
```

### 3. 并行处理注意事项
- 避免同时运行多个实例
- 确保足够的内存和GPU资源

## 🆘 获取帮助

如果问题仍然存在：

1. **查看完整日志**：
   ```bash
   tail -f results/*/run.log
   ```

2. **提供错误报告**时包含：
   - 完整的错误消息
   - 运行命令
   - 系统信息（OS、Python版本等）
   - API提供商和模型信息

3. **临时解决方案**：
   - 使用更短的时间范围测试
   - 切换到更稳定的API提供商
   - 使用本地模型避免网络问题

## 🔄 恢复运行

如果运行中断，可以从checkpoint恢复：

```bash
# 只运行test阶段（如果warmup已完成）
python run_enhanced.py test --symbol JNJ \
  --test-start-date 2020-03-16 --test-end-date 2020-03-17
```

系统会自动寻找最新的warmup结果进行继续。