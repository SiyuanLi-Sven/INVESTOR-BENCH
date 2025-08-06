# 🐛 故障排除指南

[返回文档索引](./README.md)

## 🚨 常见问题及解决方案

### 1. API调用相关问题

#### API密钥错误
```
Error code: 401 - Unauthorized
```

**解决方案**:
1. 检查config.py中的API密钥是否正确
2. 确认密钥格式正确（通常以"sk-"开头）
3. 验证密钥是否已过期或被禁用

```bash
# 测试API密钥有效性
python -c "
from openai import OpenAI
client = OpenAI(api_key='your-api-key', base_url='https://api.siliconflow.cn/v1')
try:
    response = client.models.list()
    print('✓ API密钥有效')
except Exception as e:
    print(f'✗ API密钥无效: {e}')
"
```

#### API参数无效错误
```
Error code: 400 - {'code': 20015, 'message': 'The parameter is invalid. Please check again.'}
```

**原因**: 通常是因为请求参数不符合API要求

**解决方案**:
1. 检查模型名称是否正确
2. 确认请求格式符合OpenAI API规范
3. 检查特殊参数如`response_format`是否被API支持

```python
# 修复示例：移除不支持的参数
request_params = {
    "model": self.model_config["model"],
    "messages": messages,
    "max_tokens": max_tokens,
    "temperature": temperature,
    # 注意：某些Provider可能不支持response_format
    # "response_format": {"type": "json_object"}  
}
```

#### 连接超时
```
Connection timeout
```

**解决方案**:
1. 检查网络连接
2. 增加timeout设置
3. 使用代理或VPN

```json
{
  "chat_config": {
    "chat_request_timeout": 120
  },
  "emb_config": {
    "embedding_timeout": 120
  }
}
```

### 2. 数据和配置问题

#### 日期范围错误
```
ValueError: start_date and end_date must be in env_data_pkl keys
```

**解决方案**:
检查数据文件的可用日期范围：

```bash
python -c "
import json
with open('data/jnj.json', 'r') as f:
    data = json.load(f)
dates = sorted(data.keys())
print(f'可用日期范围: {dates[0]} 到 {dates[-1]}')
"
```

更新配置文件中的日期设置：
```json
{
  "env_config": {
    "warmup_start_time": "2020-03-12",
    "warmup_end_time": "2020-09-30",
    "test_start_time": "2020-10-01",
    "test_end_time": "2021-05-06"
  }
}
```

#### 模型配置未找到
```
ValueError: Model XXX not found in configuration
```

**解决方案**:
1. 在config.py中添加模型配置
2. 检查模型名称拼写
3. 确认模型类型正确（llm_api或embedding_api）

```python
# 在config.py中添加模型
MODEL_CONFIGS = {
    "your-model-name": {
        "type": "llm_api",
        "model": "actual-model-name",
        "api_base": "https://api.example.com/v1",
        "api_key": "your-api-key",
        "provider": "provider-name"
    }
}
```

### 3. 依赖和导入问题

#### GuardRails导入失败
```
ImportError: cannot import name 'ValidChoices' from 'guardrails.hub'
```

**解决方案**:
这是已知的版本兼容问题，系统已自动处理：

```bash
# 检查guardrails版本
pip show guardrails-ai

# 如果需要降级
pip install guardrails-ai==0.4.5
```

#### 缺少依赖包
```
ModuleNotFoundError: No module named 'XXX'
```

**解决方案**:
```bash
# 重新安装依赖
pip install -r requirements.txt

# 或单独安装缺失的包
pip install missing-package-name
```

### 4. 数据库连接问题

#### Qdrant连接失败
```
Connection refused
```

**解决方案**:
1. 检查Qdrant服务是否运行
2. 验证端口配置
3. 重启Qdrant服务

```bash
# 检查服务状态
curl -X GET "http://localhost:6333/collections"

# 重启Docker容器
docker restart <qdrant_container_id>

# 或重新启动服务
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

#### Collection创建失败
```
Collection already exists
```

**解决方案**:
```bash
# 删除已存在的collection
curl -X DELETE "http://localhost:6333/collections/collection_name"

# 或清理所有数据
docker stop <qdrant_container_id>
docker rm <qdrant_container_id>
docker run -p 6333:6333 qdrant/qdrant
```

### 5. 内存和性能问题

#### 内存不足
```
OutOfMemoryError
```

**解决方案**:
1. 减少batch大小
2. 清理不必要的数据
3. 增加系统内存

```json
{
  "chat_config": {
    "chat_max_new_token": 500  // 减少token数量
  },
  "agent_config": {
    "top_k": 3  // 减少检索的记忆数量
  }
}
```

#### 运行速度慢
**优化方案**:
1. 使用本地VLLM推理
2. 调整API参数
3. 并行处理

```json
{
  "chat_config": {
    "chat_request_timeout": 30,
    "chat_parameters": {
      "temperature": 0.6
    }
  }
}
```

### 6. 文件权限问题

#### 无法创建目录
```
Permission denied
```

**解决方案**:
```bash
# 检查权限
ls -la results/

# 修改权限
chmod 755 results/
chmod -R 755 results/

# 或使用sudo创建
sudo mkdir -p results/experiment/model/symbol/
sudo chown $USER:$USER results/ -R
```

## 🔧 调试技巧

### 1. 启用详细日志

```bash
# 设置日志级别
export LOGURU_LEVEL=TRACE

# 运行时查看实时日志
tail -f results/experiment/model/symbol/log/warmup.log
```

### 2. 分步调试

```python
# 手动测试组件
python -c "
from src import MarketEnv, FinMemAgent
from config import get_model_config

# 测试市场环境
env = MarketEnv(
    symbol=['JNJ'],
    env_data_path={'JNJ': 'data/jnj.json'},
    start_date='2020-03-12',
    end_date='2020-03-20',
    momentum_window_size=3
)
print('✓ MarketEnv初始化成功')

# 测试模型配置
config = get_model_config('Qwen/Qwen3-8B')
print('✓ 模型配置获取成功')
"
```

### 3. API测试

```python
# 独立测试API调用
from src.chat.endpoint.openai_compatible import OpenAICompatibleClient

client = OpenAICompatibleClient('Qwen/Qwen3-8B')
response = client.chat_completion([
    {"role": "user", "content": "Hello, test message"}
], max_tokens=50)
print(response)
```

### 4. 内存使用监控

```bash
# 监控内存使用
python -c "
import psutil
import os
import time

while True:
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f'内存使用: {memory_mb:.1f} MB')
    time.sleep(5)
"
```

## 🆘 获取帮助

### 日志文件位置

```
results/<run_name>/<model>/<symbols>/log/
├── warmup.log          # Warmup阶段日志
├── warmup_trace.log    # 详细trace日志
├── test.log           # Test阶段日志
└── test_trace.log     # Test阶段trace日志
```

### 常用检查命令

```bash
# 1. 系统状态检查
python test_simplified.py

# 2. API连接测试  
python test_openai_compatible.py

# 3. 依赖检查
python -c "
import openai, qdrant_client, loguru, pandas, numpy
print('✓ 核心依赖正常')
"

# 4. 配置验证
python -c "
import json
config = json.load(open('configs/main_unified.json'))
print(f'✓ 配置文件有效: {config[\"meta_config\"][\"run_name\"]}')
"

# 5. 数据检查
python -c "
import json
with open('data/jnj.json') as f:
    data = json.load(f)
print(f'✓ 数据文件有效，包含 {len(data)} 天的数据')
"
```

### 紧急恢复

如果系统完全无法运行：

```bash
# 1. 清理所有结果
rm -rf results/*

# 2. 重置Qdrant
docker stop $(docker ps -q --filter "ancestor=qdrant/qdrant")
docker run -d -p 6333:6333 qdrant/qdrant

# 3. 重新安装依赖
pip install --force-reinstall -r requirements.txt

# 4. 运行基础测试
python test_simplified.py
```

## 📞 技术支持

如果问题仍然存在：

1. **收集信息**:
   - 完整的错误消息
   - 使用的配置文件
   - 系统环境信息
   - 相关日志文件

2. **提供上下文**:
   - 具体的操作步骤
   - 预期结果vs实际结果
   - 是否在其他环境中工作正常

3. **尝试最小复现**:
   - 创建最简配置复现问题
   - 隔离问题组件
   - 记录详细步骤

---

[← 上一章: CLI参考](./10-cli-reference.md) | [下一章: 开发指南 →](./12-development-guide.md)