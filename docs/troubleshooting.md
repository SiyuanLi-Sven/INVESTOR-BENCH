# 故障排除指南

完整的问题诊断和解决方案，帮助快速定位和解决系统运行中的各种问题。

## 🔍 快速诊断检查表

运行遇到问题时，按以下顺序检查：

### 1. 基础环境检查 (30秒)
```bash
# ✅ Python版本
python --version  # 应为3.8+

# ✅ 依赖包状态  
pip list | grep -E "(guardrails|qdrant|httpx|loguru|numpy)"

# ✅ Docker服务
docker ps | grep qdrant

# ✅ 配置文件
ls -la configs/main.json
```

### 2. API连通性检查 (1分钟)
```bash
# ✅ 环境变量
echo $OPENAI_API_KEY | head -c 20

# ✅ Chat API
python test_api.py

# ✅ Embedding API  
python test_embedding.py
```

### 3. 系统状态检查 (30秒)
```bash
# ✅ Qdrant健康状态
curl -s http://localhost:6333/health

# ✅ 数据文件存在
ls -la data/

# ✅ 结果目录权限
ls -la results/
```

## 🚨 常见错误及解决方案

### 1. API相关错误

#### 错误: `OpenAIEmbedding failed with unknown error`

**可能原因**:
- API密钥无效或过期
- 网络连接问题
- 模型名称不正确
- API配额耗尽

**诊断步骤**:
```bash
# 1. 验证API密钥
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.siliconflow.cn/v1/models

# 2. 检查模型列表
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.siliconflow.cn/v1/models | jq '.data[].id'

# 3. 测试embedding API
curl -X POST https://api.siliconflow.cn/v1/embeddings \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-Embedding-4B", "input": "test"}'
```

**解决方案**:
```bash
# 方案1: 更新API密钥
export OPENAI_API_KEY="sk-new-key-here"

# 方案2: 修改模型配置
pkl eval -e 'configs/main.pkl.amend { 
    embedding_model = "text-embedding-3-small" 
}' -f json -o configs/main.json

# 方案3: 增加重试延迟
pkl eval -e 'configs/chat_models.pkl.amend { 
    qwen3_8b_siliconflow.chat_request_sleep.sleep_time = 5 
}' -f json -o configs/main.json
```

#### 错误: `429 Too Many Requests`

**解决方案**:
```pkl
// 在chat_models.pkl中增加请求间隔
chat_request_sleep = new Mapping {
    ["sleep_time"] = 10           // 增加到10秒
    ["sleep_every_count"] = 3     // 每3次请求休息
}
```

#### 错误: `401 Unauthorized`

**检查步骤**:
```bash
# 1. 确认API密钥格式
echo $OPENAI_API_KEY | grep -E "^sk-[a-zA-Z0-9]{32,}$"

# 2. 检查API基础URL
echo $OPENAI_API_BASE

# 3. 测试基础连接
curl -I $OPENAI_API_BASE/models
```

### 2. 配置相关错误

#### 错误: `pkl eval` 语法错误

**常见问题**:
```pkl
// ❌ 错误写法
trading_symbols = ["AAPL", "GOOGL"]  // 使用了JSON语法

// ✅ 正确写法  
trading_symbols = new Listing {
    "AAPL"
    "GOOGL"
}
```

**验证命令**:
```bash
# 语法检查
pkl eval configs/main.pkl > /dev/null

# 类型检查
pkl eval -m configs/meta.pkl configs/main.pkl
```

#### 错误: `chat_model not in chat_model_dict`

**解决方案**:
```bash
# 1. 查看可用模型
pkl eval -p chat_models.chat_model_dict configs/chat_models.pkl

# 2. 添加新模型配置
echo 'your_model: ChatModelConfig = new { ... }' >> configs/chat_models.pkl

# 3. 更新模型字典
echo '["your-model"] = your_model' >> configs/chat_models.pkl
```

### 3. 数据库相关错误

#### 错误: `Connection refused to localhost:6333`

**诊断步骤**:
```bash
# 1. 检查Qdrant状态
docker ps -a | grep qdrant

# 2. 查看Qdrant日志
docker logs qdrant

# 3. 检查端口占用
netstat -an | grep 6333
lsof -i :6333
```

**解决方案**:
```bash
# 方案1: 重启Qdrant
docker restart qdrant

# 方案2: 重新创建容器
docker rm -f qdrant
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# 方案3: 修改端口配置
docker run -d -p 6334:6333 --name qdrant qdrant/qdrant
# 然后在configs/meta.pkl中修改:
# memory_db_endpoint = "http://localhost:6334"
```

#### 错误: `Collection already exists`

**解决方案**:
```bash
# 清理所有集合
curl -X DELETE http://localhost:6333/collections/JNJ_short
curl -X DELETE http://localhost:6333/collections/JNJ_mid
curl -X DELETE http://localhost:6333/collections/JNJ_long
curl -X DELETE http://localhost:6333/collections/JNJ_reflection

# 或重置整个数据库
docker restart qdrant
```

### 4. 内存和性能问题

#### 错误: `MemoryError` 或系统卡顿

**诊断内存使用**:
```bash
# 系统内存
free -h
top -p $(pgrep -f "python run.py")

# Python内存分析
python -m memory_profiler run.py warmup
```

**优化方案**:
```pkl
// 减少内存使用参数
top_k = 3                    // 从5减少到3
look_back_window_size = 2    // 从3减少到2

// 增加记忆清理频率
clean_up_importance_threshold = 10.0  // 从5.0增加到10.0
```

#### 性能过慢

**性能分析**:
```bash
# 代码性能分析
python -m cProfile -o profile.prof run.py warmup
python -c "import pstats; pstats.Stats('profile.prof').sort_stats('tottime').print_stats(20)"

# API调用统计
grep -c "LLM API Request" results/exp/*/JNJ/log/*.log
grep -c "EMB API Request" results/exp/*/JNJ/log/*.log
```

**优化策略**:
```bash
# 1. 启用批处理
export BATCH_EMBEDDING=true

# 2. 使用缓存
export ENABLE_CACHE=true

# 3. 减少API调用
pkl eval -e 'configs/main.pkl.amend {
    chat_parameters.temperature = 0.1  // 减少随机性，提高缓存命中
}' -f json -o configs/main.json
```

### 5. 依赖包问题

#### 错误: `ImportError: cannot import name 'ValidChoices'`

**解决方案**:
```bash
# 1. 检查guardrails版本
pip show guardrails-ai

# 2. 重新安装
pip uninstall guardrails-ai
pip install guardrails-ai==0.5.13

# 3. 使用mock版本(已内置)
# 系统会自动fallback到mock_validators.py
```

#### 错误: `numpy version conflict`

**解决方案**:
```bash
# 降级numpy
pip install "numpy<2.0"

# 检查冲突包
pip check

# 创建干净环境
conda create -n investor-bench python=3.11
conda activate investor-bench
pip install -r requirements.txt
```

### 6. 数据相关错误

#### 错误: `FileNotFoundError: data/tsla.json`

**检查数据文件**:
```bash
# 查看可用数据
ls -la data/

# 检查配置中的symbols
pkl eval -p env_config.trading_symbols configs/main.json
```

**解决方案**:
```bash
# 方案1: 修改为可用symbol
pkl eval -e 'configs/main.pkl.amend { 
    trading_symbols = new Listing { "JNJ" } 
}' -f json -o configs/main.json

# 方案2: 添加缺失数据文件(需要数据源)
# cp template.json data/tsla.json
```

#### 错误: `JSON decode error in data file`

**修复数据文件**:
```bash
# 验证JSON格式
python -m json.tool data/jnj.json > /dev/null

# 查看错误位置
python -c "
import json
try:
    with open('data/jnj.json') as f:
        json.load(f)
except json.JSONDecodeError as e:
    print(f'Error at line {e.lineno}, column {e.colno}: {e.msg}')
"
```

## 🔧 高级诊断工具

### 日志分析工具

**创建日志分析脚本**:
```bash
#!/bin/bash
# analyze_logs.sh

LOG_DIR="results/exp/*/JNJ/log"

echo "=== API调用统计 ==="
grep -c "LLM API Request" $LOG_DIR/*.log
grep -c "EMB API Request" $LOG_DIR/*.log

echo "=== 错误统计 ==="
grep -c "ERROR" $LOG_DIR/*.log
grep -c "WARNING" $LOG_DIR/*.log

echo "=== 最近错误 ==="
grep "ERROR" $LOG_DIR/*.log | tail -10

echo "=== 性能指标 ==="
grep "processing time" $LOG_DIR/*.log | tail -5
```

### 系统健康检查脚本

```python
#!/usr/bin/env python3
# health_check.py

import os
import json
import subprocess
import requests
from pathlib import Path

def check_environment():
    """检查环境配置"""
    checks = {
        'OPENAI_API_KEY': bool(os.getenv('OPENAI_API_KEY')),
        'Python Version': subprocess.run(['python', '--version'], capture_output=True).returncode == 0,
        'Docker': subprocess.run(['docker', 'ps'], capture_output=True).returncode == 0,
        'Config File': Path('configs/main.json').exists(),
        'Data Directory': Path('data').exists(),
    }
    
    for check, status in checks.items():
        print(f"{'✅' if status else '❌'} {check}")
    
    return all(checks.values())

def check_services():
    """检查服务状态"""
    checks = {}
    
    # Qdrant健康检查
    try:
        response = requests.get('http://localhost:6333/health', timeout=5)
        checks['Qdrant'] = response.status_code == 200
    except:
        checks['Qdrant'] = False
    
    # API连通性检查
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        try:
            response = requests.get(
                'https://api.siliconflow.cn/v1/models',
                headers={'Authorization': f'Bearer {api_key}'},
                timeout=10
            )
            checks['SiliconFlow API'] = response.status_code == 200
        except:
            checks['SiliconFlow API'] = False
    
    for check, status in checks.items():
        print(f"{'✅' if status else '❌'} {check}")
    
    return all(checks.values())

def check_data_integrity():
    """检查数据完整性"""
    with open('configs/main.json') as f:
        config = json.load(f)
    
    symbols = config['env_config']['trading_symbols']
    
    missing_files = []
    for symbol in symbols:
        data_file = f"data/{symbol.lower()}.json"
        if not Path(data_file).exists():
            missing_files.append(data_file)
    
    if missing_files:
        print(f"❌ 缺失数据文件: {missing_files}")
        return False
    else:
        print(f"✅ 所有数据文件存在")
        return True

if __name__ == "__main__":
    print("🔍 INVESTOR-BENCH 系统健康检查\n")
    
    env_ok = check_environment()
    print()
    
    services_ok = check_services()
    print()
    
    data_ok = check_data_integrity()
    print()
    
    if env_ok and services_ok and data_ok:
        print("🎉 系统状态良好，可以开始运行！")
    else:
        print("⚠️ 发现问题，请根据上述检查结果进行修复")
```

### 自动修复脚本

```bash
#!/bin/bash
# auto_fix.sh

echo "🔧 开始自动修复..."

# 1. 重启服务
echo "重启Qdrant..."
docker restart qdrant
sleep 5

# 2. 清理缓存
echo "清理Python缓存..."
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name "*.pyc" -delete

# 3. 重新生成配置
echo "重新生成配置..."
pkl eval -f json -o configs/main.json configs/main.pkl

# 4. 测试API连接
echo "测试API连接..."
python test_api.py

echo "✅ 自动修复完成"
```

## 📞 获取帮助

### 1. 查看详细日志
```bash
# 完整日志
tail -f results/exp/*/JNJ/log/*.log

# 只看错误
grep -E "(ERROR|CRITICAL)" results/exp/*/JNJ/log/*.log

# 特定时间段
grep "2020-07-02" results/exp/*/JNJ/log/*.log
```

### 2. 社区资源
- 查看[API集成文档](./api-integration.md)了解API配置
- 参考[快速开始](./quick-start.md)确认基础配置
- 阅读[架构文档](./architecture.md)理解系统原理

### 3. 提交Issue
如果以上方法都无法解决问题，请提交Issue并包含：
- 错误信息和完整堆栈跟踪
- 系统环境信息(`python --version`, `pip list`)
- 配置文件内容(`configs/main.json`)
- 相关日志片段

这个故障排除指南可以帮助解决99%的常见问题，让系统快速恢复正常运行。