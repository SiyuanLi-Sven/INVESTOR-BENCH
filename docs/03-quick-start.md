# 🚀 快速开始指南

[返回文档索引](./README.md)

## 🎯 5分钟快速上手

### Step 1: 验证环境

首先确认环境已正确配置：

```bash
# 检查基础依赖
python test_simplified.py

# 预期输出
# ✓ 配置文件导入成功
# ✓ 找到 X 个LLM模型
# ✓ 找到 X 个Embedding模型
# 🎉 核心功能测试通过！
```

### Step 2: 启动必要服务

启动Qdrant向量数据库：

```bash
# 使用Docker启动
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant

# 验证服务
curl -X GET "http://localhost:6333/collections"
# 应该返回: {"result":{"collections":[]},"status":"ok","time":...}
```

### Step 3: 配置API密钥

编辑 `config.py` 文件，更新您的API密钥：

```python
MODEL_CONFIGS = {
    "Qwen/Qwen3-8B": {
        "type": "llm_api",
        "model": "Qwen/Qwen3-8B",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "sk-your-actual-siliconflow-key-here",  # 🔑 更新这里
        "provider": "siliconflow"
    },
    "Qwen/Qwen3-Embedding-4B": {
        "type": "embedding_api",
        "model": "Qwen/Qwen3-Embedding-4B", 
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "sk-your-actual-siliconflow-key-here",  # 🔑 更新这里
        "provider": "siliconflow"
    }
}
```

### Step 4: 运行第一个实验

使用预配置的实验设置：

```bash
# 运行Warmup阶段（约2-5分钟）
python run.py warmup -c configs/main_unified.json

# 运行Test阶段（约5-10分钟）  
python run.py test -c configs/main_unified.json

# 生成评估报告
python run.py eval -c configs/main_unified.json
```

### Step 5: 查看结果

```bash
# 查看实验结果
ls results/exp_unified/meta-llama-3.1-8b-instruct/UVV_TSLA_JNJ/

# 查看性能指标
cat results/exp_unified/meta-llama-3.1-8b-instruct/UVV_TSLA_JNJ/metrics/performance_metrics.json
```

## 🛠️ 自定义实验

### 创建自定义配置

复制并修改配置模板：

```bash
# 复制模板
cp configs/main_unified.json configs/my_experiment.json

# 编辑配置
vim configs/my_experiment.json
```

### 关键配置项说明

```json
{
  "chat_config": {
    "chat_model": "Qwen/Qwen3-8B",                    // 选择LLM模型
    "chat_model_inference_engine": "openai_compatible", // 使用统一接口
    "chat_max_new_token": 1000,                       // 控制响应长度
    "chat_parameters": {
      "temperature": 0.6                              // 控制创造性
    }
  },
  "env_config": {
    "trading_symbols": ["JNJ"],                       // 选择交易标的
    "warmup_start_time": "2020-03-12",               // 设置时间范围
    "warmup_end_time": "2020-09-30"
  }
}
```

### 支持的资产和时间范围

#### 股票资产
```json
// JNJ - Johnson & Johnson
"trading_symbols": ["JNJ"],
"warmup_start_time": "2020-03-12",
"warmup_end_time": "2020-09-30", 
"test_start_time": "2020-10-01",
"test_end_time": "2021-05-06"

// 多资产组合
"trading_symbols": ["JNJ", "UVV", "MSFT"],
"type": "multi-assets"
```

#### 加密货币
```json
// BTC
"trading_symbols": ["BTC-USD"],
"warmup_start_time": "2023-02-11",
"test_end_time": "2023-12-19"
```

## 🔧 模型配置选项

### 1. 硅基流动模型（推荐新手）

```python
"Qwen/Qwen3-8B": {
    "type": "llm_api",
    "model": "Qwen/Qwen3-8B",
    "api_base": "https://api.siliconflow.cn/v1",
    "api_key": "sk-your-key",
    "provider": "siliconflow"
}
```

**优势**: 成本低、速度快、易于配置

### 2. OpenAI模型（推荐高质量需求）

```python
"gpt-4": {
    "type": "llm_api", 
    "model": "gpt-4",
    "api_base": "https://api.openai.com/v1",
    "api_key": "sk-your-openai-key",
    "provider": "openai"
}
```

**优势**: 质量最高、功能完整

### 3. 本地VLLM（推荐开发调试）

```python
"local-vllm": {
    "type": "llm_api",
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "api_base": "http://0.0.0.0:8000/v1", 
    "api_key": "EMPTY",
    "provider": "vllm"
}
```

**优势**: 无API成本、完全控制、调试方便

## 📊 实验场景示例

### 场景1: 单资产投资分析

快速分析单个股票的投资策略：

```json
{
  "env_config": {
    "trading_symbols": ["JNJ"],
    "warmup_start_time": "2020-03-12",
    "warmup_end_time": "2020-03-20",    // 短期warmup
    "test_start_time": "2020-03-23", 
    "test_end_time": "2020-03-30"       // 短期测试
  },
  "portfolio_config": {
    "type": "single-asset",
    "cash": 100000
  }
}
```

### 场景2: 多资产投资组合

比较多个资产的投资表现：

```json
{
  "env_config": {
    "trading_symbols": ["UVV", "TSLA", "JNJ"],
    "warmup_start_time": "2020-07-02",
    "warmup_end_time": "2020-09-30",
    "test_start_time": "2020-10-01",
    "test_end_time": "2021-05-06"
  },
  "portfolio_config": {
    "type": "multi-assets",
    "cash": 1000000
  }
}
```

### 场景3: 模型对比实验

比较不同LLM的投资决策能力：

```bash
# GPT-4实验
python run.py warmup -c configs/gpt4_config.json
python run.py test -c configs/gpt4_config.json 
python run.py eval -c configs/gpt4_config.json

# Qwen实验  
python run.py warmup -c configs/qwen_config.json
python run.py test -c configs/qwen_config.json
python run.py eval -c configs/qwen_config.json

# 比较结果
python -c "
import json
gpt4_metrics = json.load(open('results/gpt4/metrics/performance_metrics.json'))
qwen_metrics = json.load(open('results/qwen/metrics/performance_metrics.json'))
print(f'GPT-4 收益率: {gpt4_metrics[\"total_return\"]:.2%}')
print(f'Qwen 收益率: {qwen_metrics[\"total_return\"]:.2%}')
"
```

## ⚡ 快捷命令

### 创建快速启动脚本

```bash
#!/bin/bash
# quick_run.sh

echo "🚀 INVESTOR-BENCH 快速运行脚本"

# 检查服务
echo "📡 检查Qdrant服务..."
curl -s -X GET "http://localhost:6333/collections" > /dev/null || {
    echo "❌ Qdrant未运行，正在启动..."
    docker run -d -p 6333:6333 qdrant/qdrant
    sleep 3
}

# 运行实验
echo "🧪 开始运行实验..."
CONFIG="configs/main_unified.json"

python run.py warmup -c $CONFIG && \
python run.py test -c $CONFIG && \
python run.py eval -c $CONFIG

echo "✅ 实验完成！查看结果："
echo "📊 $(python -c "import json; c=json.load(open('$CONFIG')); print(c['meta_config']['result_save_path'])")"
```

### 批量实验脚本

```bash
#!/bin/bash
# batch_experiments.sh

MODELS=("Qwen/Qwen3-8B" "gpt-4" "local-vllm")
SYMBOLS=("JNJ" "UVV" "TSLA")

for model in "${MODELS[@]}"; do
    for symbol in "${SYMBOLS[@]}"; do
        echo "🔄 运行实验: $model + $symbol"
        
        # 创建配置文件
        sed "s/PLACEHOLDER_MODEL/$model/g; s/PLACEHOLDER_SYMBOL/$symbol/g" \
            configs/template.json > configs/temp_${model//\//-}_${symbol}.json
        
        # 运行实验
        python run.py warmup -c configs/temp_${model//\//-}_${symbol}.json
        python run.py test -c configs/temp_${model//\//-}_${symbol}.json  
        python run.py eval -c configs/temp_${model//\//-}_${symbol}.json
    done
done
```

## 🎛️ 调试和监控

### 实时监控

```bash
# 监控warmup进度
tail -f results/exp_unified/meta-llama-3.1-8b-instruct/UVV_TSLA_JNJ/log/warmup.log

# 监控API调用
grep -i "api" results/*/log/warmup.log | tail -20

# 监控系统资源
watch "ps aux | grep python | head -5 && echo '---' && free -h"
```

### 常用调试命令

```bash
# 验证配置
python -c "
import json
config = json.load(open('configs/main_unified.json'))
print('✓ 配置有效')
print(f'模型: {config[\"chat_config\"][\"chat_model\"]}')  
print(f'标的: {config[\"env_config\"][\"trading_symbols\"]}')
"

# 测试API连接
python -c "
from src.chat.endpoint.openai_compatible import OpenAICompatibleClient
try:
    client = OpenAICompatibleClient('Qwen/Qwen3-8B')
    print('✓ API客户端创建成功')
except Exception as e:
    print(f'✗ API客户端创建失败: {e}')
"

# 检查数据完整性
python -c "
import json
data = json.load(open('data/jnj.json'))
print(f'✓ 数据文件包含 {len(data)} 天的数据')
print(f'时间范围: {min(data.keys())} 到 {max(data.keys())}')
"
```

## 🎓 下一步学习

完成快速开始后，建议继续学习：

1. **[配置系统](./04-configuration.md)** - 深入了解配置选项
2. **[Agent架构](./05-agent-architecture.md)** - 理解智能体工作原理  
3. **[评估指标](./09-evaluation-metrics.md)** - 学习性能分析
4. **[CLI参考](./10-cli-reference.md)** - 掌握所有命令

## ❓ 常见问题

**Q: 第一次运行需要多长时间？**
A: Warmup通常需要2-10分钟，Test阶段5-30分钟，取决于数据范围和模型响应速度。

**Q: 可以中断后继续吗？**
A: 可以，使用 `warmup-checkpoint` 和 `test-checkpoint` 命令从断点继续。

**Q: 如何降低API成本？**
A: 使用硅基流动API、减少时间范围、使用本地VLLM，或调整`max_tokens`参数。

**Q: 支持哪些资产类型？**
A: 目前支持股票(JNJ、UVV、TSLA等)和加密货币(BTC、ETH)，数据覆盖2020-2023年。

---

[← 上一章: 安装与配置](./02-installation-setup.md) | [下一章: 配置系统 →](./04-configuration.md)