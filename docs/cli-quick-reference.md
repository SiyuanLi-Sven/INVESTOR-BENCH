# 🚀 CLI 快速参考

本文档提供INVESTOR-BENCH的所有常用命令，可以直接复制运行。

## 💡 重要概念

**INVESTOR-BENCH采用两阶段设计**：
- 🎓 **Warmup阶段**: AI学习历史数据，建立记忆和投资风格
- 🎯 **Test阶段**: AI基于学到的知识在"未来"数据上进行实际投资

**强烈建议**：完整的实验应该包含Warmup + Test两个阶段，而不是仅运行单个阶段！

## 📋 环境配置

### 配置API密钥
```bash
# 创建.env文件 (推荐)
cat > .env << 'EOF'
OPENAI_API_KEY="sk-your-siliconflow-api-key"
OPENAI_API_BASE="https://api.siliconflow.cn/v1"
OPENAI_MODEL="Qwen/Qwen3-8B"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B"
EOF

# 或者设置环境变量
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_API_BASE="https://api.siliconflow.cn/v1"
```

### 安装依赖
```bash
pip install openai python-dotenv loguru httpx pandas numpy
```

## 🎯 基础使用

### 快速测试 (3天)
```bash
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-02 \
  --end-date 2020-07-10 \
  --mode test
```

## 📅 标准两阶段实验

INVESTOR-BENCH采用Warmup(学习) + Test(实战)两阶段设计，推荐使用以下预设配置：

### 🎓 JNJ完整实验 (推荐新手)
```bash
# 阶段1: Warmup - AI学习期 (9天)
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-02 \
  --end-date 2020-07-10 \
  --mode warmup \
  --verbose

# 阶段2: Test - AI实战期 (7个月+)
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-10-01 \
  --end-date 2021-05-06 \
  --mode test \
  --verbose
```

### 🚀 MSFT科技股实验
```bash
# Warmup阶段
python investor_bench.py \
  --symbol MSFT \
  --start-date 2020-07-02 \
  --end-date 2020-07-10 \
  --mode warmup

# Test阶段  
python investor_bench.py \
  --symbol MSFT \
  --start-date 2020-10-01 \
  --end-date 2021-05-06 \
  --mode test
```

### ₿ 比特币完整实验 (2023)
```bash
# Warmup阶段 (2.5个月)
python investor_bench.py \
  --symbol BTC \
  --start-date 2023-01-19 \
  --end-date 2023-04-04 \
  --mode warmup

# Test阶段 (7个月)
python investor_bench.py \
  --symbol BTC \
  --start-date 2023-04-05 \
  --end-date 2023-11-05 \
  --mode test
```

### ⚡ 以太坊完整实验 (2023)
```bash
# Warmup阶段
python investor_bench.py \
  --symbol ETH \
  --start-date 2023-01-19 \
  --end-date 2023-04-02 \
  --mode warmup

# Test阶段
python investor_bench.py \
  --symbol ETH \
  --start-date 2023-04-03 \
  --end-date 2023-11-05 \
  --mode test
```

## 🔬 自定义实验

### 短期回测 (1周)
```bash
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-07-10 \
  --mode test \
  --verbose
```

### 月度回测 (30天)
```bash
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-08-13 \
  --mode test
```

### 季度回测 (90天)
```bash
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-10-01 \
  --mode test
```

## 📊 不同股票回测

### 科技股
```bash
# Microsoft
python investor_bench.py \
  --symbol MSFT \
  --start-date 2020-07-01 \
  --end-date 2020-08-01 \
  --mode test

# 特斯拉 (如果有数据)
python investor_bench.py \
  --symbol TSLA \
  --start-date 2020-07-01 \
  --end-date 2020-08-01 \
  --mode test
```

### 加密货币
```bash
# 比特币
python investor_bench.py \
  --symbol BTC \
  --start-date 2020-07-01 \
  --end-date 2020-08-01 \
  --mode test

# 以太坊
python investor_bench.py \
  --symbol ETH \
  --start-date 2020-07-01 \
  --end-date 2020-08-01 \
  --mode test
```

## 🔧 不同API配置

### SiliconFlow (默认)
```bash
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-07-15 \
  --model "Qwen/Qwen3-8B" \
  --api-base "https://api.siliconflow.cn/v1" \
  --api-key "sk-your-siliconflow-key"
```

### OpenAI官方
```bash
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-07-15 \
  --model "gpt-4" \
  --api-base "https://api.openai.com/v1" \
  --api-key "sk-your-openai-key"
```

### 本地vLLM部署
```bash
# 启动vLLM服务
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# 使用本地模型
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-07-15 \
  --model "meta-llama/Llama-3.1-8B-Instruct" \
  --api-base "http://localhost:8000/v1" \
  --api-key "fake"
```

## 📈 高级配置

### 详细日志
```bash
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-07-15 \
  --verbose
```

### 不同运行模式
```bash
# Warmup模式 (学习历史数据)
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-07-15 \
  --mode warmup

# Test模式 (实际交易回测)
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-07-15 \
  --mode test

# Backtest模式 (综合回测，默认)
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-07-15 \
  --mode backtest
```

## 🎨 实用脚本

### 完整两阶段实验脚本
```bash
#!/bin/bash
# two_stage_experiment.sh - 完整的Warmup + Test实验

run_two_stage_experiment() {
    local symbol=$1
    local warmup_start=$2
    local warmup_end=$3
    local test_start=$4
    local test_end=$5
    
    echo "=== 开始 $symbol 两阶段实验 ==="
    
    echo "阶段1: Warmup ($warmup_start 至 $warmup_end)"
    python investor_bench.py \
        --symbol "$symbol" \
        --start-date "$warmup_start" \
        --end-date "$warmup_end" \
        --mode warmup \
        --verbose
    
    echo "阶段2: Test ($test_start 至 $test_end)"
    python investor_bench.py \
        --symbol "$symbol" \
        --start-date "$test_start" \
        --end-date "$test_end" \
        --mode test \
        --verbose
        
    echo "=== $symbol 实验完成 ==="
    echo ""
}

# 运行标准实验配置
run_two_stage_experiment "JNJ" "2020-07-02" "2020-07-10" "2020-10-01" "2021-05-06"
run_two_stage_experiment "MSFT" "2020-07-02" "2020-07-10" "2020-10-01" "2021-05-06"  
run_two_stage_experiment "BTC" "2023-01-19" "2023-04-04" "2023-04-05" "2023-11-05"
run_two_stage_experiment "ETH" "2023-01-19" "2023-04-02" "2023-04-03" "2023-11-05"
```

### 批量回测多个时间段
```bash
#!/bin/bash
# batch_test.sh

symbols=("JNJ" "MSFT" "BTC" "ETH")
start_dates=("2020-07-01" "2020-08-01" "2020-09-01")
end_dates=("2020-07-31" "2020-08-31" "2020-09-30")

for symbol in "${symbols[@]}"; do
  for i in "${!start_dates[@]}"; do
    echo "Testing $symbol from ${start_dates[$i]} to ${end_dates[$i]}"
    python investor_bench.py \
      --symbol "$symbol" \
      --start-date "${start_dates[$i]}" \
      --end-date "${end_dates[$i]}" \
      --mode test
  done
done
```

### 性能测试
```bash
#!/bin/bash
# performance_test.sh

# 快速测试API连通性
echo "=== API连通性测试 ==="
python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-07-03 \
  --mode test \
  --verbose

# 中等规模测试
echo "=== 中等规模测试 (1周) ==="
time python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-07-10 \
  --mode test

# 大规模测试
echo "=== 大规模测试 (1个月) ==="
time python investor_bench.py \
  --symbol JNJ \
  --start-date 2020-07-01 \
  --end-date 2020-08-01 \
  --mode test
```

## 📊 结果分析

### 查看生成的报告
```bash
# 查看最新运行结果
ls -lt results/ | head -5

# 查看特定运行的报告
LATEST_RUN=$(ls -t results/ | head -1)
echo "查看最新运行: $LATEST_RUN"

# 查看Markdown报告
cat "results/$LATEST_RUN/analysis_report.md"

# 查看CSV数据
head -10 "results/$LATEST_RUN/trading_results.csv"

# 查看JSON元数据
cat "results/$LATEST_RUN/metadata.json" | jq '.'
```

### 性能对比脚本
```bash
#!/bin/bash
# compare_results.sh

echo "=== 最近5次运行的性能对比 ==="
for dir in $(ls -t results/ | head -5); do
  echo "运行: $dir"
  if [ -f "results/$dir/metadata.json" ]; then
    echo "  收益率: $(cat results/$dir/metadata.json | jq -r '.performance.total_return // "N/A"')"
    echo "  API调用: $(cat results/$dir/metadata.json | jq -r '.api_statistics.chat_calls // "N/A"')"
    echo "  Token消耗: $(cat results/$dir/metadata.json | jq -r '.api_statistics.total_tokens // "N/A"')"
  fi
  echo ""
done
```

## 🔍 故障排除

### 检查数据文件
```bash
# 列出可用的数据文件
ls -la data/

# 检查特定股票的数据范围
python -c "
import json
with open('data/jnj.json') as f:
    data = json.load(f)
dates = sorted(data.keys())
print(f'JNJ数据范围: {dates[0]} 到 {dates[-1]}')
print(f'总天数: {len(dates)}')
"
```

### API连通性测试
```bash
# 测试SiliconFlow API
curl -X POST "https://api.siliconflow.cn/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "Qwen/Qwen3-8B",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

### 环境检查
```bash
# 检查Python包
python -c "
import sys
packages = ['openai', 'loguru', 'numpy', 'pandas']
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg} - 需要安装: pip install {pkg}')
"

# 检查环境变量
echo "API Key: ${OPENAI_API_KEY:0:10}..."
echo "API Base: $OPENAI_API_BASE"
```

## ⚡ 一键运行脚本

### 创建便捷命令
```bash
# 创建alias
echo 'alias investor="python /path/to/investor_bench.py"' >> ~/.bashrc
source ~/.bashrc

# 现在可以直接使用
investor --symbol JNJ --start-date 2020-07-01 --end-date 2020-07-15
```

### Docker运行 (可选)
```bash
# 构建Docker镜像
cat > Dockerfile << 'EOF'
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
ENV PYTHONPATH=/app
CMD ["python", "investor_bench.py", "--help"]
EOF

# 构建和运行
docker build -t investor-bench .
docker run -e OPENAI_API_KEY="$OPENAI_API_KEY" \
  investor-bench \
  --symbol JNJ --start-date 2020-07-01 --end-date 2020-07-15
```

---

## 📞 获取帮助

```bash
# 查看完整参数说明
python investor_bench.py --help

# 查看版本信息
python investor_bench.py --version
```

*更多示例和详细说明请参考 [docs/](../docs/) 目录下的其他文档。*