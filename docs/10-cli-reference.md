# 🛠️ CLI 命令参考

[返回文档索引](./README.md)

## 📋 命令概览

INVESTOR-BENCH 提供了完整的命令行界面，支持从配置到评估的完整工作流。

```bash
python run.py [COMMAND] [OPTIONS]
```

## 🚀 主要命令

### 1. warmup - 预热训练

初始化智能体的记忆系统，建立基础的市场认知。

```bash
python run.py warmup [OPTIONS]
```

**选项**:
- `-c, --config-path TEXT`: 配置文件路径 [默认: configs/main.json]

**示例**:
```bash
# 使用默认配置
python run.py warmup

# 使用自定义配置
python run.py warmup -c configs/my_config.json

# 使用统一API配置
python run.py warmup -c configs/main_unified.json
```

**功能说明**:
- 初始化MarketEnv和FinMemAgent
- 按时间序列处理历史数据
- 建立多层次记忆结构
- 保存检查点用于恢复
- 生成详细日志

**输出文件**:
```
results/
├── <YYMMDD_HHMMSS_Model_Name_SYMBOL>/  # 时间戳格式目录
│   ├── warmup_checkpoint/              # 中间检查点
│   ├── warmup_output/                  # 最终输出
│   ├── log/                           # 日志文件
│   ├── report.md                      # 交易报告
│   └── trading_results.csv            # 交易结果CSV
```

**示例**:
```
results/250806_135830_Qwen_Qwen3-8B_JNJ/
├── warmup_checkpoint/
├── warmup_output/
├── log/
│   ├── warmup.log
│   └── warmup_trace.log
├── report.md
└── trading_results.csv
```

### 2. warmup-checkpoint - 从检查点恢复预热

从中断点继续预热过程。

```bash
python run.py warmup-checkpoint [OPTIONS]
```

**选项**:
- `-c, --config-path TEXT`: 配置文件路径 [默认: configs/main.json]

**示例**:
```bash
python run.py warmup-checkpoint -c configs/main.json
```

**使用场景**:
- API调用失败后恢复
- 系统意外中断后继续
- 网络问题导致的中断

### 3. test - 投资测试

使用预热好的智能体进行实际投资决策测试。

```bash
python run.py test [OPTIONS]
```

**选项**:
- `-c, --config-path TEXT`: 配置文件路径 [默认: configs/main.json]

**示例**:
```bash
python run.py test -c configs/main_unified.json
```

**功能说明**:
- 加载warmup阶段的结果
- 执行测试期间的投资决策
- 记录所有交易和性能数据
- 实时计算投资指标

**输出文件**:
```
results/
├── <YYMMDD_HHMMSS_Model_Name_SYMBOL>/  # 时间戳格式目录
│   ├── test_checkpoint/                # 测试检查点
│   ├── test_output/                    # 测试输出
│   ├── final_result/                   # 最终结果
│   ├── report.md                       # 交易报告
│   └── trading_results.csv             # 交易结果CSV
```

**功能增强**:
- 自动生成markdown格式交易报告
- 保存CSV格式的交易数据
- 提供完整的CLI命令参考

### 4. test-checkpoint - 从检查点恢复测试

从测试中断点继续执行。

```bash
python run.py test-checkpoint [OPTIONS]
```

**选项**:
- `-c, --config-path TEXT`: 配置文件路径 [默认: configs/main.json]

**示例**:
```bash
python run.py test-checkpoint -c configs/main.json
```

### 5. eval - 生成评估报告

分析投资表现并生成详细报告。

```bash
python run.py eval [OPTIONS]
```

**选项**:
- `-c, --config-path TEXT`: 配置文件路径 [默认: configs/main.json]

**示例**:
```bash
python run.py eval -c configs/main_unified.json
```

**功能说明**:
- 计算投资收益指标
- 生成风险分析报告
- 创建可视化图表
- 导出CSV格式数据

**输出文件**:
```
results/<YYMMDD_HHMMSS_Model_Name_SYMBOL>/
├── metrics/                    # 评估指标
│   ├── portfolio_summary.csv   # 投资组合摘要
│   ├── daily_returns.csv      # 每日收益
│   ├── trade_history.csv      # 交易历史
│   ├── performance_metrics.json # 性能指标
│   └── risk_analysis.json     # 风险分析
├── report.md                  # 完整交易报告
└── trading_results.csv        # 交易结果汇总
```

## 🕐 时间戳目录结构

### 自动生成格式

系统现在使用时间戳格式自动生成输出目录：

**目录格式**: `results/YYMMDD_HHMMSS_ModelName_SYMBOL`

**示例**:
- `results/250806_135830_Qwen_Qwen3-8B_JNJ/`
- `results/250807_002420_GPT-4_TSLA_NVDA/`

**优势**:
- ✅ 每次运行自动创建独立目录
- ✅ 避免结果覆盖
- ✅ 便于结果追踪和比较
- ✅ 支持并发运行多个实验

### 目录内容

每个时间戳目录包含完整的实验结果：

```
results/250806_135830_Qwen_Qwen3-8B_JNJ/
├── warmup_checkpoint/      # 预热检查点
├── warmup_output/          # 预热结果
├── test_checkpoint/        # 测试检查点  
├── test_output/            # 测试结果
├── final_result/           # 最终投资组合状态
├── log/                    # 详细日志
│   ├── warmup.log
│   ├── test.log
│   ├── warmup_trace.log
│   └── test_trace.log
├── report.md              # 📊 完整交易报告
└── trading_results.csv    # 📈 交易数据CSV
```

### 智能检查点恢复

系统自动找到最新的相关实验结果：

```bash
# test命令会自动查找最新的warmup结果
python run.py test -c configs/test_minimal.json

# 也可以从检查点恢复
python run.py warmup-checkpoint -c configs/test_minimal.json
python run.py test-checkpoint -c configs/test_minimal.json
```

## ⚙️ 配置参数

### 通用配置选项

所有命令都支持以下配置文件格式：

```json
{
  "chat_config": {
    "chat_model": "model_name",
    "chat_model_inference_engine": "openai_compatible",
    "chat_max_new_token": 1000,
    "chat_request_timeout": 60,
    "chat_parameters": {
      "temperature": 0.6
    }
  },
  "emb_config": {
    "emb_model_name": "embedding_model_name",
    "embedding_timeout": 60
  },
  "env_config": {
    "trading_symbols": ["SYMBOL1", "SYMBOL2"],
    "warmup_start_time": "2020-07-02",
    "warmup_end_time": "2020-09-30",
    "test_start_time": "2020-10-01", 
    "test_end_time": "2021-05-06"
  }
}
```

### 路径配置

系统自动处理以下路径：

```json
{
  "meta_config": {
    "run_name": "exp_name",
    "warmup_checkpoint_save_path": "results/exp_name/model/symbols/warmup_checkpoint",
    "warmup_output_save_path": "results/exp_name/model/symbols/warmup_output",
    "test_checkpoint_save_path": "results/exp_name/model/symbols/test_checkpoint",
    "test_output_save_path": "results/exp_name/model/symbols/test_output",
    "result_save_path": "results/exp_name/model/symbols/final_result",
    "log_save_path": "results/exp_name/model/symbols/log"
  }
}
```

## 🔧 实用工具命令

### 配置验证

验证配置文件格式：

```bash
# 验证配置语法
python -c "import json; print('✓ Config valid' if json.load(open('configs/main_unified.json')) else '✗ Config invalid')"

# 测试模型配置
python test_simplified.py

# 测试OpenAI兼容接口
python test_openai_compatible.py
```

### 系统检查

检查依赖和服务状态：

```bash
# 检查Python依赖
python -c "import openai, qdrant_client, loguru; print('✓ Dependencies OK')"

# 检查Qdrant连接
curl -X GET "http://localhost:6333/collections"

# 检查GPU可用性（如果使用本地VLLM）
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 日志查看

实时查看执行日志：

```bash
# 查看warmup日志
tail -f results/exp_unified/meta-llama-3.1-8b-instruct/UVV_TSLA_JNJ/log/warmup.log

# 查看test日志
tail -f results/exp_unified/meta-llama-3.1-8b-instruct/UVV_TSLA_JNJ/log/test.log

# 查看详细trace日志
tail -f results/exp_unified/meta-llama-3.1-8b-instruct/UVV_TSLA_JNJ/log/warmup_trace.log
```

### 清理命令

清理实验数据：

```bash
# 清理所有结果
rm -rf results/*

# 清理特定实验
rm -rf results/exp_name/

# 清理检查点（保留最终结果）
find results/ -name "*checkpoint*" -type d -exec rm -rf {} +
```

## 📊 批处理脚本示例

### 完整实验流程

```bash
#!/bin/bash
# complete_experiment.sh

CONFIG_FILE="configs/test_minimal.json"
echo "🚀 Starting enhanced INVESTOR-BENCH experiment with timestamped outputs"
echo "⏰ 将创建时间戳目录格式: YYMMDD_HHMMSS_ModelName_SYMBOL"

# Step 1: Warmup
echo "📚 Step 1: Starting warmup phase..."
python run.py warmup -c $CONFIG_FILE
if [ $? -eq 0 ]; then
    echo "✅ Warmup completed successfully"
else
    echo "❌ Warmup failed, trying to resume from checkpoint..."
    python run.py warmup-checkpoint -c $CONFIG_FILE
fi

# Step 2: Test (automatically finds latest warmup)
echo "🧪 Step 2: Starting test phase..."
echo "🔍 自动查找最新的warmup结果..."
python run.py test -c $CONFIG_FILE
if [ $? -eq 0 ]; then
    echo "✅ Test completed successfully"
    echo "📊 自动生成了交易报告和CSV文件"
else
    echo "❌ Test failed, trying to resume from checkpoint..."
    python run.py test-checkpoint -c $CONFIG_FILE
fi

# Step 3: Evaluation
echo "📊 Step 3: Generating evaluation report..."
python run.py eval -c $CONFIG_FILE
if [ $? -eq 0 ]; then
    echo "✅ Evaluation completed successfully"
    
    # 查找最新结果目录
    LATEST_DIR=$(ls -t results/ | head -n 1)
    echo "📁 最新实验结果位于: results/$LATEST_DIR/"
    echo "📄 交易报告: results/$LATEST_DIR/report.md"
    echo "📈 交易数据: results/$LATEST_DIR/trading_results.csv"
    echo "📊 评估指标: results/$LATEST_DIR/metrics/"
else
    echo "❌ Evaluation failed"
fi

echo "🎉 增强版实验完成!"
echo "💡 提示: 每次运行都会创建独立的时间戳目录，避免结果覆盖"
```

### 多配置批处理

```bash
#!/bin/bash
# batch_experiments.sh

CONFIGS=("configs/main_unified.json" "configs/gpt4_config.json" "configs/local_vllm.json")

for config in "${CONFIGS[@]}"; do
    echo "🔄 Running experiment with config: $config"
    
    python run.py warmup -c "$config"
    python run.py test -c "$config" 
    python run.py eval -c "$config"
    
    echo "✅ Completed experiment with config: $config"
done

echo "🎊 All experiments completed!"
```

## 🐛 错误处理

### 常见错误代码

| 错误类型 | 退出代码 | 解决方案 |
|---------|----------|----------|
| 配置文件错误 | 1 | 检查JSON语法和必需字段 |
| API连接失败 | 2 | 检查API密钥和网络连接 |
| 内存不足 | 3 | 增加系统内存或减少batch大小 |
| 文件权限错误 | 4 | 检查目录写入权限 |
| 模型加载失败 | 5 | 检查模型配置和可用性 |

### 调试技巧

```bash
# 启用详细日志
export PYTHONPATH=.
export LOGURU_LEVEL=TRACE

# 逐步执行
python -c "
import sys; sys.path.append('.')
from src import *
# 手动执行各组件初始化
"

# 内存使用监控
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## 📚 相关文档

- [配置系统](./04-configuration.md) - 详细配置选项
- [故障排除](./11-troubleshooting.md) - 常见问题解决
- [开发指南](./12-development-guide.md) - 开发者指南

---

[← 上一章: 评估指标](./09-evaluation-metrics.md) | [下一章: 故障排除 →](./11-troubleshooting.md)