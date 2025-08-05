# INVESTOR-BENCH v2.0

全部代码来自fork的仓库. 此处是本人学习与自用版本. 

基于LLM的智能投资回测系统 - 统一OpenAI SDK接口版本

## 🚀 快速开始

### 安装依赖
```bash
pip install openai python-dotenv loguru httpx pandas
```

### 配置API密钥
```bash
# 方法1: 环境变量 (推荐)
export OPENAI_API_KEY="sk-your-api-key"
export OPENAI_API_BASE="https://api.siliconflow.cn/v1"

# 方法2: .env文件
echo 'OPENAI_API_KEY="sk-your-api-key"' > .env
echo 'OPENAI_API_BASE="https://api.siliconflow.cn/v1"' >> .env
```

### 运行回测
```bash
# 基础用法 - 使用环境变量配置
python investor_bench.py --symbol JNJ --start-date 2020-07-02 --end-date 2020-07-10

# 指定模型和API
python investor_bench.py --symbol AAPL --model gpt-4 --api-key sk-xxx --api-base https://api.openai.com/v1

# 本地部署模型 (vLLM)
python investor_bench.py --symbol TSLA --model llama-3.1-8b --api-base http://localhost:8000/v1 --api-key fake
```

## 📋 命令行参数

### 必需参数
- `--symbol`: 股票代码 (如: JNJ, AAPL, TSLA)
- `--start-date`: 开始日期 (YYYY-MM-DD格式)
- `--end-date`: 结束日期 (YYYY-MM-DD格式)

### 可选参数
- `--mode`: 运行模式 (backtest, warmup, test) - 默认: backtest
- `--model`: LLM模型名称 - 默认: Qwen/Qwen3-8B
- `--api-key`: OpenAI API密钥 (覆盖环境变量)
- `--api-base`: API基础URL (覆盖环境变量)
- `--verbose, -v`: 启用详细输出

## 📊 输出结构

每次运行会在`results/`目录下创建带时间戳的目录：

```
results/
└── 241205_143022_Qwen_Qwen3-8B_JNJ/
    ├── metadata.json          # 运行元数据和统计
    ├── run.log               # 详细运行日志
    ├── results.json          # 完整结果数据
    ├── trading_results.csv   # 交易记录CSV
    └── analysis_report.md    # 分析报告
```

### metadata.json 包含内容
- 运行信息 (时间、命令行、持续时间)
- 配置参数 (模型、API、日期范围等)
- API使用统计 (调用次数、Token消耗、错误次数)
- 系统信息 (Python版本、平台等)

## 🎯 支持的API服务商

### 1. SiliconFlow (默认)
```bash
export OPENAI_API_BASE="https://api.siliconflow.cn/v1"
export OPENAI_API_KEY="sk-your-siliconflow-key"
```

### 2. OpenAI官方
```bash
python investor_bench.py --symbol JNJ --model gpt-4 --api-base https://api.openai.com/v1 --api-key sk-xxx
```

### 3. 本地vLLM部署
```bash
# 启动vLLM服务
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8000

# 使用本地模型
python investor_bench.py --symbol JNJ --model meta-llama/Llama-3.1-8B-Instruct --api-base http://localhost:8000/v1 --api-key fake
```

### 4. 其他OpenAI兼容服务
任何支持OpenAI API格式的服务都可以使用，只需设置正确的`--api-base`。

## 📈 功能特性

### 智能投资决策
- LLM驱动的投资分析和决策
- 基于新闻、价格、历史记忆的综合判断
- 返回BUY/SELL/HOLD决策及置信度

### 记忆系统
- 自动学习和记住重要市场信息
- 历史决策和市场趋势的上下文保持
- 支持长期策略优化

### 完整回测
- 模拟投资组合管理
- 实时计算投资收益和风险指标
- 生成详细的交易记录和分析报告

### API使用统计
- 详细的API调用次数和Token消耗统计
- 成本估算和错误监控
- 支持多种API服务商的统一管理

## 📂 数据格式

股票数据文件应放在`data/`目录，格式如下：

```json
[
  {
    "date": "2020-07-02",
    "symbol": "JNJ",
    "price": 126.30,
    "news": [
      "Johnson & Johnson公布积极的疫苗试验结果",
      "医疗器械部门营收超预期增长"
    ]
  }
]
```

## 🔧 故障排除

### 常见问题

1. **API密钥错误**
   ```bash
   ❌ API密钥未设置！请通过--api-key参数或OPENAI_API_KEY环境变量提供
   ```
   解决：设置正确的API密钥和base URL

2. **数据文件不存在**
   ```bash
   ❌ 数据文件不存在: data/jnj.json
   ```
   解决：确保数据文件存在且格式正确

3. **API调用失败**
   检查网络连接和API服务状态，查看`run.log`了解详细错误信息

### 调试模式
```bash
python investor_bench.py --symbol JNJ --start-date 2020-07-02 --end-date 2020-07-10 --verbose
```

## 🆕 版本更新

### v2.0.0 特性
- ✅ 统一OpenAI SDK接口，支持所有兼容服务
- ✅ 清晰的命令行接口，简单参数启动
- ✅ 带时间戳的结果目录结构
- ✅ 完整的metadata.json元数据记录
- ✅ 详细的API使用统计和成本监控
- ✅ 支持本地vLLM模型部署
- ✅ 清晰的日志和错误处理

### 从v1.0迁移
原有的PKL配置文件和复杂启动脚本已简化为直接的命令行参数。旧版文件已移至`old/`目录。

## 📞 获取帮助

```bash
python investor_bench.py --help
```

查看完整的参数说明和使用示例。