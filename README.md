# INVESTOR-BENCH v2.0

全部代码来自fork的仓库. 此处是本人学习与自用版本. 

基于LLM的智能投资回测系统 - 统一OpenAI SDK接口版本

## 🚀 快速开始

### 安装依赖
```bash
pip install openai python-dotenv loguru httpx pandas numpy
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

### 快速演示
```bash
# 一键运行完整的两阶段实验演示
./quick_demo.sh
```

### 运行回测
```bash
# 基础用法 - 使用环境变量配置
python investor_bench.py --symbol JNJ --start-date 2020-07-02 --end-date 2020-07-10

# 完整两阶段实验 (推荐)
# 阶段1: Warmup
python investor_bench.py --symbol JNJ --start-date 2020-07-02 --end-date 2020-07-10 --mode warmup
# 阶段2: Test  
python investor_bench.py --symbol JNJ --start-date 2020-10-01 --end-date 2021-05-06 --mode test

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

## 📅 重要说明：Warmup vs Test 模式

⚠️ **重要澄清**: 当前简化版本(`investor_bench.py`)中的warmup和test模式是**概念性的**，实际上是两个独立的回测运行，没有状态或知识的传递。

### 🔄 当前版本的模式区别

#### 简化版本 (`python investor_bench.py --mode warmup/test`)
- **Warmup模式**: 独立的回测运行，使用早期日期数据
- **Test模式**: 独立的回测运行，使用后期日期数据  
- **状态传递**: ❌ 无状态传递，每次运行都重新开始
- **记忆系统**: 简化的内存记忆，不持久化
- **适用场景**: 快速验证、简单对比分析

#### 完整版本 (`python run.py warmup` + `python run.py test`)
- **Warmup阶段**: 真正的学习期，建立四层记忆系统和知识库
- **Test阶段**: 基于warmup学到的知识进行决策，有状态传递
- **状态传递**: ✅ 完整的checkpoint和记忆系统传递
- **记忆系统**: Qdrant向量数据库 + 四层记忆架构
- **适用场景**: 严格的学术研究、完整的AI投资系统验证

#### 增强版本 (`python run_enhanced.py`) - 🆕 推荐
- **特点**: 结合完整记忆系统 + 现代化输出风格
- **状态传递**: ✅ 真正的warmup→test知识传递
- **输出**: 🎨 用户友好的emoji和进度显示
- **API调用**: 🔧 统一的OpenAI SDK接口，支持多种服务商
- **使用方式**:
  ```bash
  # 分步执行
  python run_enhanced.py warmup --symbol JNJ --start-date 2020-03-12 --end-date 2020-03-13
  python run_enhanced.py test --symbol JNJ --test-start-date 2020-03-16 --test-end-date 2020-03-17
  
  # 一键执行完整流程
  python run_enhanced.py both --symbol JNJ --start-date 2020-03-12 --end-date 2020-03-13 --test-start-date 2020-03-16 --test-end-date 2020-03-17
  ```

### 🎯 如何选择版本

1. **快速测试和学习**: 使用简化版本 `investor_bench.py`
2. **最佳体验**: 使用增强版本 `run_enhanced.py` (🆕 推荐)
3. **学术研究**: 使用完整版本 `run.py` (原始框架)

### 📊 推荐的日期配置

基于数据质量和市场重要事件，推荐以下测试日期：

| 资产类型 | 股票代码 | Warmup期间 | Test期间 | 说明 |
|---------|---------|-----------|----------|------|
| **传统股票** | JNJ | 2020-07-02 至 2020-07-10 | 2020-10-01 至 2021-05-06 | 涵盖疫情期间医药股表现 |
| **科技股** | MSFT | 2020-07-02 至 2020-07-10 | 2020-10-01 至 2021-05-06 | 科技股在疫情中的表现 |
| **比特币** | BTC | 2023-01-19 至 2023-04-04 | 2023-04-05 至 2023-11-05 | 加密货币波动期分析 |
| **以太坊** | ETH | 2023-01-19 至 2023-04-02 | 2023-04-03 至 2023-11-05 | 以太坊生态发展期 |
| **ETF** | ETF | 2019-07-29 至 2019-12-30 | 2020-01-02 至 2020-09-21 | 传统ETF表现测试 |

### 💡 使用建议
1. **新手**: 建议从JNJ开始，数据质量高且波动适中
2. **短期测试**: 可以缩短日期范围进行快速验证
3. **研究导向**: 选择特定的市场事件期间进行深入分析

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

### 完整回测与交易指标
- 模拟投资组合管理
- 专业交易指标：总收益率、年化收益率、夏普比率、最大回撤、胜率等
- 风险调整收益分析
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
- ✅ 专业交易指标计算（夏普比率、最大回撤、胜率等）
- ✅ 详细的API使用统计和成本监控
- ✅ 支持本地vLLM模型部署
- ✅ 清晰的日志和错误处理
- ✅ 完整的文档系统，包含CLI快速参考

### 从v1.0迁移
原有的PKL配置文件和复杂启动脚本已简化为直接的命令行参数。旧版文件已移至`old/`目录。

## 📚 完整文档

- **[CLI快速参考](docs/cli-quick-reference.md)** - 常用命令快速查找，可直接复制运行 🚀
- **[完整文档目录](docs/README.md)** - 系统架构、配置管理、使用指南等详细文档
- **[快速开始指南](docs/quick-start.md)** - 5分钟上手教程
- **[API集成指南](docs/api-integration.md)** - 多种API服务商集成方法
- **[故障排除](docs/troubleshooting.md)** - 常见问题和解决方案

## 📞 获取帮助

```bash
python investor_bench.py --help
```

查看完整的参数说明和使用示例。