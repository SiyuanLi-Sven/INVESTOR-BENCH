# Investor-Bench: 基于LLM的事件驱动投资回测框架

本项目几乎所有代码来自https://github.com/felis33/INVESTOR-BENCH, 此仓库为自用的版本. 

Investor-Bench是一个专为评估基于大型语言模型（LLM）的自主金融投资代理（Agent）而设计的回测框架。它模拟真实金融市场环境，让LLM驱动的代理在这个环境中进行连续的投资决策，特别关注事件驱动和新闻驱动的投资策略。

## 核心特性

- **事件驱动的模拟**: 系统以"天"为单位推进时间，在每个交易日向代理提供最新的市场信息（股价、新闻等）
- **多层记忆系统**: 代理拥有短期、中期、长期和反思记忆，能够进行长期规划和反思
- **结构化的LLM交互**: 与LLM的通信是高度结构化的，确保决策的可靠性和可复现性
- **全面的性能评估**: 提供完整的评估流水线，计算投资策略的各项金融指标（夏普比率、最大回撤等）
- **简单的OpenAI集成**: 无需复杂配置，直接使用OpenAI API进行模型调用

## 快速开始

### 环境设置

1. 创建并激活Conda环境:
```bash
conda create -n investor-bench python=3.10
conda activate investor-bench
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

3. 配置API密钥:
在项目根目录创建`.env`文件:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### 运行回测

1. 运行预热阶段:
```bash
python run.py warmup
```

2. 运行测试阶段:
```bash
python run.py test
```

3. 评估结果:
```bash
python run.py eval
```

## 命令行参数

```
python run.py [command] [options]
```

可用命令:
- `warmup`: 运行预热阶段，让代理熟悉市场环境
- `warmup-checkpoint`: 从检查点恢复预热阶段
- `test`: 运行测试阶段，执行实际回测
- `test-checkpoint`: 从检查点恢复测试阶段
- `eval`: 评估回测结果，生成性能指标

可选参数:
- `-c, --config-path`: 配置文件路径 (默认: configs/main.json)
- `--model-name`: 指定使用的OpenAI模型 (可选，默认使用配置文件中的设置)

## 配置文件

主配置文件位于`configs/main.json`，包含以下关键设置:

- `chat_config`: LLM模型配置
  - `chat_model`: 使用的模型名称 (如 "gpt-4-turbo")
  - `chat_model_inference_engine`: 推理引擎 ("openai" 或 "vllm")
  
- `env_config`: 回测环境配置
  - `trading_symbols`: 交易标的列表
  - `warmup_start_time`/`warmup_end_time`: 预热阶段时间范围
  - `test_start_time`/`test_end_time`: 测试阶段时间范围

- `portfolio_config`: 投资组合配置
  - `cash`: 初始资金
  - `look_back_window_size`: 回溯窗口大小

## 项目结构

- `src/`: 源代码
  - `agent.py`: 投资代理实现
  - `market_env.py`: 市场环境模拟
  - `memory_db.py`: 记忆系统实现
  - `chat/`: LLM交互模块
  - `eval_pipeline.py`: 评估流水线

- `configs/`: 配置文件
- `data/`: 市场数据
- `results/`: 回测结果

## 自定义模型

您可以通过命令行参数或修改配置文件来使用不同的OpenAI模型:

```bash
python run.py test --model-name "gpt-4o"
```

对于更完整的OpenAI模型配置，我们提供了一个预配置的OpenAI模型配置文件：

```bash
python run.py test -c configs/openai.json
```

您也可以结合使用配置文件和模型参数：

```bash
python run.py test -c configs/openai.json --model-name "gpt-4o"
```

## 输出结果

回测结果保存在`results/<run_name>/<model_name>/<trading_symbols>/`目录下:

- `metrics/`: 性能指标（CSV格式）
- `test_output/`: 测试阶段输出
- `log/`: 日志文件 