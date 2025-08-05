全部代码来自fork的仓库. 此处是本人学习与自用版本. 

# INVESTOR-BENCH

基于OpenAI兼容API的智能投资回测系统，支持完整的四层记忆系统和LLM驱动的交易决策。

## 🚀 快速开始

### 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 启动向量数据库
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant

# 配置API密钥
echo 'OPENAI_API_KEY="sk-your-key-here"' > .env
echo 'OPENAI_API_BASE="https://api.siliconflow.cn/v1"' >> .env
```

### 运行回测
```bash
# 生成配置
pkl eval -f json -o configs/main.json configs/main.pkl

# 执行完整回测流程
python run.py warmup    # 学习阶段
python run.py test      # 回测阶段  
python run.py eval      # 结果分析
```

## 📋 快速CLI参考

| 命令 | 功能 | 说明 |
|------|------|------|
| `python run.py warmup` | 学习阶段 | AI从专业交易员建议中学习 |
| `python run.py warmup-checkpoint` | 恢复学习 | 从检查点继续warmup |
| `python run.py test` | 回测阶段 | 基于记忆进行独立投资决策 |
| `python run.py test-checkpoint` | 恢复回测 | 从检查点继续test |
| `python run.py eval` | 结果分析 | 生成CSV报告和Markdown分析 |
| `python test_api.py` | API测试 | 测试Chat API连通性 |
| `python test_embedding.py` | Embedding测试 | 测试Embedding API功能 |
| `pkl eval -f json -o configs/main.json configs/main.pkl` | 生成配置 | 将PKL配置转换为JSON |

## 🎯 核心特性

### 四层记忆系统
- **短期记忆** (1-7天): 日常新闻、价格波动
- **中期记忆** (1周-3月): 季报分析、行业趋势  
- **长期记忆** (3月以上): 基本面知识、历史经验
- **反思记忆** (持久): 投资哲学、策略总结

### OpenAI兼容API支持
- ✅ **SiliconFlow** - Qwen3-8B + Qwen3-Embedding-4B (已验证)
- ✅ **OpenAI** - GPT-4, text-embedding-3-large
- ✅ **Anthropic** - Claude-3.5-Sonnet (需外部embedding)
- ✅ **本地部署** - 通过vLLM/Ollama等

### 完整投资组合管理
- 单资产和多资产策略支持
- 详细交易记录和证据链
- 关键性能指标计算
- 检查点保存和恢复机制

## 📊 系统架构

```
数据输入 → MarketEnv环境模拟 → FinMemAgent智能代理 → 记忆系统
                                        ↓
投资组合管理 ← LLM决策引擎 ← 记忆检索 ← Qdrant向量数据库
    ↓
结果输出 (CSV + Markdown报告)
```

## 📚 详细文档

完整的项目文档位于 `docs/` 目录：

- [快速开始指南](./docs/quick-start.md) - 5分钟上手教程
- [系统架构](./docs/architecture.md) - 深入理解系统设计
- [API集成指南](./docs/api-integration.md) - 多种API服务商集成
- [记忆系统设计](./docs/memory-system.md) - 四层记忆系统详解
- [CLI命令参考](./docs/cli-reference.md) - 完整命令行使用说明
- [故障排除指南](./docs/troubleshooting.md) - 常见问题解决方案

## 🔧 配置示例

### SiliconFlow API配置
```bash
# .env文件
OPENAI_API_KEY="sk-your-siliconflow-key"
OPENAI_API_BASE="https://api.siliconflow.cn/v1"
OPENAI_MODEL="Qwen/Qwen3-8B"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B"
```

### 自定义交易资产
```pkl
// configs/main.pkl中修改
trading_symbols = new Listing {
    "AAPL"   // Apple
    "GOOGL"  // Google  
    "MSFT"   // Microsoft
}
```

### 调整时间范围
```pkl
warmup_start_time = "2020-01-01"
warmup_end_time = "2020-06-30"
test_start_time = "2020-07-01"
test_end_time = "2020-12-31"
```

## 🎯 运行示例

当前系统正在运行JNJ（强生）的回测示例：
- **测试期间**: 2020-07-02 至 2020-07-10 (短期验证)
- **使用模型**: Qwen3-8B (SiliconFlow API)
- **记忆系统**: 四层记忆完全启用
- **实时状态**: 正在进行LLM交易决策分析

## 📈 预期输出

系统将生成以下结果文件：
- `results/exp/qwen3-8b-siliconflow/JNJ/final_result/` - CSV交易记录
- `results/exp/qwen3-8b-siliconflow/JNJ/final_result/` - Markdown分析报告
- `results/exp/qwen3-8b-siliconflow/JNJ/log/` - 详细运行日志

## 🤝 贡献

这个项目基于原INVESTOR-BENCH框架，专注于OpenAI兼容API的集成和优化。欢迎提交Issue和Pull Request。

## 📄 许可证

请参考原项目许可证要求。

---

## 附录：原项目信息

以下内容保留自原INVESTOR-BENCH项目，供参考：

#### Guardrails Tokens

The [GuardRails](https://github.com/guardrails-ai/guardrails) is used to ensure the output format for closed-sourced models.

If you do not need to evaluate on close-sourced models, comment out the lines 48 - 52 in the [Dockerfile](Dockerfile):

```bash
RUN python -m pip install -r requirements.txt
RUN python -m pip install guardrails-ai==0.5.13
RUN guardrails configure --disable-metrics --disable-remote-inferencing --token xxxxx
RUN guardrails hub install hub://guardrails/valid_choices
```

Otherwise, replace your GuardRails token in line 51 of the [Dockerfile](Dockerfile).

### Config

The configuration in the project is managed by [Pkl](<https://pkl-lang.org/index.html>). The configurations are splitted into two parts: [chat models](</configs/chat_models.pkl>) and [meta config](</configs/main.pkl>).

#### Chat Config

To deploy a fine-tuned / merged LLM model, please add an entry in the [configs/chat_models.pkl](</configs/chat_models.pkl>) that follows the following format:

```pkl
llama3_1_instruct_8b: ChatModelConfig = new {  # set the identifier for the model
    chat_model = "meta-llama/Meta-Llama-3.1-8B-Instruct" # set the model name, which is the model path in the Hugging Face Hub
    chat_model_type = "instruction"  # set the model type, which should be one of the following: instruction, chat, completion.
    # The completion model type is the similar to meta-llama/Llama-3.1-8B that generates the completion for the input text.
    chat_model_inference_engine = "vllm"  # keep it as vllm
    chat_endpoint = null  # keep it null
    chat_template_path = null  # please see detail in VLLM doc: https://github.com/vllm-project/vllm/blob/main/docs/source/serving/openai_compatible_server.md#chat-template
    chat_system_message = "You are a helpful assistant."
    chat_parameters = new Mapping {} # leave it as empty
  }
```

After adding the entry, the model is also needed to be added in the registry.

```pkl
chat_model_dict = new Mapping {
    ["llama-3.1-8b-instruct"] = llama3_1_instruct_8b # [<a short name>] = <model identifier>
  }
```

#### Meta Config

The meta config contains the configuration for the framework. The configuration is located at [configs/main.pkl](<"/configs/main.pkl">) from line 9 to line 29, which contains the following information:

```pkl
hidden config = new meta.MetaConfig {
    run_name = "exp"  # the run name can be set to any string
    agent_name = "finmem_agent"  # also can be set to any string
    trading_symbols = new Listing {
            "BTC-USD"  # the trading symbol. In our case, it either be "BTC-USD" or "ETH-USD"
    }
    warmup_start_time = "2023-02-11"  # do not change this config
    warmup_end_time = "2023-03-10"  # do not change this config
    test_start_time = "2023-03-11"  # do not change this config
    test_end_time = "2023-04-04"  # do not change this config
    top_k = 5  # do not change this config
    look_back_window_size = 3  # do not change this config
    momentum_window_size = 3  # do not change this config
    tensor_parallel_size = 2  # set the tensor parallel size for VLLM, usually set to the number of gpus available
    embedding_model = "text-embedding-3-large"  # do not change this config
    chat_model = "catMemo"  # the chat model's identifier in the chat model registry
    chat_vllm_endpoint = "http://0.0.0.0:8000"  # set this to the VLLM server endpoint, default to localhost port 8000
    chat_parameters = new Mapping {
        ["temperature"] = 0.6 # do not change this config
    }
}
```

#### Generate Config

1. Install jq

```bash
sudo apt-get update
sudo apt-get install jq
```

2. Build evaluation docker container.

```bash
docker build -t devon -f Dockerfile .
```

3. Compile and generate the configuration file.

```bash
docker run -it -v .:/workspace --network host devon config
```

### Deploy Qdrant Vector Database

1. Start a new shell session, the Qdrant server will need to be running in the background.

2. Pull the Qdrant docker image.

```bash
docker pull qdrant/qdrant
```

3. Start the Qdrant server.

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Deploy VLLM Server (Optional, not needed for closed model)

1. Start a new shell session, the VLLM server will need to be running in the background.

2. Pull the VLLM docker image.

```bash
docker pull vllm/vllm-openai:latest
```

3. Start running the VLLM server.

```bash
bash scripts/start_vllm.sh
```

### Running Framework

After deploying the VLLM server and Qdrant vector database, we can run the evaluation framework to assess trading performance. The system need to first be warmed up before running the evaluation framework.

1. Running warm-up.

```bash
docker run -it -v .:/workspace --network host devon warmup
```

If the warm-up is interrupted (OpenAI API error, etc.), please use the following command to resume from the last checkpoint.

```bash
docker run -it -v .:/workspace --network host devon warmup-checkpoint
```

2. Running testing.

```bash
docker run -it -v .:/workspace --network host devon test
```

The test can also be resumed from the last checkpoint.

```bash
docker run -it -v .:/workspace --network host devon test-checkpoint
```

3. Generate a metric report.

```bash
docker run -it -v .:/workspace --network host devon eval
```

The results will be saved in the `results/<run_name>/<chat_model>/<trading_symbols>/metrics` directory.

## Start & End times

### Equities

#### HON, JNJ, UVV, MSFT

```bash
warmup_start_time = "2020-07-01"
warmup_end_time = "2020-09-30"
test_start_time = "2020-10-01"
test_end_time = "2021-05-06"
```

### Cryptocurrencies

#### BTC

```bash
warmup_start_time = "2023-02-11"
warmup_end_time = "2023-04-04"
test_start_time = "2023-04-05"
test_end_time = "2023-12-19"
```

#### ETH

```bash
warmup_start_time = "2023-02-13"
warmup_end_time = "2023-04-02"
test_start_time = "2023-04-03"
test_end_time = "2023-12-19"
```

#### ETF

```bash
warmup_start_time = "2019-07-29",
warmup_end_time = "2019-12-30",
test_start_time = "2020-01-02",
test_end_time = "2020-09-21",
```
