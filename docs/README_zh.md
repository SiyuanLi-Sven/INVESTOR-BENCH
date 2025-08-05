# InvestorBench

## 用法指南

在本节中，我们提供一个分步指南，用于运行包含微调后LLM的评估框架。该评估框架包括三个部分：

- **VLLM 服务器**: 提供微调后LLM API的服务器。我们将使用VLLM团队提供的Docker镜像，并探索如何部署一个LLM以及一个带有LoRA头的LLM。

- **Qdrant 向量数据库**: 我们将使用Qdrant作为记忆存储的向量数据库。

- **主框架**: 部署VLLM服务器和Qdrant向量数据库后，我们将演示如何运行评估框架以评估交易性能。

### 凭证配置

#### OpenAI & HuggingFace 令牌

凭证需要保存在 [.env](/.env) 文件中。`.env` 文件应包含以下信息：

```bash
OPENAI_API_KEY=XXXXXX-XXXXXX-XXXXXX-XXXXXX-XXXXXX
HUGGING_FACE_HUB_TOKEN=XXXXXX-XXXXXX-XXXXXX-XXXXXX-XXXXXX
```

OpenAI API密钥用于为输入文本生成嵌入（embeddings）。Hugging Face Hub令牌用于下载微调后的LLM模型。请确保Hugging Face Hub令牌有权访问微调后的LLM模型/LORA头。

#### Guardrails 令牌

[GuardRails](https://github.com/guardrails-ai/guardrails) 用于确保闭源模型的输出格式正确。

如果您不需要评估闭源模型，请注释掉 [Dockerfile](Dockerfile) 中的第48 - 52行：

```bash
RUN python -m pip install -r requirements.txt
RUN python -m pip install guardrails-ai==0.5.13
RUN guardrails configure --disable-metrics --disable-remote-inferencing --token xxxxx
RUN guardrails hub install hub://guardrails/valid_choices
```

否则，请在 [Dockerfile](Dockerfile) 的第51行替换为您的GuardRails令牌。

### 配置

项目中的配置由 [Pkl](<https://pkl-lang.org/index.html>) 管理。配置分为两部分：[聊天模型](</configs/chat_models.pkl>) 和 [元配置](</configs/main.pkl>)。

#### 聊天配置

要部署一个微调/合并后的LLM模型，请在 [configs/chat_models.pkl](</configs/chat_models.pkl>) 中添加一个遵循以下格式的条目：

```pkl
llama3_1_instruct_8b: ChatModelConfig = new {  # 为模型设置标识符
    chat_model = "meta-llama/Meta-Llama-3.1-8B-Instruct" # 设置模型名称，即Hugging Face Hub中的模型路径
    chat_model_type = "instruction"  # 设置模型类型，应为以下之一: instruction, chat, completion。
    # completion模型类型类似于meta-llama/Llama-3.1-8B，为输入文本生成补全。
    chat_model_inference_engine = "vllm"  # 保持为vllm
    chat_endpoint = null  # 保持为null
    chat_template_path = null  # 详细信息请参见VLLM文档: https://github.com/vllm-project/vllm/blob/main/docs/source/serving/openai_compatible_server.md#chat-template
    chat_system_message = "You are a helpful assistant."
    chat_parameters = new Mapping {} # 留空
}
```

添加条目后，还需要将模型添加到注册表中。

```pkl
chat_model_dict = new Mapping {
    ["llama-3.1-8b-instruct"] = llama3_1_instruct_8b # [<简称>] = <模型标识符>
}
```

#### 元配置

元配置包含框架的配置。该配置位于 [configs/main.pkl](</configs/main.pkl>) 的第9至29行，包含以下信息：

```pkl
hidden config = new meta.MetaConfig {
    run_name = "exp"  # 运行名称，可设置为任何字符串
    agent_name = "finmem_agent"  # 同样可设置为任何字符串
    trading_symbols = new Listing {
            "BTC-USD"  # 交易标的。在我们的案例中，为 "BTC-USD" 或 "ETH-USD"
    }
    warmup_start_time = "2023-02-11"  # 不要更改此配置
    warmup_end_time = "2023-03-10"  # 不要更改此配置
    test_start_time = "2023-03-11"  # 不要更改此配置
    test_end_time = "2023-04-04"  # 不要更改此配置
    top_k = 5  # 不要更改此配置
    look_back_window_size = 3  # 不要更改此配置
    momentum_window_size = 3  # 不要更改此配置
    tensor_parallel_size = 2  # 设置VLLM的张量并行大小，通常设置为可用GPU的数量
    embedding_model = "text-embedding-3-large"  # 不要更改此配置
    chat_model = "catMemo"  # 聊天模型在模型注册表中的标识符
    chat_vllm_endpoint = "http://0.0.0.0:8000"  # 设置为VLLM服务器端点，默认为localhost端口8000
    chat_parameters = new Mapping {
        ["temperature"] = 0.6 # 不要更改此配置
    }
}
```

#### 生成配置

1. 安装 jq

```bash
sudo apt-get update
sudo apt-get install jq
```

2. 构建评估Docker容器。

```bash
docker build -t devon -f Dockerfile .
```

3. 编译并生成配置文件。

```bash
docker run -it -v .:/workspace --network host devon config
```

### 部署 Qdrant 向量数据库

1. 启动一个新的shell会话，Qdrant服务器需要在后台运行。

2. 拉取Qdrant docker镜像。

```bash
docker pull qdrant/qdrant
```

3. 启动Qdrant服务器。

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### 部署 VLLM 服务器 (可选，闭源模型不需要)

1. 启动一个新的shell会话，VLLM服务器需要在后台运行。

2. 拉取VLLM docker镜像。

```bash
docker pull vllm/vllm-openai:latest
```

3. 启动VLLM服务器。

```bash
bash scripts/start_vllm.sh
```

### 运行框架

部署VLLM服务器和Qdrant向量数据库后，我们可以运行评估框架以评估交易性能。在运行评估框架之前，系统需要先进行预热。

1. 运行预热。

```bash
docker run -it -v .:/workspace --network host devon warmup
```

如果预热被中断（如OpenAI API错误等），请使用以下命令从上一个检查点恢复。

```bash
docker run -it -v .:/workspace --network host devon warmup-checkpoint
```

2. 运行测试。

```bash
docker run -it -v .:/workspace --network host devon test
```

测试也可以从上一个检查点恢复。

```bash
docker run -it -v .:/workspace --network host devon test-checkpoint
```

3. 生成指标报告。

```bash
docker run -it -v .:/workspace --network host devon eval
```

结果将保存在 `results/<run_name>/<chat_model>/<trading_symbols>/metrics` 目录中。

## 起始与结束时间

### 股票 (Equities)

#### HON, JNJ, UVV, MSFT

```bash
warmup_start_time = "2020-07-01"
warmup_end_time = "2020-09-30"
test_start_time = "2020-10-01"
test_end_time = "2021-05-06"
```

### 加密货币 (Cryptocurrencies)

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