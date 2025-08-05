# 模块详解: 金融代理 (Agent)

**文件路径**: `src/agent.py`

欢迎来到 `Investor-Bench` 中最核心、最复杂模块的深度解析。**`FinMemAgent`** 是整个系统的“大脑”，它负责感知市场环境、处理信息、利用记忆、与大型语言模型（LLM）交互，并最终做出投资决策。

## 1. 核心设计思想

`FinMemAgent` 的设计哲学是模仿一个高度专业且经验丰富的金融分析师。它并非一个简单的反应式程序，其行为基于以下几个核心原则：

- **记忆驱动决策 (Memory-Driven)**: 代理的行为不仅仅基于当前的市场信息，更深层次地依赖于其内部的、动态变化的记忆库。这使得代理能够识别长期模式、从过去的成功与失败中学习。
- **思考链 (Chain-of-Thought)**: 通过精心设计的提示（Prompt），引导 LLM 进行一步一步的推理，而不是直接输出一个决策。这包括分析新闻、回顾历史、形成判断、并最终决定行动。
- **结构化交互**: 与LLM的所有通信都强制遵循预定义的 JSON 格式，极大地提高了决策过程的可靠性和可解释性。
- **模块化集成**: `FinMemAgent` 作为一个中心协调器，有机地整合了记忆系统（`MemoryDB`）、投资组合（`Portfolio`）和 LLM 通信模块（`src/chat/`）。

---

## 2. 关键类与方法

### 2.1 `FinMemAgent` 类

#### 初始化 `__init__`

```python
# src/agent.py

class FinMemAgent:
    def __init__(
        self,
        agent_config: Dict[str, Any],
        emb_config: Dict[str, Any],
        chat_config: Dict[str, Any],
        portfolio_config: Dict[str, Any],
        task_type: TaskType,
    ) -> None:
        # ...
        self.memory_db = MemoryDB(...)
        self.chat_schema, self.chat_endpoint, self.chat_prompt = get_chat_model(...)
        self._config_memory_settings()
        self.portfolio = construct_portfolio(...)
```

- **输入参数**: 代理的初始化依赖于从 `configs/main.json` 解析出的多个配置字典，涵盖了代理本身、嵌入模型、聊天模型和投资组合的全部设置。
- **核心逻辑**:
    1.  **初始化记忆数据库 (`MemoryDB`)**: 创建一个 `MemoryDB` 实例，这是代理的记忆中枢。
    2.  **获取聊天模型 (`get_chat_model`)**: 根据配置，动态地加载并初始化与LLM交互所需的三个关键组件：
        - `chat_prompt`: 提示构造器。
        - `chat_schema`: 期望LLM返回的JSON结构生成器。
        - `chat_endpoint`: 实际与LLM API通信的端点。
    3.  **配置记忆参数 (`_config_memory_settings`)**: 初始化记忆系统中各种复杂的参数，如不同记忆层的衰减率、重要性阈值等。这些参数决定了记忆如何被遗忘、筛选和强化。
    4.  **构建投资组合 (`construct_portfolio`)**: 创建一个 `Portfolio` 实例，用于跟踪代理的资产状况。

#### 核心方法: `step()`

`step()` 是代理的“主函数”，它封装了代理在一个交易日内的完整认知-决策循环。

```python
# src/agent.py

def step(
    self, market_info: OneDayMarketInfo, run_mode: RunMode, task_type: TaskType
) -> None:
    logger.info("AGENT-Handling new information")
    self._handling_new_information(market_info=market_info)
    
    logger.info("AGENT-Querying memories")
    queried_memories = self._query_memories()

    if task_type == TaskType.SingleAsset:
        self._single_asset_trade_action(...)
    else:
        self._multi_assets_trade_action(...)

    # ... Memory decay, clean up, and flow ...
```

- **输入参数**:
    - `market_info`: 从 `MarketEnv` 接收到的 `OneDayMarketInfo` 对象，包含了当天的所有市场数据。
    - `run_mode`: 当前的运行模式 (`WARMUP` 或 `TEST`)。
    - `task_type`: 任务类型（单资产或多资产）。
- **核心逻辑 (一个完整的决策流程)**:
    1.  **信息处理 (`_handling_new_information`)**: 代理首先处理 `market_info` 中的新信息。最主要的是，它会将当天的新闻 (`cur_news`) 转化为记忆片段，并存入 `MemoryDB` 的**短期记忆层**。
    2.  **记忆检索 (`_query_memories`)**: 这是决策的关键一步。代理会向 `MemoryDB` 的**所有记忆层**（短期、中期、长期、反思）发起查询。查询的内容是基于代理人设的固定问题（例如，“我对这只股票的总体看法是什么？”）。`MemoryDB` 会利用向量相似度结合记忆的重要性、新近度得分，返回最相关的记忆片段。
    3.  **执行交易动作 (`_single_asset_trade_action` / `_multi_assets_trade_action`)**: 这是与 LLM 交互的核心环节。
        - **构建提示**: 代理调用 `self.chat_prompt`，将**当前的市场信息**（价格、动量）和**上一步检索到的各层记忆**组合成一个非常长的、结构化的提示。
        - **构建Schema**: 同时，调用 `self.chat_schema` 生成一个规定了LLM应如何回答的JSON格式。
        - **调用LLM**: `self.chat_endpoint` 将提示和Schema发送给LLM。
        - **处理响应**: LLM返回一个JSON对象，其中包含了交易决策（`investment_decision`）、决策理由（`summary_reason`）以及它做出决策所依据的记忆ID（`short_memory_ids`等）。
        - **记录行动**: 代理解析LLM的响应，并将交易指令（买/卖/持有）、价格、以及作为决策证据的记忆ID列表，一同记录到 `self.portfolio` 中。
        - **创建反思记忆**: LLM生成的 `summary_reason` 被认为是一种更高层次的思考，它会被作为一个新的记忆存入 `MemoryDB` 的**反思记忆层**。
    4.  **更新记忆状态**: 在一个决策周期结束后，代理会执行一系列记忆维护操作：
        - **衰减 (`decay`)**: 所有记忆的重要性（importance）和新近度（recency）得分都会随着时间推移而降低。
        - **清理 (`clean_up`)**: 得分过低的“不重要”记忆会被从数据库中删除。
        - **流动 (`memory_flow`)**: 记忆会在不同层次间流动。例如，一个在短期记忆中被频繁访问、重要性得分足够高的记忆，会自动“晋升”到中期记忆层。反之，长期记忆如果长期不被访问，也可能“降级”到中期记忆。这是实现长期学习的关键机制。
    5.  **更新反馈 (`_update_feedback_response`)**: 代理从 `portfolio` 获取上一步交易的反馈（例如，这个决策是否导致了盈利），然后用这个反馈来更新那些作为决策依据的记忆的重要性得分。正确的决策会强化相关记忆，错误的则会削弱。

---

## 3. 核心概念的实现

### 3.1 交易决策 (`_single_asset_trade_action`)

让我们更深入地看看LLM交互的细节：

```python
# src/agent.py

def _single_asset_trade_action(...):
    # ...
    cur_prompt = self.chat_prompt(...)
    cur_schema = self.chat_schema(...)
    cur_response = self.chat_endpoint(prompt=cur_prompt, schema=cur_schema)
    # ...
```

这三行代码是整个智能的核心。
- `cur_prompt` 精心打包了代理此刻“看到”和“想到”的一切。
- `cur_schema` 像一个严格的指令，告诉LLM必须以机器可读的格式回答。
- `cur_response` 包含了LLM经过“思考”后返回的结构化输出。

如果 `cur_response` 因为任何原因（网络问题、LLM输出格式错误）导致解析失败，代理会默认执行 `TradeAction.HOLD`，这是一个保证系统稳定性的容错机制。

### 3.2 WARMUP 模式

在 `step` 方法中，`run_mode` 参数起到了关键作用。

- **`RunMode.TEST`**: 在测试模式下，代理完全自主决策。它从LLM处获取交易指令 `cur_response.investment_decision` 并执行。
- **`RunMode.WARMUP`**: 在预热模式下，代理依然会执行完整的思考流程（查询记忆、调用LLM等），但它**不会**执行LLM返回的决策。取而代之，它会调用 `_get_warmup_trade_action` 方法，该方法会利用环境提供的“未来信息”(`cur_future_price_diff`)来生成一个完美的、有监督的交易指令。

这种设计的目的是：在正式测试开始前，让代理的记忆库（特别是反思记忆）通过一个有监督的阶段被“预热”，填充一些高质量的初始经验，从而避免在测试初期由于缺乏经验而做出完全随机的决策。

---

## 4. 二次开发指引

- **改变代理行为**:
    - 如果您想让代理更保守或更激进，最直接的方式是修改 `src/chat/prompt/` 中的提示模板，在人设描述中加入相应的指令。
- **扩展代理能力**:
    - 假设您想让代理能够执行除了买卖之外的新动作（例如，查询某个公司的财报详情）。
    1.  您需要在 `src/portfolio_tools.py` 中定义一个新的工具函数。
    2.  修改 `src/chat/prompt/`，将新工具的描述和用法添加到提示中。
    3.  修改 `src/chat/structure_generation/` 中的JSON Schema，使其能够容纳新的动作类型。
    4.  在 `FinMemAgent.step()` 中增加逻辑，以处理LLM返回的新动作。
- **更换代理模型**:
    - 该项目只实现了一个 `FinMemAgent`。您可以模仿 `agent.py` 的结构，创建一个全新的代理类，例如 `MyAwesomeAgent`。只要您的新代理类也实现了 `step`, `save_checkpoint`, `load_checkpoint` 等核心方法，您就可以在 `run.py` 中轻松地替换它，而无需改动框架的其他部分。 