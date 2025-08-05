# 模块详解: 与LLM的交互 (LLM Interaction)

**文件路径**: `src/chat/`

本篇文档是 `Investor-Bench` 核心模块解析的最后一站，我们将深入探讨代理是如何与大型语言模型（LLM）进行通信的。这个交互层被精心设计为三个解耦的子模块，共同协作，以实现高效、可靠且可控的“人机对话”。这三个子模块都位于 `src/chat/` 目录下。

## 1. 核心设计思想

与LLM的交互是整个代理智能行为的来源，其设计遵循了现代LLM应用的黄金法则：

- **分离关注点 (Separation of Concerns)**: 将一个复杂的任务（与LLM对话）分解为三个独立、清晰的步骤：
    1.  **我该说什么？(Prompting)**: 负责动态地组织和格式化要发送给LLM的信息。
    2.  **我希望你怎么回答？(Structuring)**: 负责定义并强制LLM返回的响应必须遵循的精确格式。
    3.  **我们来谈谈吧。(Endpoint)**: 负责处理实际的API通信、错误处理和重试逻辑。
- **抽象与可扩展性**: 每个子模块都定义了一个抽象基类（ABC），使得添加对新型号、新提示策略或新LLM服务的支持变得非常简单，只需继承基类并实现相应方法即可。
- **鲁棒性 (Robustness)**: 交互层内置了强大的容错机制。如果LLM的返回结果不符合预定义的结构，系统不会崩溃，而是会优雅地回退到一个安全的默认行为（例如，持有仓位）。

下面是这三个子模块协同工作的流程图：

```mermaid
graph TD
    subgraph FinMemAgent 的决策流程中
        A[1. 准备数据<br/>(当前市场信息, 检索到的记忆)] --> B;
        B(2. 调用 Prompt 构造器) -- 输入数据 --> C[prompt.py];
        C -- 输出 --> D[格式化的 Prompt 字符串];
        
        A --> E;
        E(3. 调用 Schema 构造器) -- 输入记忆ID等 --> F[structure_generation.py];
        F -- 输出 --> G[定义了响应格式的 JSON Schema];
        
        D --> H;
        G --> H;
        H(4. 调用 Endpoint) -- 传入 Prompt 和 Schema --> I[endpoint.py];
        I -- 发送 API 请求 --> J[LLM 服务 (如OpenAI)];
        J -- 返回 JSON 响应 --> I;
        I -- 解析和验证响应 --> K[结构化的 Pydantic 对象];
        K -- 返回给 --> FinMemAgent;
    end
```

---

## 2. 子模块详解

### 2.1 `prompt/` - 提示工程

**职责**: 构造发送给LLM的最终输入文本。

- **基类**: `SingleAssetBasePromptConstructor`, `MultiAssetBasePromptConstructor`
- **实现**: `vllm_prompt.py` (尽管命名为vllm，但其逻辑是通用的)

提示是整个系统的灵魂。在 `vllm_prompt.py` 中，您可以看到一个巨大的 f-string 模板。这个模板将 `FinMemAgent` 传递过来的所有上下文信息——**当前日期、股价、动量、以及从短期到反思四个层次的记忆**——全部动态地、有条理地组织成一个连贯的文本。

**核心逻辑**:
- **人设设定 (Persona Setting)**: 提示的开头部分为LLM设定了一个非常具体的人设，例如“你是一位专业的金融分析师...”。
- **信息分段**: 所有信息都被清晰地标记和分段（例如，`## Short Term Memory`, `## Market Information`），以帮助LLM更好地理解上下文。
- **指令清晰**: 提示的结尾部分给出了明确的指令，要求LLM基于以上信息，以指定的JSON格式完成思考和决策。
- **WARMUP模式的特殊处理**: 在 `WARMUP` 模式下，提示中会包含 `future_record` (未来价格信息)，并明确指示LLM将其作为正确答案来学习，从而生成高质量的反思记忆。

### 2.2 `structure_generation/` - 结构化输出

**职责**: 定义并生成一个JSON Schema，用于强制约束LLM的输出格式。

- **基类**: `SingleAssetBaseStructureGenerationSchema`, `MultiAssetsBaseStructureGenerationSchema`
- **实现**: `guardrail_sg.py`, `vllm_sg.py`

如果说提示是告诉LLM“思考什么”，那么Schema就是告诉它“如何表达你的思考”。通过向LLM API提供一个JSON Schema，我们可以极大地提高其输出的稳定性和可靠性，使其返回的不再是自由文本，而是可以直接被程序解析的JSON对象。

**核心逻辑 (`guardrail_sg.py`)**:
- **动态构建Schema**: Schema是动态生成的。例如，对于证据字段（`short_memory_ids`等），它会根据当前实际检索到的记忆ID列表，将这些ID作为 `enum` 值动态地插入到Schema中。这确保了LLM只能从提供给它的记忆中选择证据，而不会“幻觉”出不存在的记忆ID。
- **字段定义**: 定义了LLM响应中必须包含的字段，例如：
    - `investment_decision`: 交易决策，必须是 `BUY`, `SELL`, `HOLD` 之一。
    - `summary_reason`: 做出该决策的详细文字理由。
    - `short_memory_ids`, `mid_memory_ids`, etc.: 作为决策依据的记忆ID列表。

### 2.3 `endpoint/` - 通信端点

**职责**: 管理与LLM服务的实际API通信。

- **基类**: `SingleAssetStructuredGenerationChatEndPoint`, `MultiAssetsStructuredGenerationChatEndPoint`
- **实现**: `guardrails.py`, `vllm.py`

这个模块是与外部世界连接的桥梁。它接收`prompt`和`schema`，将它们打包成一个API请求，发送出去，然后接收并解析返回的结果。

**核心逻辑 (`guardrails.py`)**:
- **使用 `Guardrails AI`**: 该实现利用了 `Guardrails AI` 这个库来与 OpenAI 兼容的 API 进行通信。`Guardrails` 的一个核心功能就是能够将JSON Schema（或其自定义的RAIL格式）附加到API请求中，并自动验证返回的JSON是否符合规范。
- **请求构造**: 它将 `prompt` 和 `schema` 格式化为符合特定LLM服务（如OpenAI Chat Completion API）要求的请求体。
- **响应处理与容错**:
    - **成功**: 如果LLM返回了符合Schema的JSON，它会被解析成一个预定义的 `Pydantic` 模型（如 `SingleAssetStructureOutputResponse`）并返回。
    - **失败**: 如果LLM的返回无法通过Schema验证（例如，缺少字段、类型错误），`Guardrails` 会捕获这个错误。此时，端点不会让程序崩溃，而是会返回一个预定义的失败对象 `SingleAssetStructureGenerationFailure`，其中包含一个安全的默认动作 `TradeAction.HOLD`。

---

## 3. 二次开发指引

- **支持新的LLM服务**:
    - 假设您想接入一个不兼容OpenAI API的新模型（例如，Anthropic的Claude）。
    1.  在 `endpoint/` 目录下创建一个新文件，例如 `claude_endpoint.py`。
    2.  创建一个新类 `ClaudeEndpoint`，继承自 `SingleAssetStructuredGenerationChatEndPoint`。
    3.  在新类中实现 `__call__` 方法，编写调用Claude API的特定逻辑。您可能需要找到一个支持结构化输出的适用于Claude的库，或者手动处理重试和验证。
    4.  最后，在 `src/chat/__init__.py` 的 `get_chat_model` 函数中，增加一个分支，当 `chat_config` 中指定使用 "claude" 时，就实例化您的 `ClaudeEndpoint`。
- **修改提示策略**:
    - 这是最常见的二次开发需求。您可以直接修改 `prompt/vllm_prompt.py` 中的提示模板，尝试不同的措辞、人设或信息组织方式，来引导LLM产生不同的行为模式。
- **调整输出结构**:
    - 如果您想让LLM在决策之外返回更多信息（例如，一个置信度分数）。
    1.  在 `structure_generation/guardrail_sg.py` 中的 `SINGLE_ASSET_TRADE_ACTION_RAIL_STR` (这是一个定义输出结构的字符串) 中，增加一个新的字段定义，如 `<float name="confidence_score" ... />`。
    2.  在 `endpoint/base.py` 中的 `SingleAssetStructureOutputResponse` Pydantic模型里，也增加一个对应的字段 `confidence_score: float`。
    3.  修改 `prompt`，告诉LLM现在需要额外输出一个置信度分数。 