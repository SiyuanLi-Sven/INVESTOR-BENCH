# 模块详解: 记忆系统 (Memory)

**文件路径**: `src/memory_db.py`, `src/embedding.py`

本篇文档深入探讨 `Investor-Bench` 代理的认知核心——其复杂且动态的**记忆系统**。这个系统远不止是一个简单的信息存储库；它是一个模仿生物记忆机制的、分层的、基于向量的数据库，赋予了代理学习、遗忘和联想的能力。

## 1. 核心设计思想

代理的记忆系统旨在克服 LLM 的两大固有局限：有限的上下文窗口和无状态性。其设计理念基于以下几点：

- **分层记忆 (Layered Memory)**: 模仿人类记忆模型，将信息存储在不同的层次中。新信息进入**短期记忆**，重要的信息会逐渐沉淀到**中期**和**长期记忆**，而更高层次的归纳和总结则存放在**反思记忆**中。
- **基于相关性的检索 (Relevance-Based Retrieval)**: 使用向量嵌入（Vector Embeddings）来度量信息的语义相似度。当代理需要回忆时，它会根据当前情境的“相关性”来检索记忆，而不是进行关键词匹配。
- **动态评分系统 (Dynamic Scoring)**: 每个记忆片段都附带着动态变化的**重要性（Importance）**和**新近度（Recency）**得分。这两个得分决定了记忆的“显著性”，直接影响它被检索的概率和在层次间流动的方式。
- **生命周期管理 (Lifecycle Management)**: 记忆有其生命周期。它们会随着时间**衰减（Decay）**，变得不那么重要；得分过低的记忆会被**清理（Clean Up）**；而重要的记忆则会在层次间**流动（Flow）**。

这个系统的技术实现主要依赖于 `Qdrant`，一个高性能的开源向量数据库。

---

## 2. 关键组件与流程

### 2.1 `MemoryDB` 类

这是记忆系统的主要接口类，封装了与 `Qdrant` 数据库的所有交互逻辑。

#### 2.1.1 记忆的存储 (`add_memory`)

当代理需要记事时（例如，处理当天的新闻），它会调用 `add_memory` 方法。

```python
# src/memory_db.py

def add_memory(
    self,
    memory_input: List[Dict],
    layer: str,
    importance_init_func: ConstantImportanceInitialization,
    recency_init_func: ConstantRecencyInitialization,
    # ...
) -> List[NonNegativeInt]:
    # ...
    to_emb_texts = [m.text for m in memories_records]
    text_embs = self.emb_model(texts=to_emb_texts)
    # ...
    points.append(
        PointStruct(
            id=cur_m.id,
            payload={
                "symbol": ..., "date": ..., "text": ...,
                "delta": 0,
                "importance": importance_init_func(),
                "recency": recency_init_func(),
                "access_counter": 0,
                "layer": layer,
            },
            vector=cur_emb,
        )
    )
    # ...
    self.connection_client.upsert(..., points=points, ...)
```

- **核心逻辑**:
    1.  **文本嵌入**: `memory_input` 中的文本内容首先被送入 `self.emb_model` (`OpenAIEmbedding` 的实例) 进行处理，转化成能够表示其语义的向量（Vector Embedding）。
    2.  **构造数据点 (`PointStruct`)**: 每个记忆片段在 `Qdrant` 中都是一个“数据点”（Point）。这个数据点包含三个核心部分：
        - `id`: 一个独一无二的整数ID，由 `IDGenerator` 生成。
        - `vector`: 上一步生成的文本嵌入向量。
        - `payload`: 一个包含了所有元数据（metadata）的JSON对象。这是记忆评分系统的关键所在。
    3.  **Payload 详解**:
        - `text`, `symbol`, `date`: 记忆的原始内容。
        - `layer`: 该记忆所属的层次（"short", "mid", "long", "reflection"）。
        - `importance`: **重要性得分**。通过 `importance_init_func` 初始化。
        - `recency`: **新近度得分**。通过 `recency_init_func` 初始化，通常为1.0。
        - `delta`: 距离上次更新过去了多少天，初始为0。
        - `access_counter`: 这个记忆被访问过多少次。
    4.  **存入数据库**: 构造好的数据点通过 `upsert` 命令批量写入 `Qdrant` 数据库。

#### 2.1.2 记忆的检索 (`query`)

当代理需要根据当前情境回忆相关信息时，会调用 `query` 方法。

```python
# src/memory_db.py

def query(
    self,
    query_input: Queries,
    layer: str,
    linear_compound_func: LinearCompoundScore,
) -> List[Tuple[List[str], List[int]]]:
    # ...
    emb_vector = self.emb_model(texts=to_emb)
    # ...
    search_results = self.connection_client.search_batch(...)
    # ...
    cur_result_subset = sorted(
        [
            {
                "compound_score": linear_compound_func(
                    similarity_score=r.score,
                    importance_score=r.payload["importance"],
                    recency_score=r.payload["recency"],
                ),
                "text": r.payload["text"],
                "id": r.id,
            }
            # ...
        ],
        key=lambda x: -x["compound_score"],
    )[: cur_query["k"]]
    # ...
```

- **核心逻辑**:
    1.  **查询嵌入**: 首先，查询的文本（`query_input`）本身也被转换成向量。
    2.  **向量搜索**: `Qdrant` 执行一次向量相似度搜索，找出在指定 `layer` 中与查询向量最相似的一批记忆点。`r.score` 就是数据库返回的原始相似度得分。
    3.  **复合评分 (`compound_score`)**: **这是检索机制的精髓**。系统并不仅仅依赖向量相似度。它会使用 `linear_compound_func` 将三个维度结合起来，计算一个最终的“复合分数”：
        - `similarity_score`: 向量搜索返回的语义相似度。
        - `importance_score`: 记忆自身的重要性。
        - `recency_score`: 记忆的新近度。
        **复合分数 = 相似度 + 标准化重要性 + 新近度**
    4.  **排序与筛选**: `Qdrant` 返回的所有候选项会根据这个**复合分数**进行降序排序，然后取出前 `k` 个作为最终的检索结果。这意味着，一个记忆即使与查询不那么相似，但如果它被标记为非常重要或刚刚发生，它仍然有很大概率被回忆起来。

### 2.2 记忆的生命周期管理

代理在每个 `step` 的末尾都会执行一系列维护操作，以模拟记忆的动态演化。

#### 2.2.1 衰减 (`decay`)

```python
# src/memory_db.py
def decay(..., importance_decay_func: ImportanceDecay, recency_decay_func: RecencyDecay, ...):
    # ...
    cur_new_importance = importance_decay_func(cur_val=r["payload"]["importance"])
    cur_new_recency = recency_decay_func(delta=cur_new_delta)
    # ...
```

- **逻辑**: 对于数据库中的每一条记忆，它的 `importance` 得分会乘以一个小于1的衰减因子，`recency` 得分则会根据其存在的时间（`delta`）呈指数衰减。这模拟了“遗忘”过程。

#### 2.2.2 清理 (`clean_up`)

- **逻辑**: 在衰减之后，任何 `importance` 或 `recency` 得分低于预设阈值的记忆都会被从数据库中彻底删除。这防止了数据库被大量无用的信息填满。

#### 2.2.3 流动 (`memory_flow`)

这是实现长期学习最有趣的部分。

```python
# src/memory_db.py
def memory_flow(...):
    # short -> mid
    cur_short_up_mem = self.prepare_jump(..., layer="short", threshold=...)
    self.accept_jump(..., jump_dict=cur_short_up_mem, ..., target_layer="mid")
    
    # mid -> long
    cur_mid_up_mem = self.prepare_jump(..., layer="mid", threshold=...)
    self.accept_jump(..., jump_dict=cur_mid_up_mem, ..., target_layer="long")

    # mid -> short (demotion)
    # long -> mid (demotion)
```

- **逻辑**:
    - **晋升 (Promotion)**: 系统会检查 `short` 层中 `importance` 得分**高于**某个阈值的记忆，将它们从 `short` 层删除，然后重新添加到 `mid` 层。同样，`mid` 层中足够重要的记忆也会被晋升到 `long` 层。在晋升时，它们的 `recency` 得分会被重置，赋予它们“第二次生命”。
    - **降级 (Demotion)**: 反之，`long` 或 `mid` 层中 `importance` 得分**低于**某个阈值的记忆，会被降级回下一层。

### 2.3 `OpenAIEmbedding` 类

**文件路径**: `src/embedding.py`

这个类是对文本嵌入模型的简单封装。它负责通过 HTTP 请求调用 OpenAI 或兼容 OpenAI 格式的 API，并将一批文本转换为一批向量。它的实现很直接，主要是处理 API 的认证和网络请求。

---

## 3. 二次开发指引

- **更换向量数据库**:
    - 如果您想将 `Qdrant` 更换为其他向量数据库（如 Milvus, Weaviate），您需要重写 `MemoryDB` 类。您需要确保新的类同样提供 `add_memory`, `query`, `decay` 等核心接口，这样上层的 `FinMemAgent` 代码就无需改动。
- **调整记忆机制**:
    - 您可以非常轻松地在 `FinMemAgent._config_memory_settings` 中调整各种参数（衰减率、阈值等）来观察代理行为的变化。
    - 您也可以在 `memory_db.py` 中实现新的评分函数（例如，非线性的 `LinearCompoundScore`）来改变记忆的检索方式。
- **更换嵌入模型**:
    - 如果您想使用自己的、或者其他非 OpenAI 的嵌入模型，您只需创建一个新的类，继承自 `EmbeddingModel` 基类，并实现其 `__call__` 方法。然后，在 `MemoryDB` 的初始化中实例化您的新模型即可。 