# 为了向后兼容，保留原始接口
# 但现在使用统一的配置系统

from .embedding_unified import (
    EmbeddingModel,
    UnifiedOpenAIEmbedding,
    UnifiedEmbeddingError,
    EmbeddingObject,
    EmbeddingResponse
)

# 为了向后兼容，重新导出为原来的名称
OpenAIEmbedding = UnifiedOpenAIEmbedding
OpenAIEmbeddingError = UnifiedEmbeddingError
