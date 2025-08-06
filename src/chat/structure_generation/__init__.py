from .base import (
    SingleAssetBaseStructureGenerationSchema,
    MultiAssetsBaseStructureGenerationSchema,
)
from .vllm_sg import (
    SingleAssetVLLMStructureGenerationSchema,
    MultiAssetsVLLMStructureGenerationSchema,
)

# 尝试导入guardrail，如果失败则创建一个placeholder
try:
    from .guardrail_sg import GuardrailStructureGenerationSchema
except ImportError as e:
    import warnings
    warnings.warn(f"Failed to import GuardrailStructureGenerationSchema: {e}. Creating placeholder.")
    
    # 创建一个具体的placeholder类
    from typing import Dict, List, Union
    from ...utils import RunMode
    from .base import SingleAssetBaseStructureGenerationSchema
    
    class GuardrailStructureGenerationSchema(SingleAssetBaseStructureGenerationSchema):
        @staticmethod
        def __call__(
            run_mode: RunMode,
            short_memory_ids: Union[List[int], None] = None,
            mid_memory_ids: Union[List[int], None] = None,
            long_memory_ids: Union[List[int], None] = None,
            reflection_memory_ids: Union[List[int], None] = None,
        ):
            # 返回一个简单的placeholder schema
            from pydantic import BaseModel, Field
            from typing import Optional
            
            class PlaceholderSchema(BaseModel):
                investment_decision: Optional[str] = Field(default="hold")
                summary_reason: str = Field(default="Placeholder response due to guardrails import error")
                
            return PlaceholderSchema
