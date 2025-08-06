from typing import Dict, Tuple, Union

from loguru import logger

from .endpoint import (
    SingleAssetStructuredGenerationChatEndPoint,
    MultiAssetsStructuredGenerationChatEndPoint,
    SingleAssetVLLMStructureGeneration,
    MultiAssetsVLLMStructureGeneration,
    SingleAssetStructureGenerationFailure,
    MultiAssetsStructureGenerationFailure,
    SingleAssetStructureOutputResponse,
    MultiAssetsStructureOutputResponse,
    GPTGuardRailStructureGeneration,
    ClaudeGuardRailStructureGeneration,
    SingleAssetOpenAICompatibleGeneration,
    MultiAssetsOpenAICompatibleGeneration,
)

from .prompt import (
    SingleAssetBasePromptConstructor,
    MultiAssetBasePromptConstructor,
    SingleAssetVLLMPromptConstructor,
    MultiAssetsVLLMPromptConstructor,
    GuardrailPromptConstructor,
)

from .structure_generation import (
    SingleAssetBaseStructureGenerationSchema,
    MultiAssetsBaseStructureGenerationSchema,
    SingleAssetVLLMStructureGenerationSchema,
    MultiAssetsVLLMStructureGenerationSchema,
    GuardrailStructureGenerationSchema,
)

from ..utils import TaskType

single_asset_return_type = Tuple[
    SingleAssetBaseStructureGenerationSchema,
    SingleAssetStructuredGenerationChatEndPoint,
    SingleAssetBasePromptConstructor,
]

multi_asset_return_type = Tuple[
    MultiAssetsBaseStructureGenerationSchema,
    MultiAssetsStructuredGenerationChatEndPoint,
    MultiAssetBasePromptConstructor,
]


def get_chat_model(
    chat_config: Dict, task_type: TaskType
) -> Union[single_asset_return_type, multi_asset_return_type]:
    logger.trace("SYS-Initializing chat model, prompt, and schema")
    
    inference_engine = chat_config.get("chat_model_inference_engine", "openai_compatible")
    
    # 新的统一OpenAI兼容接口 (推荐使用)
    if inference_engine == "openai_compatible" or inference_engine == "unified":
        logger.trace("SYS-使用统一OpenAI兼容接口")
        if task_type == TaskType.SingleAsset:
            return (
                GuardrailStructureGenerationSchema(),  # 使用现有的schema
                SingleAssetOpenAICompatibleGeneration(chat_config=chat_config),
                GuardrailPromptConstructor(),  # 使用现有的prompt构造器
            )
        else:
            return (
                MultiAssetsVLLMStructureGenerationSchema(),  # 使用现有的schema
                MultiAssetsOpenAICompatibleGeneration(chat_config=chat_config),
                MultiAssetsVLLMPromptConstructor(),  # 使用现有的prompt构造器
            )
    
    # 保留原有的VLLM直接调用方式（向后兼容）
    elif inference_engine == "vllm":
        logger.trace("SYS-Chat model is VLLM (legacy mode)")
        if task_type == TaskType.SingleAsset:
            return (
                SingleAssetVLLMStructureGenerationSchema(),
                SingleAssetVLLMStructureGeneration(chat_config=chat_config),
                SingleAssetVLLMPromptConstructor(),
            )
        else:
            return (
                MultiAssetsVLLMStructureGenerationSchema(),
                MultiAssetsVLLMStructureGeneration(chat_config=chat_config),
                MultiAssetsVLLMPromptConstructor(),
            )
    
    # 保留原有的OpenAI Guardrails方式（向后兼容）
    elif inference_engine == "openai":
        logger.trace("SYS-Chat model is OpenAI (legacy guardrails mode)")
        if task_type == TaskType.SingleAsset:
            return (
                GuardrailStructureGenerationSchema(),
                GPTGuardRailStructureGeneration(chat_config=chat_config),
                GuardrailPromptConstructor(),
            )
        else:
            raise NotImplementedError("Multi-asset not implemented for OpenAI legacy mode")
    
    # 保留原有的Anthropic方式（向后兼容）
    elif inference_engine == "anthropic":
        logger.trace("SYS-Chat model is Anthropic (legacy guardrails mode)")
        if task_type == TaskType.SingleAsset:
            return (
                GuardrailStructureGenerationSchema(),
                ClaudeGuardRailStructureGeneration(chat_config=chat_config),
                GuardrailPromptConstructor(),
            )
        else:
            raise NotImplementedError("Multi-asset not implemented for Claude legacy mode")
    
    else:
        logger.error(f"SYS-Model inference engine {inference_engine} not implemented")
        logger.error("SYS-Supported engines: openai_compatible, unified, vllm, openai, anthropic")
        logger.error("SYS-Exiting")
        raise NotImplementedError(
            f"Model inference engine {inference_engine} not implemented. "
            f"Supported: openai_compatible, unified, vllm, openai, anthropic"
        )
