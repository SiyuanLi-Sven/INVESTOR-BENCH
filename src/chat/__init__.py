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
    OpenAIEndpoint,
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
    if chat_config["chat_model_inference_engine"] == "vllm":
        logger.trace("SYS-Chat model is VLLM")
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
    elif chat_config["chat_model_inference_engine"] == "openai":
        if task_type == TaskType.SingleAsset:
            # 使用新的OpenAI端点实现
            logger.trace("SYS-Chat model is OpenAI (Direct)")
            return (
                GuardrailStructureGenerationSchema(),
                OpenAIEndpoint(chat_config=chat_config),
                GuardrailPromptConstructor(),
            )
        else:
            # 使用新的OpenAI端点实现，但目前仅支持单资产
            logger.error("SYS-Multi-asset not implemented for OpenAI")
            raise NotImplementedError("Multi-asset not implemented for OpenAI")
    elif chat_config["chat_model_inference_engine"] == "openai-guardrails":
        # 保留原来的guardrails实现，以向后兼容
        if task_type == TaskType.SingleAsset:
            logger.trace("SYS-Chat model is OpenAI (Guardrails)")
            return (
                GuardrailStructureGenerationSchema(),
                GPTGuardRailStructureGeneration(chat_config=chat_config),
                GuardrailPromptConstructor(),
            )
        else:
            logger.error("SYS-Multi-asset not implemented for OpenAI")
            raise NotImplementedError("Multi-asset not implemented for OpenAI")
    elif chat_config["chat_model_inference_engine"] == "anthropic":
        if task_type == TaskType.SingleAsset:
            return (
                GuardrailStructureGenerationSchema(),
                ClaudeGuardRailStructureGeneration(chat_config=chat_config),
                GuardrailPromptConstructor(),
            )
        else:
            raise NotImplementedError("Multi-asset not implemented for Claude")
    else:
        logger.error(
            f"SYS-Model {chat_config['chat_model_inference_engine']} not implemented"
        )
        logger.error("SYS-Exiting")
        raise NotImplementedError(
            f"Model {chat_config['chat_model_inference_engine']} not implemented"
        )
