"""
简化版的guardrail structure generation，移除ValidChoices避免序列化问题
"""
from typing import List, Union
from pydantic import BaseModel, Field

from ...utils import RunMode
from .base import (
    SingleAssetBaseStructureGenerationSchema as BaseStructureGenerationSchema,
)

# prompt template
warmup_memory_id_extract_prompt = "Provide the piece of information  related the most to the investment decisions from mainstream sources such as the investment suggestions major fund firms such as ARK, Two Sigma, Bridgewater Associates, etc. in the {memory_layer} memory?"
test_memory_id_extract_prompt = "Provide the piece of information related most to your investment decisions in the {memory_layer} memory?"
short_memory_id_desc = "The id of the short-term information."
mid_memory_id_desc = "The id of the mid-term information."
long_memory_id_desc = "The id of the long-term information."
reflection_memory_id_desc = "The id of the reflection-term information."
warmup_trade_reason_summary = "Given a professional trader's trading suggestion, can you explain to me why the trader drive such a decision with the information provided to you?"
test_invest_action_choice = "Given the information, please make an investment decision: buy the stock, sell, and hold the stock"
test_trade_reason_summary = "Given the information of text and the summary of the stock price movement. Please explain the reason why you make the investment decision."


def _train_memory_factory(memory_layer: str, id_list: List[int]):
    class Memory(BaseModel):
        memory_index: int = Field(
            ...,
            description=warmup_memory_id_extract_prompt.format(
                memory_layer=memory_layer
            ),
        )

    return Memory


def _test_memory_factory(memory_layer: str, id_list: List[int]):
    class Memory(BaseModel):
        memory_index: int = Field(
            ...,
            description=test_memory_id_extract_prompt.format(memory_layer=memory_layer),
        )

    return Memory


def _train_reflection_factory(
    short_id_list: Union[List[int], None],
    mid_id_list: Union[List[int], None],
    long_id_list: Union[List[int], None],
):
    ShortMem = _train_memory_factory("short-level", short_id_list) if short_id_list else None
    MidMem = _train_memory_factory("mid-level", mid_id_list) if mid_id_list else None
    LongMem = _train_memory_factory("long-level", long_id_list) if long_id_list else None

    class ReflectionInfo(BaseModel):
        trade_reason: str = Field(
            ...,
            description=warmup_trade_reason_summary,
        )
        # 动态添加字段
        pass

    # 动态添加字段
    if short_id_list:
        ReflectionInfo.model_fields["short_memory_ids"] = Field(
            ..., 
            description=short_memory_id_desc
        )
    if mid_id_list:
        ReflectionInfo.model_fields["mid_memory_ids"] = Field(
            ..., 
            description=mid_memory_id_desc
        )
    if long_id_list:
        ReflectionInfo.model_fields["long_memory_ids"] = Field(
            ..., 
            description=long_memory_id_desc
        )

    return ReflectionInfo


def _test_reflection_factory(
    short_id_list: Union[List[int], None],
    mid_id_list: Union[List[int], None],
    long_id_list: Union[List[int], None],
    reflection_id_list: Union[List[int], None],
):
    ShortMem = _test_memory_factory("short-level", short_id_list) if short_id_list else None
    MidMem = _test_memory_factory("mid-level", mid_id_list) if mid_id_list else None
    LongMem = _test_memory_factory("long-level", long_id_list) if long_id_list else None
    ReflectionMem = _test_memory_factory("reflection-level", reflection_id_list) if reflection_id_list else None

    class InvestInfo(BaseModel):
        investment_decision: str = Field(
            ...,
            description=test_invest_action_choice,
        )
        summary_reason: str = Field(
            ...,
            description=test_trade_reason_summary,
        )

    # 动态添加字段
    if short_id_list:
        InvestInfo.model_fields["short_memory_ids"] = Field(
            ..., 
            description=short_memory_id_desc
        )
    if mid_id_list:
        InvestInfo.model_fields["mid_memory_ids"] = Field(
            ..., 
            description=mid_memory_id_desc
        )
    if long_id_list:
        InvestInfo.model_fields["long_memory_ids"] = Field(
            ..., 
            description=long_memory_id_desc
        )
    if reflection_id_list:
        InvestInfo.model_fields["reflection_memory_ids"] = Field(
            ..., 
            description=reflection_memory_id_desc
        )

    return InvestInfo


def GuardRailStructureGeneration(
    run_mode: RunMode,
    short_id_list: Union[List[int], None] = None,
    mid_id_list: Union[List[int], None] = None,
    long_id_list: Union[List[int], None] = None,
    reflection_id_list: Union[List[int], None] = None,
) -> BaseStructureGenerationSchema:

    if run_mode == RunMode.WARMUP:
        reflection_schema = _train_reflection_factory(
            short_id_list,
            mid_id_list,
            long_id_list,
        )
    elif run_mode == RunMode.TEST:
        reflection_schema = _test_reflection_factory(
            short_id_list,
            mid_id_list,
            long_id_list,
            reflection_id_list,
        )
    else:
        raise ValueError(f"Invalid run mode: {run_mode}")

    return BaseStructureGenerationSchema(reflection_schema=reflection_schema)