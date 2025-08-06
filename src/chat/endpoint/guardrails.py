import json
import os
import time
from typing import Any, Callable, Dict, Tuple, Union

import guardrails as gd
import httpx
from loguru import logger

from .base import (
    SingleAssetStructuredGenerationChatEndPoint as StructuredGenerationChatEndPoint,
)
from .base import (
    SingleAssetStructureGenerationFailure as StructureGenerationFailure,
)
from .base import (
    SingleAssetStructureOutputResponse as StructureOutputResponse,
)
from .base import (
    delete_placeholder_info,
)


class BaseGuardRailStructureGeneration(StructuredGenerationChatEndPoint):
    def __init__(self, chat_config: Dict[str, Any]) -> None:
        self.chat_config = chat_config
        self.chat_model = chat_config["chat_model"]
        self.chat_max_new_token = chat_config["chat_max_new_token"]
        self.chat_model_type = chat_config["chat_model_type"]
        self.endpoint = chat_config["chat_endpoint"]
        self.chat_request_timeout = chat_config["chat_request_timeout"]
        self.chat_parameters = chat_config["chat_parameters"]
        
        # 添加重试配置
        self.max_retries = 3
        self.retry_delay = 5  # 秒

        self.chat_end_point_func = self.endpoint_func()

    def endpoint_func(self) -> Callable[[str], str]:
        raise NotImplementedError("This method should be overridden by subclasses.")

    def __call__(
        self, prompt: Tuple[str, str], schema: Any
    ) -> Union[StructureGenerationFailure, StructureOutputResponse]:
        invest_info_prompt, ask_prompt = prompt
        
        # 使用更短的超时时间并添加重试机制
        guard = gd.Guard.from_pydantic(
            output_class=schema, prompt=ask_prompt, num_reasks=2  # 减少重试次数
        )
        
        # 创建带有错误处理的端点函数
        endpoint_func = self._create_robust_endpoint_func()
        
        try:
            validated_outcomes = guard(
                llm_api=endpoint_func, prompt_params={"investment_info": invest_info_prompt}
            )
        except Exception as e:
            logger.error(f"❌ GuardRails执行失败: {str(e)}")
            # 如果是超时错误，尝试降级处理
            if "timeout" in str(e).lower() or "timed out" in str(e).lower():
                logger.warning("⚠️ 检测到超时错误，尝试降级处理...")
                return self._fallback_generation(invest_info_prompt, ask_prompt)
            return StructureGenerationFailure()

        validated_output_dicts = {}
        if (validated_outcomes.validated_output is None) or not isinstance(  # type: ignore
            validated_outcomes.validated_output,  # type: ignore
            dict,  # type: ignore
        ):
            return StructureGenerationFailure()

        try:
            validated_output_dicts = delete_placeholder_info(
                validated_outcomes.validated_output  # type: ignore
            )
        except json.JSONDecodeError:
            return StructureGenerationFailure()

        if "investment_decision" not in validated_output_dicts:
            validated_output_dicts_out = {
                "summary_reason": validated_output_dicts["summary_reason"]
            }
        else:
            validated_output_dicts_out = {
                "investment_decision": validated_output_dicts["investment_decision"],
                "summary_reason": validated_output_dicts["summary_reason"],
            }
        if "short_memory_ids" in validated_output_dicts:
            validated_output_dicts_out["short_memory_ids"] = [
                item["memory_index"]
                for item in validated_output_dicts["short_memory_ids"]
            ]
        if "mid_memory_ids" in validated_output_dicts:
            validated_output_dicts_out["mid_memory_ids"] = [
                item["memory_index"]
                for item in validated_output_dicts["mid_memory_ids"]
            ]
        if "long_memory_ids" in validated_output_dicts:
            validated_output_dicts_out["long_memory_ids"] = [
                item["memory_index"]
                for item in validated_output_dicts["long_memory_ids"]
            ]
        if "reflection_memory_ids" in validated_output_dicts:
            validated_output_dicts_out["reflection_memory_ids"] = [
                item["memory_index"]
                for item in validated_output_dicts["reflection_memory_ids"]
            ]

        return StructureOutputResponse(**validated_output_dicts_out)
    
    def _create_robust_endpoint_func(self) -> Callable[[str], str]:
        """创建带有重试机制的端点函数"""
        original_func = self.endpoint_func()
        
        def robust_endpoint(prompt: str, **kwargs) -> str:
            last_error = None
            
            for attempt in range(self.max_retries):
                try:
                    # 使用更短的超时时间（最大5分钟）
                    short_timeout = min(self.chat_request_timeout, 300)  
                    
                    # 临时修改超时设置
                    original_timeout = self.chat_request_timeout
                    self.chat_request_timeout = short_timeout
                    
                    try:
                        result = original_func(prompt, **kwargs)
                        # 恢复原始超时设置
                        self.chat_request_timeout = original_timeout
                        return result
                    finally:
                        # 确保恢复原始超时设置
                        self.chat_request_timeout = original_timeout
                        
                except Exception as e:
                    last_error = e
                    logger.warning(f"⚠️ API调用失败 (尝试 {attempt + 1}/{self.max_retries}): {str(e)}")
                    
                    if attempt < self.max_retries - 1:
                        logger.info(f"💤 等待 {self.retry_delay} 秒后重试...")
                        time.sleep(self.retry_delay)
                    
            # 所有重试都失败了
            logger.error(f"❌ 所有重试都失败，最后错误: {str(last_error)}")
            raise last_error
        
        return robust_endpoint
    



class ClaudeGuardRailStructureGeneration(BaseGuardRailStructureGeneration):
    def __init__(self, chat_config: Dict[str, Any]) -> None:
        super().__init__(chat_config)
        self.headers = {
            "content-type": "application/json",
            "x-api-key": os.environ["ANTHROPIC_API_KEY"],
            "anthropic-version": "2023-06-01",
        }

    def endpoint_func(self) -> Callable[[str], str]:
        def end_point(prompt: str, **kwargs) -> str:
            request_data = {
                **{
                    "model": self.chat_model,
                    "messages": [
                        {
                            "role": "user",
                            "content": f"You are a helpful assistant only capable of communicating with valid JSON, and no other text. {prompt}",
                        }
                    ],
                },
                **self.chat_parameters,
            }
            logger.info("LLM API Request sent")
            with httpx.Client(timeout=self.chat_request_timeout) as client:
                response = client.post(
                    url=self.endpoint, headers=self.headers, json=request_data
                )
            if response.status_code != 200:
                logger.error(
                    f"LLM API Request failed with status code {response.status_code}"
                )
                logger.error(f"LLM API Request failed with response {response.json()}")
                return ""
            logger.info("LLM API Request successful")
            response_json = json.loads(response.text)
            return response_json["content"][0]["text"]

        return end_point


class GPTGuardRailStructureGeneration(BaseGuardRailStructureGeneration):
    def __init__(self, chat_config: Dict[str, Any]) -> None:
        super().__init__(chat_config)
        # 支持从.env文件或环境变量获取API key
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def endpoint_func(self) -> Callable[[str], str]:
        def end_point(prompt: str, **kwargs) -> str:
            if "o1-preview" not in self.chat_model:
                request_data = {
                    **{
                        "model": self.chat_model,
                        # "max_tokens": self.chat_max_new_token,
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are a helpful assistant only capable of communicating with valid JSON, and no other text.",
                            },
                            {"role": "user", "content": f"{prompt}"},
                        ],
                    },
                    **self.chat_parameters,
                }
            else:
                request_data = {
                    "model": self.chat_model,
                    "messages": [
                        {"role": "user", "content": f"{prompt}"},
                    ],
                }
            logger.info("LLM API Request sent")
            with httpx.Client(timeout=self.chat_request_timeout) as client:
                response = client.post(
                    url=self.endpoint, headers=self.headers, json=request_data
                )
            if response.status_code != 200:
                logger.error(
                    f"LLM API Request failed with status code {response.status_code}"
                )
                logger.error(f"LLM API Request failed with response {response.json()}")
                return ""
            logger.info("LLM API Request successful")
            response_json = json.loads(response.text)
            return response_json["choices"][0]["message"]["content"]

        return end_point
