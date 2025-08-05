import json
import os
from typing import Any, Dict, Tuple, Union

import httpx
from loguru import logger

from .base import (
    SingleAssetStructuredGenerationChatEndPoint,
    SingleAssetStructureGenerationFailure,
    SingleAssetStructureOutputResponse,
    delete_placeholder_info,
)


class OpenAIEndpoint(SingleAssetStructuredGenerationChatEndPoint):
    """
    OpenAI端点实现，直接使用OpenAI API进行结构化输出生成。
    不依赖guardrails库，使用OpenAI的JSON模式功能。
    """
    
    def __init__(self, chat_config: Dict[str, Any]) -> None:
        """初始化OpenAI端点"""
        self.chat_config = chat_config
        self.chat_model = chat_config["chat_model"]
        self.chat_max_new_token = chat_config["chat_max_new_token"]
        self.chat_request_timeout = chat_config["chat_request_timeout"]
        self.chat_parameters = chat_config["chat_parameters"]
        
        # 设置API端点
        self.endpoint = "https://api.openai.com/v1/chat/completions"
        if "chat_endpoint" in chat_config and chat_config["chat_endpoint"]:
            self.endpoint = chat_config["chat_endpoint"]
            
        # 设置请求头
        try:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        except KeyError as e:
            logger.error("无法找到OpenAI API密钥")
            raise ValueError("无法找到OpenAI API密钥，请在.env文件中设置OPENAI_API_KEY") from e
            
        self.headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }
        
        logger.info(f"CHAT-初始化OpenAI端点，使用模型: {self.chat_model}")
    
    def __call__(
        self, prompt: Tuple[str, str], schema: Any
    ) -> Union[SingleAssetStructureGenerationFailure, SingleAssetStructureOutputResponse]:
        """调用OpenAI API生成结构化输出"""
        invest_info_prompt, ask_prompt = prompt
        
        # 构建完整提示
        full_prompt = f"{invest_info_prompt}\n\n{ask_prompt}"
        
        # 构建请求数据
        request_data = {
            "model": self.chat_model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个金融投资助手，专注于分析市场数据并做出投资决策。请仅返回JSON格式的响应。"
                },
                {"role": "user", "content": full_prompt}
            ],
            "response_format": {"type": "json_object"},
            **self.chat_parameters
        }
        
        # 如果模型不支持json_object格式，移除该字段
        if any(model in self.chat_model for model in ["gpt-3.5-turbo-0301", "gpt-4-0314"]):
            del request_data["response_format"]
            request_data["messages"][0]["content"] += " 你必须只返回有效的JSON格式，不要包含任何其他文本。"
        
        # 发送请求
        logger.info("LLM API请求已发送")
        try:
            with httpx.Client(timeout=self.chat_request_timeout) as client:
                response = client.post(
                    url=self.endpoint,
                    headers=self.headers,
                    json=request_data
                )
                
            # 检查响应状态
            if response.status_code != 200:
                logger.error(f"LLM API请求失败，状态码: {response.status_code}")
                logger.error(f"错误响应: {response.text}")
                return SingleAssetStructureGenerationFailure()
                
            # 解析响应
            response_json = response.json()
            content = response_json["choices"][0]["message"]["content"]
            
            # 尝试解析JSON
            try:
                validated_output = json.loads(content)
                validated_output_dicts = delete_placeholder_info(validated_output)
                
                # 构建输出响应
                output_dict = {}
                
                # 处理投资决策
                if "investment_decision" not in validated_output_dicts:
                    output_dict = {
                        "summary_reason": validated_output_dicts.get("summary_reason", "未提供决策理由")
                    }
                else:
                    output_dict = {
                        "investment_decision": validated_output_dicts["investment_decision"],
                        "summary_reason": validated_output_dicts.get("summary_reason", "未提供决策理由"),
                    }
                
                # 处理记忆ID
                for memory_type in ["short_memory_ids", "mid_memory_ids", "long_memory_ids", "reflection_memory_ids"]:
                    if memory_type in validated_output_dicts:
                        if isinstance(validated_output_dicts[memory_type], list):
                            if all(isinstance(item, dict) and "memory_index" in item for item in validated_output_dicts[memory_type]):
                                output_dict[memory_type] = [
                                    item["memory_index"] for item in validated_output_dicts[memory_type]
                                ]
                            elif all(isinstance(item, (int, str)) for item in validated_output_dicts[memory_type]):
                                output_dict[memory_type] = validated_output_dicts[memory_type]
                
                return SingleAssetStructureOutputResponse(**output_dict)
                
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"解析LLM响应失败: {e}")
                logger.error(f"原始响应内容: {content}")
                return SingleAssetStructureGenerationFailure()
                
        except Exception as e:
            logger.error(f"调用OpenAI API时发生错误: {e}")
            return SingleAssetStructureGenerationFailure() 