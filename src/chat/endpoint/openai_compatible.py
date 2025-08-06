"""
OpenAI兼容的统一LLM客户端
支持所有OpenAI兼容的API，包括OpenAI、VLLM、硅基流动等
"""

import json
from typing import Any, Dict, List, Union

from openai import OpenAI
from loguru import logger

from ...portfolio import TradeAction

# 为了避免guardrails导入问题，直接从base导入
try:
    from .base import (
        MultiAssetsStructuredGenerationChatEndPoint,
        MultiAssetsStructureGenerationFailure,
        MultiAssetsStructureOutputResponse,
        SingleAssetStructuredGenerationChatEndPoint,
        SingleAssetStructureGenerationFailure,
        SingleAssetStructureOutputResponse,
    )
except ImportError as e:
    # 如果base模块有问题，定义最小的兼容接口
    from abc import ABC, abstractmethod
    from typing import Any, Dict, List, Union
    
    class SingleAssetStructuredGenerationChatEndPoint(ABC):
        @abstractmethod
        def __init__(self, chat_config: Dict[str, Any]) -> None:
            pass
        
        @abstractmethod
        def __call__(self, prompt: str, schema: Any) -> Any:
            pass
    
    class MultiAssetsStructuredGenerationChatEndPoint(ABC):
        @abstractmethod  
        def __init__(self, chat_config: Dict[str, Any]) -> None:
            pass
            
        @abstractmethod
        def __call__(self, prompt: str, schema: Any, symbols: List[str]) -> Any:
            pass

# 导入配置
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
from config import get_model_config


class OpenAICompatibleClient:
    """OpenAI兼容的统一客户端"""
    
    def __init__(self, model_name: str):
        """初始化客户端
        
        Args:
            model_name: 模型名称，必须在config.py中定义
        """
        self.model_config = get_model_config(model_name)
        self.model_name = model_name
        
        # 创建OpenAI客户端
        self.client = OpenAI(
            api_key=self.model_config["api_key"],
            base_url=self.model_config["api_base"]
        )
        
        logger.trace(f"OpenAI兼容客户端初始化: {model_name}")
        logger.trace(f"API Base: {self.model_config['api_base']}")
        logger.trace(f"Provider: {self.model_config.get('provider', 'unknown')}")
        
    def chat_completion(self, 
                       messages: List[Dict[str, str]], 
                       max_tokens: int = 1000,
                       temperature: float = 0.6,
                       response_format: Dict = None,
                       **kwargs) -> str:
        """统一的聊天完成接口
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "..."}]
            max_tokens: 最大生成token数
            temperature: 温度参数
            response_format: 响应格式，用于结构化输出
            **kwargs: 其他参数
            
        Returns:
            str: 模型响应内容
        """
        try:
            # 准备请求参数
            request_params = {
                "model": self.model_config["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            # 如果指定了响应格式（用于结构化输出）
            if response_format:
                request_params["response_format"] = response_format
            
            # 调用API
            response = self.client.chat.completions.create(**request_params)
            
            # 提取响应内容
            content = response.choices[0].message.content
            logger.trace(f"OpenAI兼容API调用成功: {self.model_name}")
            
            return content
            
        except Exception as e:
            logger.error(f"OpenAI兼容API调用失败: {self.model_name}, 错误: {str(e)}")
            raise
    
    def chat_completion_with_json(self, 
                                messages: List[Dict[str, str]], 
                                schema: Dict = None,
                                max_tokens: int = 1000,
                                temperature: float = 0.6,
                                **kwargs) -> Dict:
        """支持JSON响应格式的聊天完成
        
        Args:
            messages: 消息列表
            schema: JSON schema（如果支持的话）
            max_tokens: 最大生成token数
            temperature: 温度参数
            **kwargs: 其他参数
            
        Returns:
            Dict: 解析后的JSON响应
        """
        try:
            # 调试: 检查messages格式
            logger.trace(f"Messages类型: {type(messages)}")
            if messages:
                logger.trace(f"第一个message类型: {type(messages[-1])}")
                logger.trace(f"Message keys: {list(messages[-1].keys()) if isinstance(messages[-1], dict) else 'Not a dict'}")
            
            # 构建基础请求参数
            request_params = {
                "model": self.model_config["model"],
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature
            }
            
            # 根据provider调整参数
            provider = self.model_config.get("provider", "")
            
            # 确保prompt中包含JSON要求，并指定正确的字段名
            if messages and isinstance(messages, list) and len(messages) > 0:
                last_message = messages[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    content = last_message["content"]
                    if isinstance(content, str) and "json" not in content.lower():
                        json_instruction = '\n\nPlease respond with valid JSON format. Use "summary_reason" as the field name for your explanation, and "investment_decision" for your decision (buy/sell/hold).'
                        messages[-1]["content"] = content + json_instruction
            
            # 只对明确支持的模型添加response_format
            if provider == "openai" and any(model in self.model_config["model"] for model in ["gpt-4", "gpt-3.5"]):
                request_params["response_format"] = {"type": "json_object"}
            
            # 添加其他参数，但过滤掉可能不支持的参数
            safe_kwargs = {}
            for key, value in kwargs.items():
                if key not in ["response_format", "functions", "function_call"]:  # 过滤可能不支持的参数
                    safe_kwargs[key] = value
            request_params.update(safe_kwargs)
            
            logger.trace(f"发送API请求参数: {list(request_params.keys())}")
            logger.trace(f"请求参数详情: {request_params}")
            
            # 调用API
            response = self.client.chat.completions.create(**request_params)
            content = response.choices[0].message.content
            
            # 解析JSON
            try:
                result = json.loads(content)
                logger.trace(f"JSON解析成功: {self.model_name}")
                
                # 处理字段名称映射，确保符合Pydantic模型要求
                if "summary" in result and "summary_reason" not in result:
                    result["summary_reason"] = result.pop("summary")
                
                # 处理investment_decision的大小写问题
                if "investment_decision" in result and isinstance(result["investment_decision"], str):
                    result["investment_decision"] = result["investment_decision"].lower()
                
                return result
            except json.JSONDecodeError:
                logger.warning(f"JSON解析失败，尝试修复: {self.model_name}")
                # 尝试简单的JSON修复
                import json_repair
                try:
                    result = json_repair.repair_json(content, return_objects=True)
                    if isinstance(result, dict):
                        logger.trace(f"JSON修复成功: {self.model_name}")
                        return result
                    else:
                        raise ValueError("修复后仍不是有效的JSON对象")
                except:
                    logger.error(f"JSON修复失败: {content[:200]}...")
                    # 返回一个基本的fallback响应
                    return {
                        "investment_decision": "hold",
                        "summary_reason": f"API响应解析失败，采用默认决策。原始响应: {content[:100]}..."
                    }
                    
        except Exception as e:
            logger.error(f"JSON格式聊天完成失败: {self.model_name}, 错误: {str(e)}")
            # 返回一个基本的fallback响应而不是抛出异常
            return {
                "investment_decision": "hold", 
                "summary_reason": f"API调用失败: {str(e)}"
            }


class SingleAssetOpenAICompatibleGeneration(SingleAssetStructuredGenerationChatEndPoint):
    """单资产OpenAI兼容结构化生成"""
    
    def __init__(self, chat_config: Dict[str, Any]) -> None:
        logger.trace("初始化单资产OpenAI兼容生成器")
        
        self.chat_config = chat_config
        self.model_name = chat_config["chat_model"]
        self.max_tokens = chat_config.get("chat_max_new_token", 1000)
        self.timeout = chat_config.get("chat_request_timeout", 60)
        self.parameters = chat_config.get("chat_parameters", {})
        self.system_message = chat_config.get("chat_system_message", "You are a helpful assistant.")
        
        # 初始化OpenAI兼容客户端
        self.client = OpenAICompatibleClient(self.model_name)
        
        logger.trace(f"单资产生成器初始化完成: {self.model_name}")
    
    def __call__(
        self, prompt: str, schema: Any
    ) -> Union[
        SingleAssetStructureGenerationFailure, SingleAssetStructureOutputResponse
    ]:
        try:
            # 处理prompt格式 - 如果是tuple则连接为字符串
            if isinstance(prompt, tuple):
                prompt_str = " ".join(str(p) for p in prompt)
            else:
                prompt_str = str(prompt)
            
            # 清理模板变量
            prompt_str = prompt_str.replace("${investment_info}", "")
            prompt_str = prompt_str.replace("${gr.complete_json_suffix_v2}", "")
            
            # 构建消息
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt_str}
            ]
            
            # 调用API获取JSON响应
            response_dict = self.client.chat_completion_with_json(
                messages=messages,
                schema=schema,
                max_tokens=self.max_tokens,
                **self.parameters
            )
            
            # 处理内存ID去重
            for memory_type in ["short_memory_ids", "mid_memory_ids", "long_memory_ids", "reflection_memory_ids"]:
                if memory_type in response_dict and response_dict[memory_type]:
                    response_dict[memory_type] = list(set(response_dict[memory_type]))
            
            # 创建Pydantic响应对象
            response_pydantic = SingleAssetStructureOutputResponse(**response_dict)
            logger.trace("单资产结构化响应生成成功")
            
            return response_pydantic
            
        except Exception as e:
            logger.error(f"单资产结构化生成失败: {str(e)}")
            return SingleAssetStructureGenerationFailure()


class MultiAssetsOpenAICompatibleGeneration(MultiAssetsStructuredGenerationChatEndPoint):
    """多资产OpenAI兼容结构化生成"""
    
    def __init__(self, chat_config: Dict[str, Any]) -> None:
        logger.trace("初始化多资产OpenAI兼容生成器")
        
        self.chat_config = chat_config
        self.model_name = chat_config["chat_model"]
        self.max_tokens = chat_config.get("chat_max_new_token", 1000)
        self.timeout = chat_config.get("chat_request_timeout", 60)
        self.parameters = chat_config.get("chat_parameters", {})
        self.system_message = chat_config.get("chat_system_message", "You are a helpful assistant.")
        
        # 初始化OpenAI兼容客户端
        self.client = OpenAICompatibleClient(self.model_name)
        
        logger.trace(f"多资产生成器初始化完成: {self.model_name}")
    
    def __call__(
        self, prompt: str, schema: Any, symbols: List[str]
    ) -> Union[
        MultiAssetsStructureGenerationFailure, MultiAssetsStructureOutputResponse
    ]:
        try:
            # 处理prompt格式 - 如果是tuple则连接为字符串
            if isinstance(prompt, tuple):
                prompt_str = " ".join(str(p) for p in prompt)
            else:
                prompt_str = str(prompt)
            
            # 清理模板变量
            prompt_str = prompt_str.replace("${investment_info}", "")
            prompt_str = prompt_str.replace("${gr.complete_json_suffix_v2}", "")
            
            # 构建消息
            messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": prompt_str}
            ]
            
            # 调用API获取JSON响应
            response_dict = self.client.chat_completion_with_json(
                messages=messages,
                schema=schema,
                max_tokens=self.max_tokens,
                **self.parameters
            )
            
            # 解析多资产响应格式
            summary_reason = {
                symbol: response_dict["symbols_summary"][f"{symbol}_summary_reason"]
                for symbol in symbols
                if f"{symbol}_summary_reason" in response_dict.get("symbols_summary", {})
            }
            
            investment_decision = {
                symbol: response_dict["symbols_summary"].get(f"{symbol}_investment_decision", TradeAction.HOLD)
                for symbol in symbols
            }
            
            # 处理各种内存ID
            short_memory_ids = {
                symbol: list(set(response_dict.get(f"{symbol}_short_memory_ids", [])))
                for symbol in symbols
                if f"{symbol}_short_memory_ids" in response_dict
            }
            
            mid_memory_ids = {
                symbol: list(set(response_dict.get(f"{symbol}_mid_memory_ids", [])))
                for symbol in symbols
                if f"{symbol}_mid_memory_ids" in response_dict
            }
            
            long_memory_ids = {
                symbol: list(set(response_dict.get(f"{symbol}_long_memory_ids", [])))
                for symbol in symbols
                if f"{symbol}_long_memory_ids" in response_dict
            }
            
            reflection_memory_ids = {
                symbol: list(set(response_dict.get(f"{symbol}_reflection_memory_ids", [])))
                for symbol in symbols
                if f"{symbol}_reflection_memory_ids" in response_dict
            }
            
            # 创建响应对象
            return MultiAssetsStructureOutputResponse(
                investment_decision=investment_decision,
                summary_reason=summary_reason,
                short_memory_ids=short_memory_ids,
                mid_memory_ids=mid_memory_ids,
                long_memory_ids=long_memory_ids,
                reflection_memory_ids=reflection_memory_ids,
            )
            
        except Exception as e:
            logger.error(f"多资产结构化生成失败: {str(e)}")
            return MultiAssetsStructureGenerationFailure(
                investment_decision={symbol: TradeAction.HOLD for symbol in symbols}
            )