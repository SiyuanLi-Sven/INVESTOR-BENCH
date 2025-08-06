"""
统一的Embedding模块，使用OpenAI兼容接口
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

from openai import OpenAI
from loguru import logger
from pydantic import BaseModel, ValidationError

# 导入配置
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from config import get_model_config


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str
    usage: Dict[str, int]


class UnifiedEmbeddingError(Exception):
    """统一的Embedding错误类"""
    def __init__(self, message: str, provider: str = "unknown") -> None:
        self.message = f"Embedding API调用失败 ({provider}): {message}"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class EmbeddingModel(ABC):
    """Embedding模型抽象基类"""
    
    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def __call__(self, texts: Union[List[str], str]) -> List[List[float]]:
        pass


class UnifiedOpenAIEmbedding(EmbeddingModel):
    """统一的OpenAI兼容Embedding客户端"""
    
    def __init__(self, emb_config: Dict) -> None:
        self.config = emb_config
        self.model_name = emb_config["emb_model_name"]
        
        # 从config.py获取模型配置
        try:
            self.model_config = get_model_config(self.model_name)
        except ValueError as e:
            logger.error(f"模型配置未找到: {self.model_name}")
            raise ValueError(f"模型配置未找到: {self.model_name}") from e
        
        # 验证是否为embedding类型
        if self.model_config["type"] != "embedding_api":
            raise ValueError(f"模型 {self.model_name} 不是embedding类型")
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            api_key=self.model_config["api_key"],
            base_url=self.model_config["api_base"]
        )
        
        self.provider = self.model_config.get("provider", "unknown")
        self.timeout = emb_config.get("embedding_timeout", 60)
        
        logger.trace(f"统一Embedding客户端初始化: {self.model_name}")
        logger.trace(f"Provider: {self.provider}")
        logger.trace(f"API Base: {self.model_config['api_base']}")

    def __call__(self, texts: Union[List[str], str]) -> List[List[float]]:
        """获取文本的embedding向量
        
        Args:
            texts: 单个文本或文本列表
            
        Returns:
            List[List[float]]: embedding向量列表
        """
        if isinstance(texts, str):
            texts = [texts]

        try:
            logger.trace(f"调用Embedding API: {self.model_name}, 文本数量: {len(texts)}")
            
            # 调用OpenAI兼容的embedding接口
            response = self.client.embeddings.create(
                input=texts,
                model=self.model_config["model"],
                encoding_format="float"
            )
            
            # 确保按索引排序
            embeddings_data = sorted(response.data, key=lambda x: x.index)
            embeddings = [item.embedding for item in embeddings_data]
            
            logger.trace(f"Embedding API调用成功: {self.model_name}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding API调用失败: {self.model_name}, 错误: {str(e)}")
            raise UnifiedEmbeddingError(
                message=str(e),
                provider=self.provider
            ) from e


# 为了保持向后兼容性，保留原来的类名
class OpenAIEmbedding(UnifiedOpenAIEmbedding):
    """向后兼容的类名别名"""
    pass