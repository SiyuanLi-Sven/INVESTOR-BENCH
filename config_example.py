# 除非另有说明, 全部的服务都是openai sdk兼容的
# 包括llm服务和embedding服务
# 哪怕是本地部署的vllm, 也要走openai的sdk来进行

# 核心理念是每个模型都可以设置独立的api_base


# 模型配置 - 统一使用OpenAI兼容的API调用
MODEL_CONFIGS = {
    # === LLM 模型配置 ===
    
    # # OpenAI 官方模型
    # "gpt-4": {
    #     "type": "llm_api",
    #     "model": "gpt-4",
    #     "api_base": "https://api.openai.com/v1",
    #     "api_key": "YOUR_OPENAI_API_KEY",
    #     "provider": "openai"
    # },
    # "gpt-4-turbo": {
    #     "type": "llm_api", 
    #     "model": "gpt-4-turbo",
    #     "api_base": "https://api.openai.com/v1",
    #     "api_key": "YOUR_OPENAI_API_KEY",
    #     "provider": "openai"
    # },
    # "gpt-3.5-turbo": {
    #     "type": "llm_api",
    #     "model": "gpt-3.5-turbo", 
    #     "api_base": "https://api.openai.com/v1",
    #     "api_key": "YOUR_OPENAI_API_KEY",
    #     "provider": "openai"
    # },
    
    # # Anthropic 模型 (通过OpenAI兼容接口)
    # "claude-3-opus": {
    #     "type": "llm_api",
    #     "model": "claude-3-opus-20240229",
    #     "api_base": "https://api.anthropic.com/v1", 
    #     "api_key": "YOUR_ANTHROPIC_API_KEY",
    #     "provider": "anthropic"
    # },
    # "claude-3-sonnet": {
    #     "type": "llm_api",
    #     "model": "claude-3-sonnet-20240229",
    #     "api_base": "https://api.anthropic.com/v1",
    #     "api_key": "YOUR_ANTHROPIC_API_KEY", 
    #     "provider": "anthropic"
    # },
    
    # 硅基流动 API
    "Qwen/Qwen3-8B": {
        "type": "llm_api",
        "model": "Qwen/Qwen3-8B",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "you'r",
        "provider": "siliconflow"
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "type": "llm_api",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "api_base": "https://api.siliconflow.cn/v1", 
        "api_key": "your_key",
        "provider": "siliconflow"
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "type": "llm_api",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "your_key", 
        "provider": "siliconflow"
    },
    
    # 本地部署的VLLM (通过OpenAI兼容接口)
    "local-vllm": {
        "type": "llm_api",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",  # 或其他本地模型
        "api_base": "http://0.0.0.0:8000/v1", 
        "api_key": "EMPTY",  # VLLM本地部署通常不需要key
        "provider": "vllm"
    },
    
    # === Embedding 模型配置 ===
    
    # # OpenAI Embedding
    # "text-embedding-3-large": {
    #     "type": "embedding_api",
    #     "model": "text-embedding-3-large",
    #     "api_base": "https://api.openai.com/v1",
    #     "api_key": "YOUR_OPENAI_API_KEY",
    #     "provider": "openai",
    #     "dimensions": 3072
    # },
    # "text-embedding-3-small": {
    #     "type": "embedding_api", 
    #     "model": "text-embedding-3-small",
    #     "api_base": "https://api.openai.com/v1",
    #     "api_key": "YOUR_OPENAI_API_KEY",
    #     "provider": "openai",
    #     "dimensions": 1536
    # },
    # "text-embedding-ada-002": {
    #     "type": "embedding_api",
    #     "model": "text-embedding-ada-002", 
    #     "api_base": "https://api.openai.com/v1",
    #     "api_key": "YOUR_OPENAI_API_KEY",
    #     "provider": "openai",
    #     "dimensions": 1536
    # },
    
    # 硅基流动 Embedding
    "Qwen/Qwen3-Embedding-4B": {
        "type": "embedding_api", 
        "model": "Qwen/Qwen3-Embedding-4B",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "your_key",
        "provider": "siliconflow",
        "dimensions": 4096
    },
    
    # # 本地Embedding (通过OpenAI兼容接口)
    # "local-embedding": {
    #     "type": "embedding_api",
    #     "model": "BAAI/bge-large-zh-v1.5", # 或其他本地embedding模型
    #     "api_base": "http://localhost:8001/v1",  
    #     "api_key": "EMPTY",
    #     "provider": "local",
    #     "dimensions": 1024
    # }
}

# 其他配置
HUGGING_FACE_HUB_TOKEN = "XXXXXX-XXXXXX-XXXXXX-XXXXXX-XXXXXX"

def get_model_config(model_name: str) -> dict:
    """获取指定模型的配置信息"""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_name} not found in configuration")
    return MODEL_CONFIGS[model_name]

def get_all_llm_models() -> list:
    """获取所有可用的LLM模型列表"""
    return [name for name, config in MODEL_CONFIGS.items() if config["type"] == "llm_api"]

def get_all_embedding_models() -> list:
    """获取所有可用的Embedding模型列表"""  
    return [name for name, config in MODEL_CONFIGS.items() if config["type"] == "embedding_api"]
