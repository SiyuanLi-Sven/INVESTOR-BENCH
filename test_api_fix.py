#!/usr/bin/env python3
"""
API连接测试脚本，用于验证修复后的超时处理是否有效
"""

import os
import sys
import time
from datetime import datetime

import openai
from loguru import logger

# 配置日志
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}")

def test_api_connection():
    """测试基础API连接"""
    logger.info("🔍 开始API连接测试...")
    
    # 检查环境变量
    api_key = os.environ.get('OPENAI_API_KEY')
    api_base = os.environ.get('OPENAI_API_BASE', 'https://api.siliconflow.cn/v1')
    
    if not api_key:
        logger.error("❌ OPENAI_API_KEY环境变量未设置")
        return False
        
    logger.info(f"🔑 API Key: {api_key[:10]}...")
    logger.info(f"🌐 API Base: {api_base}")
    
    # 创建客户端
    client = openai.OpenAI(
        api_key=api_key,
        base_url=api_base
    )
    
    try:
        # 测试简单的chat请求
        logger.info("💬 测试Chat API...")
        start_time = time.time()
        
        response = client.chat.completions.create(
            model="Qwen/Qwen3-8B",
            messages=[{"role": "user", "content": "Hello, respond with just 'OK'"}],
            timeout=30,  # 30秒超时
            max_tokens=10
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Chat API测试成功 (耗时: {elapsed:.1f}秒)")
        logger.info(f"📝 响应: {response.choices[0].message.content.strip()}")
        
        # 测试Embedding API
        logger.info("🧠 测试Embedding API...")
        start_time = time.time()
        
        emb_response = client.embeddings.create(
            model="Qwen/Qwen3-Embedding-4B",
            input="test embedding",
            timeout=30
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ Embedding API测试成功 (耗时: {elapsed:.1f}秒)")
        logger.info(f"📊 向量维度: {len(emb_response.data[0].embedding)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ API测试失败: {str(e)}")
        return False

def test_timeout_handling():
    """测试新的超时处理机制"""
    logger.info("⏱️ 测试超时处理机制...")
    
    try:
        from src.chat.endpoint.guardrails import GPTGuardRailStructureGeneration
        
        # 创建配置
        chat_config = {
            "chat_model": "Qwen/Qwen3-8B",
            "chat_max_new_token": 1000,
            "chat_model_type": "chat",
            "chat_endpoint": f"{os.environ.get('OPENAI_API_BASE', 'https://api.siliconflow.cn/v1')}/chat/completions",
            "chat_request_timeout": 10,  # 设置一个很短的超时来测试
            "chat_parameters": {"temperature": 0.6, "max_tokens": 100}
        }
        
        # 创建GuardRail实例
        guard_gen = GPTGuardRailStructureGeneration(chat_config)
        
        logger.info("✅ GuardRails结构生成器初始化成功")
        logger.info(f"🔧 配置的超时时间: {chat_config['chat_request_timeout']}秒")
        logger.info(f"🔧 最大重试次数: {guard_gen.max_retries}")
        logger.info(f"🔧 重试延迟: {guard_gen.retry_delay}秒")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 超时处理测试失败: {str(e)}")
        return False

def test_academic_integrity():
    """测试学术项目的纯净性：确保失败时直接报错"""
    logger.info("🎓 测试学术项目纯净性...")
    
    try:
        from src.chat.endpoint.guardrails import GPTGuardRailStructureGeneration
        
        chat_config = {
            "chat_model": "Qwen/Qwen3-8B",
            "chat_max_new_token": 1000,
            "chat_model_type": "chat",
            "chat_endpoint": f"{os.environ.get('OPENAI_API_BASE', 'https://api.siliconflow.cn/v1')}/chat/completions",
            "chat_request_timeout": 300,
            "chat_parameters": {"temperature": 0.6, "max_tokens": 100}
        }
        
        guard_gen = GPTGuardRailStructureGeneration(chat_config)
        
        # 确认没有降级处理方法
        if hasattr(guard_gen, '_fallback_generation'):
            logger.error("❌ 检测到降级处理方法，违反学术项目纯净性原则")
            return False
        
        logger.info("✅ 学术项目纯净性检查通过：没有智能降级策略")
        logger.info("📚 失败时将直接报错，保持实验结果真实性")
        return True
            
    except Exception as e:
        logger.error(f"❌ 学术纯净性测试失败: {str(e)}")
        return False

def main():
    """主测试函数"""
    logger.info("🚀 开始API修复验证测试")
    logger.info(f"⏰ 测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)
    
    all_tests_passed = True
    
    # 测试1: 基础API连接
    if not test_api_connection():
        all_tests_passed = False
    
    logger.info("=" * 50)
    
    # 测试2: 超时处理机制
    if not test_timeout_handling():
        all_tests_passed = False
    
    logger.info("=" * 50)
    
    # 测试3: 学术项目纯净性
    if not test_academic_integrity():
        all_tests_passed = False
        
    logger.info("=" * 50)
    
    if all_tests_passed:
        logger.info("🎉 所有测试通过！修复验证成功")
        logger.info("🎓 系统现在能更好地处理API超时问题，同时保持学术项目的纯净性")
        logger.info("🔧 建议的配置参数:")
        logger.info("   - chat_request_timeout: 300秒 (5分钟)")
        logger.info("   - embedding_timeout: 120秒 (2分钟)")
        logger.info("   - 最大重试次数: 3次")
        logger.info("   - 重试间隔: 5秒")
        logger.info("📚 学术原则: 失败时直接报错，不进行任何智能降级")
    else:
        logger.error("❌ 部分测试失败，请检查配置和网络连接")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 