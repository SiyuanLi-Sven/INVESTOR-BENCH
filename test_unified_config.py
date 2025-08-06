#!/usr/bin/env python3
"""
测试统一配置系统是否正常工作
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

def test_config_import():
    """测试配置文件导入"""
    try:
        from config import MODEL_CONFIGS, get_model_config, get_all_llm_models, get_all_embedding_models
        print("✓ 配置文件导入成功")
        return True
    except Exception as e:
        print(f"✗ 配置文件导入失败: {e}")
        return False

def test_model_configs():
    """测试模型配置"""
    try:
        from config import MODEL_CONFIGS, get_all_llm_models, get_all_embedding_models
        
        # 测试获取LLM模型列表
        llm_models = get_all_llm_models()
        print(f"✓ 找到 {len(llm_models)} 个LLM模型: {llm_models}")
        
        # 测试获取Embedding模型列表  
        emb_models = get_all_embedding_models()
        print(f"✓ 找到 {len(emb_models)} 个Embedding模型: {emb_models}")
        
        # 测试获取特定模型配置
        if llm_models:
            from config import get_model_config
            test_model = llm_models[0]
            config = get_model_config(test_model)
            print(f"✓ 成功获取模型配置: {test_model}")
            print(f"  - API Base: {config['api_base']}")
            print(f"  - Provider: {config.get('provider', 'unknown')}")
        
        return True
    except Exception as e:
        print(f"✗ 模型配置测试失败: {e}")
        return False

def test_embedding_unified():
    """测试统一embedding模块"""
    try:
        from src.embedding_unified import UnifiedOpenAIEmbedding, EmbeddingModel
        print("✓ 统一Embedding模块导入成功")
        return True
    except Exception as e:
        print(f"✗ 统一Embedding模块导入失败: {e}")
        return False

def test_openai_compatible():
    """测试OpenAI兼容模块"""
    try:
        from src.chat.endpoint.openai_compatible import OpenAICompatibleClient
        print("✓ OpenAI兼容模块导入成功") 
        return True
    except Exception as e:
        print(f"✗ OpenAI兼容模块导入失败: {e}")
        return False

def test_chat_module():
    """测试chat模块更新"""
    try:
        from src.chat import get_chat_model
        from src.utils import TaskType
        print("✓ 更新后的chat模块导入成功")
        return True
    except Exception as e:
        print(f"✗ chat模块导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("=" * 50)
    print("统一配置系统测试")
    print("=" * 50)
    
    tests = [
        ("配置文件导入", test_config_import),
        ("模型配置", test_model_configs), 
        ("统一Embedding模块", test_embedding_unified),
        ("OpenAI兼容模块", test_openai_compatible),
        ("Chat模块更新", test_chat_module),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        if test_func():
            passed += 1
        print("-" * 30)
    
    print(f"\n测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试通过！统一配置系统已经就绪。")
        print("\n使用说明:")
        print("1. 在config.py中配置您的模型API密钥")
        print("2. 在配置文件中设置 'chat_model_inference_engine': 'openai_compatible'")
        print("3. 使用 'chat_model': '<model_name>' 指定要使用的模型")
        print("4. 模型名称必须在config.py的MODEL_CONFIGS中定义")
        return True
    else:
        print("❌ 部分测试失败，请检查配置。")
        return False

if __name__ == "__main__":
    main()