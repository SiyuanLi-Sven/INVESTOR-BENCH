# ⚙️ 安装与配置

[返回文档索引](./README.md)

## 🎯 环境要求

### 系统要求
- **操作系统**: Linux、macOS、Windows (推荐 Linux/macOS)
- **Python版本**: Python 3.8+
- **内存**: 至少 8GB RAM (推荐 16GB+)
- **存储**: 至少 10GB 可用空间
- **网络**: 稳定的互联网连接(用于API调用)

### 硬件要求
- **CPU**: 多核处理器 (推荐 8核+)
- **GPU**: 可选，如果使用本地VLLM推理
- **网络**: 带宽足够支持API调用

## 📦 依赖安装

### 1. 克隆项目

```bash
git clone <repository_url>
cd INVESTOR-BENCH
```

### 2. Python环境设置

#### 使用conda (推荐)

```bash
# 创建虚拟环境
conda create -n investor-bench python=3.10
conda activate investor-bench

# 安装依赖
pip install -r requirements.txt
```

#### 使用pip

```bash
# 创建虚拟环境
python -m venv investor-bench
source investor-bench/bin/activate  # Linux/macOS
# investor-bench\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 3. 验证安装

```bash
# 基础功能测试
python test_simplified.py

# API接口测试  
python test_openai_compatible.py
```

## 🔑 API密钥配置

### 方法1: config.py配置 (推荐)

在 `config.py` 中配置您的API密钥：

```python
MODEL_CONFIGS = {
    # 硅基流动API (推荐，成本低)
    "Qwen/Qwen3-8B": {
        "type": "llm_api",
        "model": "Qwen/Qwen3-8B",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "sk-your-siliconflow-api-key-here",  # 🔑 替换为实际密钥
        "provider": "siliconflow"
    },
    
    # OpenAI API
    "gpt-4": {
        "type": "llm_api",
        "model": "gpt-4",
        "api_base": "https://api.openai.com/v1",
        "api_key": "sk-your-openai-api-key-here",  # 🔑 替换为实际密钥
        "provider": "openai"
    },
    
    # Embedding模型
    "Qwen/Qwen3-Embedding-4B": {
        "type": "embedding_api",
        "model": "Qwen/Qwen3-Embedding-4B",
        "api_base": "https://api.siliconflow.cn/v1",
        "api_key": "sk-your-siliconflow-api-key-here",  # 🔑 替换为实际密钥
        "provider": "siliconflow"
    }
}
```

### 方法2: 环境变量配置

创建 `.env` 文件：

```bash
# OpenAI配置
OPENAI_API_KEY=sk-your-openai-api-key-here

# HuggingFace配置 (如果需要下载模型)
HUGGING_FACE_HUB_TOKEN=hf-your-huggingface-token-here

# 硅基流动配置
SILICONFLOW_API_KEY=sk-your-siliconflow-api-key-here

# Anthropic配置
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
```

### API密钥获取指南

#### 1. 硅基流动 (推荐)
- 访问: https://cloud.siliconflow.cn/
- 注册账号并创建API密钥
- 成本低，响应快，支持多种开源模型

#### 2. OpenAI
- 访问: https://platform.openai.com/
- 创建账号并生成API Key
- 需要绑定支付方式

#### 3. HuggingFace
- 访问: https://huggingface.co/settings/tokens
- 创建Access Token
- 用于下载开源模型

## 🗄️ 数据库配置

### Qdrant向量数据库

#### Docker部署 (推荐)

```bash
# 拉取镜像
docker pull qdrant/qdrant

# 启动服务
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

#### 本地安装

```bash
# 使用cargo安装
cargo install qdrant

# 或下载二进制文件
wget https://github.com/qdrant/qdrant/releases/latest/download/qdrant
chmod +x qdrant
./qdrant --config-path config/production.yaml
```

#### 配置验证

```bash
# 检查服务状态
curl -X GET "http://localhost:6333/collections"

# 应该返回空的collections列表
# {"result":{"collections":[]},"status":"ok","time":0.000}
```

### 数据库连接配置

在配置文件中设置向量数据库连接：

```json
{
  "agent_config": {
    "memory_db_config": {
      "memory_db_endpoint": "http://localhost:6333"
    }
  }
}
```

## 🤖 模型服务配置

### 本地VLLM部署 (可选)

如果需要使用本地LLM推理：

#### 1. 安装VLLM

```bash
# 使用pip安装
pip install vllm

# 或使用Docker
docker pull vllm/vllm-openai:latest
```

#### 2. 启动VLLM服务

```bash
# 启动脚本
bash scripts/start_vllm.sh

# 或手动启动
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 2
```

#### 3. 配置模型

在 `config.py` 中添加本地模型：

```python
"local-vllm": {
    "type": "llm_api",
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "api_base": "http://0.0.0.0:8000/v1",
    "api_key": "EMPTY",
    "provider": "vllm"
}
```

### 云端API服务

使用云端API服务更简单，只需配置API密钥即可。

## 🔧 系统配置验证

### 1. 运行全面测试

```bash
# 基础配置测试
python test_simplified.py

# 预期输出
# ==================================================
# 统一配置系统 - 简化测试
# ==================================================
# 
# [基础配置功能]
# ✓ 配置文件导入成功
# ✓ 找到 4 个LLM模型
# ✓ 找到 1 个Embedding模型
# ✓ 成功获取模型配置
# ------------------------------
# 
# 测试结果: 4/5 通过
# 🎉 核心功能测试通过！
```

### 2. API连接测试

```bash
# OpenAI兼容接口测试
python test_openai_compatible.py

# 预期输出应显示成功的配置和连接
```

### 3. 服务状态检查

```bash
# 检查Qdrant
curl -X GET "http://localhost:6333/collections"

# 检查VLLM (如果使用)
curl -X GET "http://localhost:8000/health"

# 检查GPU状态 (如果使用)
nvidia-smi
```

## 📁 目录结构

安装完成后的项目结构：

```
INVESTOR-BENCH/
├── config.py                    # 🔥 统一配置文件
├── configs/                     # 配置文件目录
│   ├── main.json               # 原始配置
│   └── main_unified.json       # 统一API配置
├── src/                         # 源代码
│   ├── chat/                   # LLM接口
│   ├── embedding_unified.py    # 统一Embedding接口
│   └── ...
├── data/                        # 市场数据
├── results/                     # 实验结果
├── docs/                        # 项目文档
├── requirements.txt             # Python依赖
├── run.py                      # 主程序入口
└── test_*.py                   # 测试脚本
```

## 🚀 快速验证

### 创建测试配置

```bash
# 使用提供的统一配置模板
cp configs/main_unified.json configs/my_test.json

# 编辑配置文件，更新API密钥
vim configs/my_test.json
```

### 运行简单测试

```bash
# 验证配置
python -c "
import json
config = json.load(open('configs/my_test.json'))
print('✓ Configuration loaded successfully')
print(f'LLM Model: {config[\"chat_config\"][\"chat_model\"]}')
print(f'Embedding Model: {config[\"emb_config\"][\"emb_model_name\"]}')
"

# 测试依赖
python -c "
import openai, qdrant_client, loguru
from src import FinMemAgent, MarketEnv
print('✓ All dependencies imported successfully')
"
```

## 🔐 安全注意事项

### API密钥安全
1. **不要提交密钥**: 将 `.env` 和包含密钥的配置文件添加到 `.gitignore`
2. **定期轮换**: 定期更新API密钥
3. **权限控制**: 使用最小权限原则配置API密钥

### 网络安全
1. **防火墙配置**: 确保Qdrant端口(6333)不对外开放
2. **HTTPS使用**: 生产环境使用HTTPS连接
3. **访问控制**: 限制API访问来源

## 🐛 常见问题

### 1. 依赖安装失败
```bash
# 清理pip缓存
pip cache purge

# 使用国内镜像源
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

### 2. API连接失败
```bash
# 检查网络连接
curl -I https://api.openai.com/v1/models

# 验证API密钥格式
python -c "
import openai
client = openai.OpenAI(api_key='your-api-key')
print('✓ API key format valid')
"
```

### 3. Qdrant连接问题
```bash
# 检查端口占用
netstat -tulpn | grep 6333

# 重启Qdrant服务
docker restart <qdrant_container_id>
```

## 📚 下一步

安装完成后，请继续阅读：

- [快速开始](./03-quick-start.md) - 运行第一个实验
- [配置系统](./04-configuration.md) - 详细配置选项
- [CLI参考](./10-cli-reference.md) - 命令行使用指南

---

[← 上一章: 项目概述](./01-project-overview.md) | [下一章: 快速开始 →](./03-quick-start.md)