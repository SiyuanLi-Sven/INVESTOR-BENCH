# 环境配置指南

完整的系统环境搭建指南，确保INVESTOR-BENCH系统在你的环境中稳定运行。

## 🖥️ 系统要求

### 硬件要求
- **内存**: 8GB+ RAM (推荐16GB+)
- **存储**: 10GB+ 可用空间
- **网络**: 稳定的互联网连接 (用于API调用)
- **GPU**: 可选 (仅本地LLM部署需要)

### 操作系统支持
- ✅ **macOS** 10.15+ (已测试)
- ✅ **Linux** Ubuntu 18.04+ / CentOS 7+ (推荐)
- ⚠️ **Windows** 10+ (通过WSL2支持)

### 软件依赖
- **Python** 3.8+ (推荐3.11)
- **Docker** 20.10+ (用于Qdrant数据库)
- **PKL** 0.25+ (配置语言)
- **Git** (版本控制)

## 📦 安装步骤

### 1. Python环境配置

#### 选项A: 使用Conda (推荐)
```bash
# 创建新环境
conda create -n investor-bench python=3.11
conda activate investor-bench

# 验证Python版本
python --version  # 应显示 Python 3.11.x
```

#### 选项B: 使用系统Python
```bash
# 检查Python版本
python3 --version

# 创建虚拟环境
python3 -m venv investor-bench-env
source investor-bench-env/bin/activate  # Linux/macOS
# 或
investor-bench-env\\Scripts\\activate    # Windows
```

### 2. 安装项目依赖

```bash
# 克隆项目
git clone <your-repo-url>
cd INVESTOR-BENCH

# 升级pip
pip install --upgrade pip

# 安装项目依赖
pip install -r requirements.txt
```

#### 依赖包说明
```bash
# 核心依赖
guardrails-ai==0.5.13     # LLM输出结构化
qdrant-client==1.12.0     # 向量数据库客户端
httpx==0.27.2             # HTTP客户端 
loguru==0.7.2             # 日志系统
pandas==2.2.3             # 数据处理
numpy<2.0                 # 数值计算 (注意版本限制)

# AI相关
openai==1.60.1            # OpenAI API客户端
pydantic==2.9.2           # 数据验证
tiktoken==0.8.0           # Token计算

# 系统工具
python-dotenv==1.0.1      # 环境变量管理
```

### 3. Docker环境配置

#### 安装Docker
```bash
# macOS (使用Homebrew)
brew install --cask docker

# Linux (Ubuntu/Debian)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER  # 添加当前用户到docker组
newgrp docker  # 刷新组权限

# 验证安装
docker --version
docker run hello-world
```

#### 启动Qdrant向量数据库
```bash
# 拉取镜像
docker pull qdrant/qdrant

# 启动服务 (后台运行)
docker run -d -p 6333:6333 -p 6334:6334 --name qdrant qdrant/qdrant

# 验证服务状态
curl http://localhost:6333/health
# 应返回: {"title":"qdrant - vector search engine","version":"1.x.x"}

# 查看容器状态
docker ps | grep qdrant
```

### 4. PKL配置语言安装

#### macOS
```bash
# 使用Homebrew
brew install pkl

# 验证安装
pkl --version
```

#### Linux
```bash
# 下载并安装
curl -L -o pkl https://github.com/apple/pkl/releases/latest/download/pkl-linux-amd64
chmod +x pkl
sudo mv pkl /usr/local/bin/

# 验证安装
pkl --version
```

#### 手动安装 (所有平台)
```bash
# 从GitHub下载适合你系统的版本
wget https://github.com/apple/pkl/releases/download/0.25.2/pkl-linux-amd64
chmod +x pkl-linux-amd64
sudo mv pkl-linux-amd64 /usr/local/bin/pkl
```

### 5. API密钥配置

#### 创建.env文件
```bash
# 在项目根目录创建.env文件
cat > .env << EOF
# OpenAI兼容API配置
OPENAI_API_KEY="sk-your-api-key-here"
OPENAI_API_BASE="https://api.siliconflow.cn/v1"
OPENAI_MODEL="Qwen/Qwen3-8B"
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B"

# 可选: 其他API配置
# ANTHROPIC_API_KEY="your-claude-key"
# HUGGING_FACE_HUB_TOKEN="your-hf-token"
EOF
```

#### 环境变量说明
| 变量名 | 必需 | 说明 | 示例 |
|--------|------|------|------|
| `OPENAI_API_KEY` | ✅ | OpenAI兼容API密钥 | `sk-xxxxx` |
| `OPENAI_API_BASE` | ⚠️ | API基础URL | `https://api.siliconflow.cn/v1` |
| `OPENAI_MODEL` | ⚠️ | 默认Chat模型 | `Qwen/Qwen3-8B` |
| `EMBEDDING_MODEL` | ⚠️ | 默认Embedding模型 | `Qwen/Qwen3-Embedding-4B` |

### 6. 验证安装

#### 系统环境检查
```bash
# 运行健康检查脚本
python -c "
import sys
print(f'Python: {sys.version}')

import pandas, numpy, httpx, loguru
print('✅ 核心依赖正常')

import guardrails, qdrant_client, openai
print('✅ AI依赖正常')

import os
from dotenv import load_dotenv
load_dotenv()
if os.getenv('OPENAI_API_KEY'):
    print('✅ API密钥已配置')
else:
    print('❌ API密钥未配置')
"
```

#### API连通性测试
```bash
# 测试Chat API
python test_api.py

# 测试Embedding API
python test_embedding.py
```

#### 配置文件生成测试
```bash
# 生成配置文件
pkl eval -f json -o configs/main.json configs/main.pkl

# 验证配置
python -c "
import json
with open('configs/main.json') as f:
    config = json.load(f)
print(f'✅ 配置加载成功: {config[\"meta_config\"][\"run_name\"]}')
"
```

## 🐳 Docker部署 (可选)

如果你希望使用完全容器化的部署：

### Dockerfile创建
```dockerfile
FROM python:3.11-slim

# 安装系统依赖
RUN apt-get update && apt-get install -y \\
    git curl wget jq \\
    && rm -rf /var/lib/apt/lists/*

# 安装PKL
RUN curl -L -o /usr/local/bin/pkl \\
    https://github.com/apple/pkl/releases/latest/download/pkl-linux-amd64 \\
    && chmod +x /usr/local/bin/pkl

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 生成配置
RUN pkl eval -f json -o configs/main.json configs/main.pkl

# 默认命令
CMD ["python", "run.py", "warmup"]
```

### Docker Compose配置
```yaml
version: '3.8'

services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant_data:/qdrant/storage

  investor-bench:
    build: .
    depends_on:
      - qdrant
    env_file:
      - .env
    volumes:
      - ./results:/app/results
      - ./data:/app/data
    command: python run.py warmup

volumes:
  qdrant_data:
```

### 使用Docker Compose
```bash
# 启动所有服务
docker-compose up -d

# 查看日志
docker-compose logs -f investor-bench

# 停止服务
docker-compose down
```

## 🔧 常见问题解决

### 1. Python依赖冲突

**问题**: `pip install` 时出现版本冲突
```bash
# 解决方案1: 使用专用环境
conda create -n investor-bench python=3.11
conda activate investor-bench

# 解决方案2: 强制重装
pip install --force-reinstall -r requirements.txt

# 解决方案3: 使用pip-tools
pip install pip-tools
pip-compile requirements.in
pip-sync requirements.txt
```

### 2. Numpy版本问题

**问题**: `numpy 2.x` 与其他包不兼容
```bash
# 降级到1.x版本
pip install "numpy<2.0"

# 检查兼容性
pip check
```

### 3. Docker权限问题

**问题**: 无法连接Docker daemon
```bash
# Linux: 添加用户到docker组
sudo usermod -aG docker $USER
newgrp docker

# macOS: 确保Docker Desktop运行
open /Applications/Docker.app
```

### 4. Qdrant连接失败

**问题**: `Connection refused to localhost:6333`
```bash
# 检查容器状态
docker ps | grep qdrant

# 重启容器
docker restart qdrant

# 检查端口占用
netstat -tulpn | grep :6333

# 使用其他端口
docker run -d -p 6334:6333 --name qdrant-alt qdrant/qdrant
```

### 5. API密钥问题

**问题**: API调用返回401错误
```bash
# 检查环境变量
echo $OPENAI_API_KEY

# 测试API密钥
curl -H "Authorization: Bearer $OPENAI_API_KEY" \\
     https://api.siliconflow.cn/v1/models

# 重新加载环境变量
source .env  # 如果使用bash
export $(cat .env | xargs)  # 通用方法
```

## 📊 性能优化建议

### 1. 内存优化
```bash
# 设置Python内存限制
export PYTHONHASHSEED=0
export MALLOC_ARENA_MAX=2

# 使用更小的数据类型
# 在配置中减少batch_size和top_k参数
```

### 2. 网络优化
```bash
# 设置HTTP连接池
export HTTP_MAX_CONNECTIONS=20
export HTTP_KEEPALIVE_CONNECTIONS=5

# 配置DNS缓存
echo "nameserver 8.8.8.8" | sudo tee -a /etc/resolv.conf
```

### 3. 存储优化
```bash
# 使用SSD存储Qdrant数据
docker run -d -p 6333:6333 \\
  -v /fast/ssd/path:/qdrant/storage \\
  --name qdrant qdrant/qdrant

# 定期清理日志
find results/ -name "*.log" -mtime +7 -delete
```

现在你的INVESTOR-BENCH环境已经完全配置好了！可以开始运行你的第一次智能投资回测了。