# 快速开始指南

5分钟快速上手INVESTOR-BENCH系统，开始你的第一次智能投资回测。

## 🚀 前置条件

### 系统要求
- Python 3.8+
- Docker (用于Qdrant向量数据库)
- 8GB+ RAM
- macOS/Linux (推荐)

### API准备
- OpenAI兼容API密钥 (如SiliconFlow、OpenAI等)
- 建议同时支持Chat和Embedding API

## ⚡ 5分钟快速部署

### 1. 环境准备 (2分钟)

```bash
# 克隆项目
git clone <your-repo-url>
cd INVESTOR-BENCH

# 安装依赖
pip install -r requirements.txt

# 启动向量数据库
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant
```

### 2. 配置API密钥 (1分钟)

创建 `.env` 文件:
```bash
# OpenAI兼容API配置
OPENAI_API_KEY="sk-your-api-key-here"
OPENAI_API_BASE="https://api.siliconflow.cn/v1"  # 或其他兼容API
OPENAI_MODEL="Qwen/Qwen3-8B"                      # 或其他模型
EMBEDDING_MODEL="Qwen/Qwen3-Embedding-4B"        # 或其他embedding模型
```

### 3. 生成配置文件 (30秒)

```bash
# 生成JSON配置
pkl eval -f json -o configs/main.json configs/main.pkl
```

### 4. 运行第一次回测 (1.5分钟)

```bash
# 短期测试 (推荐首次运行)
python run.py warmup

# 查看运行状态
tail -f results/exp/*/JNJ/log/*.log
```

## 📊 验证结果

### 查看输出文件
```bash
# 查看结果目录
ls -la results/exp/qwen3-8b-siliconflow/JNJ/

# 主要输出文件:
# - warmup_output/    # Warmup阶段输出
# - test_output/      # 测试阶段输出  
# - final_result/     # 最终结果和分析
# - log/             # 详细日志
```

### 检查系统状态
```bash
# 检查API连通性
python test_api.py

# 检查Embedding功能  
python test_embedding.py

# 查看配置
cat configs/main.json | jq .
```

## 🎯 下一步

### 完整回测流程
```bash
# 1. Warmup阶段 - 学习阶段
python run.py warmup

# 2. Test阶段 - 实际回测
python run.py test  

# 3. 结果评估
python run.py eval
```

### 自定义配置

#### 修改交易资产
编辑 `configs/main.pkl`:
```pkl
trading_symbols = new Listing {
    "AAPL"  // 改为Apple
    "GOOGL" // 添加Google
}
```

#### 修改时间范围
```pkl
warmup_start_time = "2020-01-01"
warmup_end_time = "2020-06-30"
test_start_time = "2020-07-01"  
test_end_time = "2020-12-31"
```

#### 切换LLM模型
```pkl
chat_model = "gpt-4o"  // 切换到GPT-4
// 或
chat_model = "claude-sonnet-35"  // 切换到Claude
```

## 🔧 常见问题

### Q: API调用失败
```bash
# 检查API配置
echo $OPENAI_API_KEY
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.siliconflow.cn/v1/models
```

### Q: Qdrant连接失败
```bash
# 检查Qdrant状态
docker ps | grep qdrant

# 重启Qdrant
docker restart qdrant

# 检查端口
netstat -an | grep 6333
```

### Q: 内存不足
```bash
# 减少批处理大小
# 在configs/main.pkl中修改:
top_k = 3  # 从5减少到3
look_back_window_size = 2  # 从3减少到2
```

### Q: 数据文件缺失
```bash
# 检查数据文件
ls -la data/
# 应该包含: jnj.json, uvv.json 等

# 如果缺失，检查配置中的trading_symbols
# 确保每个symbol都有对应的数据文件
```

## 📈 性能优化建议

### 1. API调用优化
- 使用批处理减少API调用次数
- 配置合适的请求间隔避免限流
- 使用缓存减少重复计算

### 2. 内存优化  
- 定期清理过期记忆
- 使用流式处理大数据集
- 合理设置记忆重要性阈值

### 3. 并行处理
- 多资产可并行处理
- 使用异步API调用
- 合理配置进程数

## 📚 进阶阅读

- [系统架构](./architecture.md) - 深入理解系统设计
- [配置管理](./configuration.md) - 高级配置选项
- [API集成](./api-integration.md) - 更多API集成方式
- [故障排除](./troubleshooting.md) - 详细问题解决方法

## 🎉 成功标志

如果看到以下输出，说明系统运行正常:

```
✅ API连接成功!
✅ Embedding API成功!
✅ 记忆系统正常
🔄 开始交易决策...
📊 Warmup进度: 100%
🎯 回测完成，开始生成报告...
```

恭喜！你已经成功运行了第一次智能投资回测。现在可以开始探索更多高级功能了。