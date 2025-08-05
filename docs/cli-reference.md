# CLI命令参考

INVESTOR-BENCH提供完整的命令行接口，支持所有核心功能的自动化执行。

## 📋 命令概览

```bash
# 主要命令
python run.py <command> [options]

# 可用命令:
warmup              # Warmup学习阶段
warmup-checkpoint   # 从检查点恢复warmup  
test               # 实际回测阶段
test-checkpoint    # 从检查点恢复test
eval               # 结果评估和分析

# 配置和工具
pkl eval -f json -o configs/main.json configs/main.pkl  # 生成配置
python test_api.py        # 测试API连通性
python test_embedding.py  # 测试embedding功能
```

## 🎯 核心命令详解

### 1. warmup - 学习阶段

**用途**: 让AI从专业交易员的建议中学习，建立记忆库

**基本用法**:
```bash
python run.py warmup
```

**详细流程**:
1. 加载历史市场数据和新闻
2. 对每天的信息进行embedding向量化
3. 存储到四层记忆系统中
4. 学习专业交易员的决策模式
5. 保存学习进度检查点

**输出文件**:
```
results/exp/{model_name}/{symbol}/
├── warmup_checkpoint/     # 检查点文件
├── warmup_output/        # 详细输出日志
└── log/                  # 系统运行日志
```

**典型输出示例**:
```
2025-08-05T20:19:21.308441+0800 INFO ENV- current date: 2020-07-02
2025-08-05T20:19:21.308552+0800 INFO ENV-price: {'JNJ': 126.29952239990234}
2025-08-05T20:19:23.120794+0800 INFO AGENT-Queried memories for symbol: JNJ
2025-08-05T20:19:23.120925+0800 INFO AGENT-Short Memory: 2 items found
Warmup remaining: 6 steps                    100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 2. warmup-checkpoint - 恢复学习

**用途**: 从之前保存的检查点继续warmup阶段

**基本用法**:
```bash
python run.py warmup-checkpoint
```

**使用场景**:
- 系统意外中断后恢复
- 调整参数后继续训练
- 分段执行长时间warmup

**检查点内容**:
```python
checkpoint = {
    'agent_state': agent.get_state(),
    'portfolio_state': portfolio.get_state(), 
    'memory_db_state': memory_db.get_state(),
    'current_date': current_date,
    'completed_steps': step_count
}
```

### 3. test - 实际回测

**用途**: 使用训练好的记忆系统进行独立投资决策

**基本用法**:
```bash
python run.py test
```

**前置条件**: 必须先完成warmup阶段

**决策流程**:
1. 从记忆库检索相关历史信息
2. 分析当前市场状况
3. 做出buy/sell/hold决策
4. 执行交易并记录结果
5. 更新投资组合状态

**输出示例**:
```json
{
  "date": "2020-10-01",
  "decision": "buy", 
  "reason": "基于positive news和technical analysis",
  "confidence": 0.85,
  "portfolio_value": 100500.0,
  "holdings": {"JNJ": 100}
}
```

### 4. test-checkpoint - 恢复回测

**用途**: 从检查点继续test阶段回测

```bash
python run.py test-checkpoint
```

### 5. eval - 结果评估

**用途**: 分析回测结果，生成性能报告

**基本用法**:
```bash
python run.py eval
```

**生成报告**:
- 详细的CSV交易记录
- Markdown格式分析报告
- 关键性能指标计算
- 可视化图表(如果配置)

## ⚙️ 配置管理命令

### PKL配置系统

**生成主配置**:
```bash
pkl eval -f json -o configs/main.json configs/main.pkl
```

**验证配置语法**:
```bash
pkl eval configs/main.pkl
```

**查看配置结构**:
```bash
pkl eval -f yaml configs/main.pkl
```

**生成特定配置**:
```bash
# 只生成chat配置
pkl eval -p chat_config configs/main.pkl

# 只生成memory配置  
pkl eval -p agent_config.memory_db_config configs/main.pkl
```

## 🧪 测试和诊断命令

### API连通性测试

**Chat API测试**:
```bash
python test_api.py
```

输出示例:
```
🔧 测试配置:
   API Base: https://api.siliconflow.cn/v1
   Model: Qwen/Qwen3-8B
🚀 发送API请求...
✅ API连接成功!
📝 响应内容: {"analysis": "市场分析...", "recommendation": "投资建议"}
```

**Embedding API测试**:
```bash
python test_embedding.py
```

输出示例:
```
🧠 开始Embedding API测试...
✅ Embedding API成功!
📝 向量维度: 2560
📝 前5个值: [-0.0005367548, 0.031304926, ...]
```

### 系统状态检查

**检查Docker服务**:
```bash
docker ps | grep qdrant
```

**检查配置文件**:
```bash
cat configs/main.json | jq '.chat_config'
cat configs/main.json | jq '.emb_config'
```

**检查结果目录**:
```bash
ls -la results/exp/*/
tree results/ -L 3
```

**查看日志**:
```bash
# 实时查看日志
tail -f results/exp/*/JNJ/log/*.log

# 搜索错误日志
grep -r "ERROR" results/exp/*/JNJ/log/

# 查看特定日期日志
grep "2020-07-02" results/exp/*/JNJ/log/*.log
```

## 🔧 高级用法

### 并行执行多个资产

**配置多资产**:
```pkl  
// configs/main.pkl
trading_symbols = new Listing {
    "AAPL"
    "GOOGL" 
    "MSFT"
}
```

**并行运行**:
```bash
# 方法1: 使用系统内置并行支持
python run.py warmup  # 自动并行处理多资产

# 方法2: 手动并行 (高级用户)
python run.py warmup --symbol AAPL &
python run.py warmup --symbol GOOGL &
python run.py warmup --symbol MSFT &
wait
```

### 自定义时间范围

**修改配置**:
```pkl
warmup_start_time = "2019-01-01"
warmup_end_time = "2019-12-31"  
test_start_time = "2020-01-01"
test_end_time = "2020-12-31"
```

**分段执行**:
```bash
# 第一阶段
pkl eval -e 'configs/main.pkl.amend { warmup_end_time = "2019-06-30" }' \
  -f json -o configs/phase1.json

python run.py warmup --config configs/phase1.json

# 第二阶段
pkl eval -e 'configs/main.pkl.amend { warmup_start_time = "2019-07-01" }' \
  -f json -o configs/phase2.json
  
python run.py warmup-checkpoint --config configs/phase2.json
```

### 批量实验执行

**实验脚本示例**:
```bash
#!/bin/bash
# batch_experiment.sh

models=("gpt-4o" "claude-sonnet-35" "qwen3-8b-siliconflow")
symbols=("AAPL" "GOOGL" "MSFT")

for model in "${models[@]}"; do
    for symbol in "${symbols[@]}"; do
        echo "Running experiment: $model on $symbol"
        
        # 生成配置
        pkl eval -e "configs/main.pkl.amend { 
            chat_model = \"$model\"
            trading_symbols = new Listing { \"$symbol\" }
            run_name = \"exp_${model}_${symbol}\"
        }" -f json -o "configs/exp_${model}_${symbol}.json"
        
        # 执行实验
        python run.py warmup --config "configs/exp_${model}_${symbol}.json"
        python run.py test --config "configs/exp_${model}_${symbol}.json"
        python run.py eval --config "configs/exp_${model}_${symbol}.json"
        
        echo "Completed: $model on $symbol"
    done
done
```

### 性能分析和优化

**内存使用监控**:
```bash
# 运行时监控
python -m memory_profiler run.py warmup

# 结果分析
mprof run python run.py warmup
mprof plot
```

**性能基准测试**:
```bash
# 时间测量
time python run.py warmup

# 详细性能分析
python -m cProfile -o profile_output.prof run.py warmup
python -m pstats profile_output.prof
```

**API调用统计**:
```bash
# 启用API调用记录
export LOG_API_CALLS=true
python run.py warmup

# 分析API使用
grep "LLM API Request" results/exp/*/JNJ/log/*.log | wc -l
grep "EMB API Request" results/exp/*/JNJ/log/*.log | wc -l
```

## 🚨 故障排除命令

### 常见问题诊断

**配置问题**:
```bash  
# 验证PKL语法
pkl eval configs/main.pkl > /dev/null && echo "配置语法正确" || echo "配置语法错误"

# 检查必需字段
pkl eval -p trading_symbols configs/main.pkl
pkl eval -p chat_model configs/main.pkl  
pkl eval -p embedding_model configs/main.pkl
```

**依赖问题**:
```bash
# 检查Python包
pip list | grep -E "(guardrails|qdrant|httpx|loguru)"

# 检查版本兼容性
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import guardrails; print(f'guardrails: {guardrails.__version__}')"
```

**网络连接问题**:
```bash
# 测试网络连接
curl -I https://api.siliconflow.cn/v1/models

# 测试Qdrant连接
curl http://localhost:6333/health

# 检查防火墙
telnet localhost 6333
```

**清理和重置**:
```bash
# 清理结果目录
rm -rf results/exp/*/

# 重置Qdrant数据库
docker restart qdrant

# 清理缓存
rm -rf __pycache__/ .pytest_cache/
find . -name "*.pyc" -delete
```

这个CLI参考指南可以帮助你熟练使用INVESTOR-BENCH的所有功能，无论是日常使用还是高级定制都能找到对应的命令。