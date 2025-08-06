# INVESTOR-BENCH 交易报告

## 基本信息

- **运行名称**: 250806_135830_Qwen_Qwen3-8B_JNJ
- **时间戳**: 250806_135830
- **模型**: Qwen_Qwen3-8B
- **交易标的**: JNJ
- **结果路径**: results/250806_135830_Qwen_Qwen3-8B_JNJ

## 执行概览

### 文件结构

```
results/250806_135830_Qwen_Qwen3-8B_JNJ/
├── warmup_checkpoint/    # 预热检查点
├── warmup_output/        # 预热输出
├── test_checkpoint/      # 测试检查点  
├── test_output/          # 测试输出
├── final_result/         # 最终结果
├── log/                  # 日志文件
├── report.md            # 本报告
└── trading_results.csv  # 交易结果CSV
```

## 性能分析

### 投资决策执行状态

- **Warmup阶段**: 未完成
- **Test阶段**: 未完成
- **最终结果**: 未生成

### 日志文件

- **Warmup日志**: results/250806_135830_Qwen_Qwen3-8B_JNJ/log/warmup.log
- **Test日志**: results/250806_135830_Qwen_Qwen3-8B_JNJ/log/test.log
- **Trace日志**: results/250806_135830_Qwen_Qwen3-8B_JNJ/log/warmup_trace.log, results/250806_135830_Qwen_Qwen3-8B_JNJ/log/test_trace.log

## CLI命令参考

### 快速执行

```bash
# 运行完整流程
python run.py warmup -c configs/test_minimal.json
python run.py test -c configs/test_minimal.json  
python run.py eval -c configs/test_minimal.json
```

### 详细执行

```bash
# 1. 预热阶段 - 建立智能体记忆
echo "开始预热阶段..."
python run.py warmup --config-path configs/test_minimal.json

# 2. 测试阶段 - 执行投资决策
echo "开始测试阶段..."
python run.py test --config-path configs/test_minimal.json

# 3. 评估阶段 - 生成性能报告  
echo "生成评估报告..."
python run.py eval --config-path configs/test_minimal.json

# 4. 查看结果
echo "结果位于: results/250806_135830_Qwen_Qwen3-8B_JNJ"
ls -la results/250806_135830_Qwen_Qwen3-8B_JNJ/
```

### 从检查点恢复

```bash
# 从warmup检查点恢复
python run.py warmup-checkpoint -c configs/test_minimal.json

# 从test检查点恢复  
python run.py test-checkpoint -c configs/test_minimal.json
```

## 数据文件说明

- **trading_results.csv**: 包含所有交易决策和市场数据
- **warmup_output/**: 预热阶段的智能体状态和环境快照
- **test_output/**: 测试阶段的执行结果
- **final_result/**: 最终的投资组合状态和性能指标

## 快速重现

要重现此次实验，请执行：

```bash
git clone <repository>
cd INVESTOR-BENCH
pip install -r requirements.txt

# 启动Qdrant向量数据库
docker run -p 6333:6333 qdrant/qdrant

# 运行实验
python run.py warmup -c configs/test_minimal.json
python run.py test -c configs/test_minimal.json
python run.py eval -c configs/test_minimal.json
```

---

*报告生成时间: 2025-08-07 00:30:12*
*INVESTOR-BENCH v1.0 - LLM驱动的投资决策评估框架*
