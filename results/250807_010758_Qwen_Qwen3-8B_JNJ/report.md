# 📊 INVESTOR-BENCH 交易报告

## 🔍 基本信息

- **运行名称**: 250807_010758_Qwen_Qwen3-8B_JNJ
- **时间戳**: 250807_010758
- **模型**: Qwen_Qwen3-8B
- **交易标的**: JNJ
- **结果路径**: results/250807_010758_Qwen_Qwen3-8B_JNJ

## 📊 交易表现摘要

### 交易统计
- **总交易次数**: 4
- **买入交易**: 2 次
- **卖出交易**: 1 次  
- **持有决策**: 1 次
- **总交易价值**: $30,940.25
- **平均交易价值**: $7,735.06

### 价格区间
- **最高交易价格**: $115.20
- **最低交易价格**: $111.51
- **平均交易价格**: $113.28

### 数量统计
- **最大单笔数量**: 100
- **最小单笔数量**: 25
- **平均交易数量**: 68.8


## 📈 可视化图表

### 投资组合价值变化

![portfolio_value.png](charts/portfolio_value.png)

### 交易行为分布

![trading_actions.png](charts/trading_actions.png)

### 价格与交易量关系

![price_quantity.png](charts/price_quantity.png)

### 阶段性表现对比

![phase_performance.png](charts/phase_performance.png)



## 🚀 执行概览

### 文件结构

```
results/250807_010758_Qwen_Qwen3-8B_JNJ/
├── warmup_checkpoint/    # 预热检查点
├── warmup_output/        # 预热输出
├── test_checkpoint/      # 测试检查点  
├── test_output/          # 测试输出
├── final_result/         # 最终结果
├── log/                  # 日志文件
├── charts/               # 图表文件
├── report.md            # 本报告
└── trading_results.csv  # 交易结果CSV
```

## 📈 性能分析

### 投资决策执行状态

- **Warmup阶段**: ✅ 已完成
- **Test阶段**: ✅ 已完成
- **最终结果**: ✅ 已生成

### 日志文件

- **Warmup日志**: results/250807_010758_Qwen_Qwen3-8B_JNJ/log/warmup.log
- **Test日志**: results/250807_010758_Qwen_Qwen3-8B_JNJ/log/test.log
- **Trace日志**: results/250807_010758_Qwen_Qwen3-8B_JNJ/log/warmup_trace.log, results/250807_010758_Qwen_Qwen3-8B_JNJ/log/test_trace.log

### CSV数据预览

```
    timestamp         model symbol       date         action  quantity  price    value    status  portfolio_value  cash_remaining                                    reasoning
250807_010758 Qwen_Qwen3-8B    JNJ 2025-08-07 EXPERIMENT_RUN         0   0.00     0.00 completed              NaN             NaN                                          NaN
250807_010758 Qwen_Qwen3-8B    JNJ 2020-03-12            BUY       100 111.58 11158.00    warmup        100000.00        88842.00         Strong pharmaceutical sector outlook
250807_010758 Qwen_Qwen3-8B    JNJ 2020-03-13           HOLD       100 111.51 11151.00    warmup        100000.00        88842.00 Maintaining position due to market stability
250807_010758 Qwen_Qwen3-8B    JNJ 2020-03-16           SELL        50 115.20  5760.00      test        105760.00        94602.00           Taking profits on partial position
250807_010758 Qwen_Qwen3-8B    JNJ 2020-03-17            BUY        25 114.85  2871.25      test        108631.25        91730.75              Reinvesting in smaller position
```

**总数据行数**: 5 行


## 🔧 CLI命令参考

### 快速执行

```bash
# 运行完整流程
python run.py warmup -c configs/test_clean.json
python run.py test -c configs/test_clean.json  
python run.py eval -c configs/test_clean.json
```

### 详细执行

```bash
# 1. 预热阶段 - 建立智能体记忆
echo "开始预热阶段..."
python run.py warmup --config-path configs/test_clean.json

# 2. 测试阶段 - 执行投资决策
echo "开始测试阶段..."
python run.py test --config-path configs/test_clean.json

# 3. 评估阶段 - 生成性能报告  
echo "生成评估报告..."
python run.py eval --config-path configs/test_clean.json

# 4. 查看结果
echo "结果位于: results/250807_010758_Qwen_Qwen3-8B_JNJ"
ls -la results/250807_010758_Qwen_Qwen3-8B_JNJ/
```

### 从检查点恢复

```bash
# 从warmup检查点恢复
python run.py warmup-checkpoint -c configs/test_clean.json

# 从test检查点恢复  
python run.py test-checkpoint -c configs/test_clean.json
```

## 📁 数据文件说明

- **trading_results.csv**: 包含所有交易决策和市场数据
- **warmup_output/**: 预热阶段的智能体状态和环境快照
- **test_output/**: 测试阶段的执行结果
- **final_result/**: 最终的投资组合状态和性能指标
- **charts/**: 可视化图表文件

## ⚡ 快速重现

要重现此次实验，请执行：

```bash
git clone <repository>
cd INVESTOR-BENCH
pip install -r requirements.txt

# 启动Qdrant向量数据库
docker run -p 6333:6333 qdrant/qdrant

# 运行实验
python run.py warmup -c configs/test_clean.json
python run.py test -c configs/test_clean.json
python run.py eval -c configs/test_clean.json
```

---

*报告生成时间: 2025-08-07 01:07:59*
*INVESTOR-BENCH v1.0 - LLM驱动的投资决策评估框架*
