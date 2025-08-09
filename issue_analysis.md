# 单资产交易数量过小问题分析

## 问题描述

在对 `results/250808_235624_Qwen_Qwen2.5-7B-Instruct_JNJ/trading_results.csv` 的交易结果进行分析时，我们发现模型在进行单资产交易时，每次的交易数量都非常小（通常为1或2股）。

## 问题调查过程

1.  **初步调查**：我们首先检查了 `configs` 和 `src/chat/prompt` 目录下的相关文件，但没有发现任何明确限制交易数量的配置或prompt。

2.  **代码追溯**：我们追溯了代码的执行流程，从 `agent.py` 到 `portfolio.py` 和 `portfolio_tools.py`。我们发现，对于单资产交易，`PortfolioSingleAsset` 类只记录了交易的方向（买入/卖出/持有），而没有确定实际的交易数量。

3.  **与原始框架对比**：我们将当前的代码与 `references/INVESTOR-BENCH-main` 中的原始版本进行了比较，发现 `portfolio.py` 和 `portfolio_tools.py` 文件没有变化。这表明问题并非由我们的修改引入。

4.  **深入分析 `run.py`**：我们发现，当前版本的 `run.py` 中新增了 `save_trading_results_csv` 函数，该函数负责将交易结果保存到CSV文件。在这个函数中，交易数量是根据 `PortfolioSingleAsset` 中头寸的变化来计算的。由于 `PortfolioSingleAsset` 中的头寸值只能是-1, 0, 1，因此计算出的交易数量也只能是-2, -1, 0, 1, 2。

5.  **分析原始框架的评估逻辑**：我们接着研究了 `references/INVESTOR-BENCH-main/run.py` 和 `references/INVESTOR-BENCH-main/src/eval_pipeline.py`。我们发现，原始框架在评估单资产交易时，使用的是 `output_metrics_summary_single` 函数。该函数在计算回报时，只考虑了交易的方向（-1, 0, 1），而没有考虑实际的交易数量。它实际上是在评估模型的**方向预测能力**，而不是一个完整的交易策略。

## 结论

当前版本中观察到的单资产交易数量过小的问题，是由于新增的 `save_trading_results_csv` 函数与原始 `PortfolioSingleAsset` 类的设计不匹配造成的。原始框架的设计目标是评估模型的方向预测能力，因此 `PortfolioSingleAsset` 只记录了交易方向，而没有实现交易数量的逻辑。

新增的 `save_trading_results_csv` 函数错误地将 `PortfolioSingleAsset` 中的头寸变化解释为交易数量，从而导致了我们观察到的问题。
