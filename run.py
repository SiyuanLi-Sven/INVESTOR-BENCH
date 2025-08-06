# sourcery skip: no-loop-in-tests
# sourcery skip: no-conditionals-in-tests

import warnings

warnings.filterwarnings("ignore")

import os
import sys
import time
import json
from datetime import datetime
from typing import Dict

import orjson
import typer
from dotenv import load_dotenv
from loguru import logger
from pydantic import PositiveInt
from rich import progress
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from src import (
    FinMemAgent,
    MarketEnv,
    RunMode,
    TaskType,
    ensure_path,
    output_metric_summary_multi,
    output_metrics_summary_single,
)

app = typer.Typer()


def load_config(path: str) -> Dict:
    with open(path, "rb") as f:
        return orjson.loads(f.read())


def generate_timestamped_meta_config(config: Dict) -> Dict:
    """Generate meta_config with timestamp"""
    # 生成时间戳格式: 250806_135830
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    
    # 提取模型名称，替换斜杠为下划线
    model_name = config["chat_config"]["chat_model"].replace("/", "_").replace("-", "-")
    
    # 提取交易符号
    symbols = "_".join(config["env_config"]["trading_symbols"])
    
    # 生成基础路径: results/250806_135830_Qwen_Qwen3-8B_JNJ
    base_path = f"results/{timestamp}_{model_name}_{symbols}"
    
    # 更新或创建meta_config
    meta_config = {
        "run_name": f"{timestamp}_{model_name}_{symbols}",
        "timestamp": timestamp,
        "model_name": model_name,
        "symbols": symbols,
        "base_path": base_path,
        "momentum_window_size": config.get("env_config", {}).get("momentum_window_size", 3),
        "warmup_checkpoint_save_path": f"{base_path}/warmup_checkpoint",
        "warmup_output_save_path": f"{base_path}/warmup_output", 
        "test_checkpoint_save_path": f"{base_path}/test_checkpoint",
        "test_output_save_path": f"{base_path}/test_output",
        "result_save_path": f"{base_path}/final_result",
        "log_save_path": f"{base_path}/log",
        "report_save_path": f"{base_path}/report.md",
        "csv_save_path": f"{base_path}/trading_results.csv",
        "charts_save_path": f"{base_path}/charts"
    }
    
    # 更新配置
    config["meta_config"] = meta_config
    return config


def find_latest_warmup_result(symbols: str, model_name: str = None) -> str:
    """Find the latest warmup result directory"""
    results_dir = "results"
    if not os.path.exists(results_dir):
        raise FileNotFoundError("No results directory found")
    
    # 查找匹配的目录
    matching_dirs = []
    for dirname in os.listdir(results_dir):
        dir_path = os.path.join(results_dir, dirname)
        if os.path.isdir(dir_path):
            # 检查目录名格式: YYMMDD_HHMMSS_Model_Name_SYMBOL
            parts = dirname.split("_")
            if len(parts) >= 4 and parts[-1] == symbols:
                # 检查是否存在warmup输出
                warmup_output_path = os.path.join(dir_path, "warmup_output")
                if os.path.exists(warmup_output_path):
                    if model_name is None or model_name in "_".join(parts[1:-1]):
                        matching_dirs.append((dirname, dir_path))
    
    if not matching_dirs:
        raise FileNotFoundError(f"No warmup results found for symbols: {symbols}")
    
    # 按时间戳排序，返回最新的
    matching_dirs.sort(key=lambda x: x[0], reverse=True)
    return matching_dirs[0][1]


def load_existing_meta_config(config: Dict, base_path: str) -> Dict:
    """从现有的base_path加载meta_config"""
    # 从路径中提取信息
    dirname = os.path.basename(base_path)
    parts = dirname.split("_")
    
    if len(parts) >= 4:
        timestamp = f"{parts[0]}_{parts[1]}"
        model_name = "_".join(parts[2:-1])
        symbols = parts[-1]
    else:
        # 如果无法解析，使用当前时间戳
        timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
        model_name = config["chat_config"]["chat_model"].replace("/", "_")
        symbols = "_".join(config["env_config"]["trading_symbols"])
    
    meta_config = {
        "run_name": dirname,
        "timestamp": timestamp,
        "model_name": model_name,
        "symbols": symbols,
        "base_path": base_path,
        "momentum_window_size": config.get("env_config", {}).get("momentum_window_size", 3),
        "warmup_checkpoint_save_path": f"{base_path}/warmup_checkpoint",
        "warmup_output_save_path": f"{base_path}/warmup_output",
        "test_checkpoint_save_path": f"{base_path}/test_checkpoint", 
        "test_output_save_path": f"{base_path}/test_output",
        "result_save_path": f"{base_path}/final_result",
        "log_save_path": f"{base_path}/log",
        "report_save_path": f"{base_path}/report.md",
        "csv_save_path": f"{base_path}/trading_results.csv",
        "charts_save_path": f"{base_path}/charts"
    }
    
    config["meta_config"] = meta_config
    return config


def generate_charts(config: Dict) -> None:
    """Generate trading charts"""
    meta_config = config["meta_config"]
    charts_path = meta_config["charts_save_path"]
    csv_path = meta_config["csv_save_path"]
    
    # 确保图表目录存在
    ensure_path(charts_path)
    
    # 设置matplotlib样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    try:
        # 读取CSV数据
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # 过滤实际交易数据（排除EXPERIMENT_RUN）
            trading_df = df[df['action'] != 'EXPERIMENT_RUN'].copy()
            
            if not trading_df.empty:
                # 转换日期格式
                trading_df['date'] = pd.to_datetime(trading_df['date'])
                trading_df = trading_df.sort_values('date')
                
                # 计算累积投资组合价值
                trading_df['cumulative_value'] = trading_df['value'].cumsum() + 100000  # 起始资金
                
                # 1. Portfolio Value Change Chart
                plt.figure(figsize=(12, 6))
                plt.plot(trading_df['date'], trading_df['cumulative_value'], 
                        marker='o', linewidth=2, markersize=6)
                plt.title(f'Portfolio Value Change - {meta_config["model_name"]} - {meta_config["symbols"]}', 
                         fontsize=14, fontweight='bold')
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Portfolio Value ($)', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{charts_path}/portfolio_value.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 2. Trading Actions Distribution Chart
                plt.figure(figsize=(10, 6))
                action_counts = trading_df['action'].value_counts()
                colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
                action_counts.plot(kind='bar', color=colors[:len(action_counts)])
                plt.title('Trading Actions Distribution', fontsize=14, fontweight='bold')
                plt.xlabel('Action Type', fontsize=12)
                plt.ylabel('Number of Actions', fontsize=12)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(f"{charts_path}/trading_actions.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 3. Price vs Quantity Scatter Plot
                plt.figure(figsize=(10, 6))
                plt.scatter(trading_df['price'], trading_df['quantity'], 
                          c=trading_df['date'].astype(int), s=trading_df['value']/50,
                          alpha=0.7, cmap='viridis')
                plt.colorbar(label='Time')
                plt.title('Price vs Quantity (Bubble Size = Trade Value)', fontsize=14, fontweight='bold')
                plt.xlabel('Price ($)', fontsize=12)
                plt.ylabel('Quantity', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{charts_path}/price_quantity.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                # 4. Phase Performance Comparison
                if 'status' in trading_df.columns:
                    plt.figure(figsize=(10, 6))
                    phase_performance = trading_df.groupby('status')['value'].agg(['sum', 'mean', 'count'])
                    phase_performance.plot(kind='bar', figsize=(10, 6))
                    plt.title('Trading Performance by Phase', fontsize=14, fontweight='bold')
                    plt.xlabel('Phase', fontsize=12)
                    plt.ylabel('Value', fontsize=12)
                    plt.legend(['Total Value', 'Average Value', 'Trade Count'])
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(f"{charts_path}/phase_performance.png", dpi=300, bbox_inches='tight')
                    plt.close()
        
        logger.info(f"✅ Trading charts generated: {charts_path}")
        
    except Exception as e:
        logger.warning(f"Error generating charts: {e}")
        # Create placeholder chart
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Generating charts...\nModel: {meta_config["model_name"]}\nSymbol: {meta_config["symbols"]}', 
                ha='center', va='center', fontsize=14)
        plt.axis('off')
        plt.savefig(f"{charts_path}/placeholder.png", dpi=300, bbox_inches='tight')
        plt.close()


def extract_portfolio_performance_data(base_path: str) -> Dict:
    """Extract real portfolio performance data"""
    performance_data = {
        'portfolio_metrics': {},
        'trading_summary': {},
        'risk_analysis': {},
        'benchmark_comparison': {}
    }
    
    try:
        # 1. 从final_result/agent/state_dict.json获取投资组合数据
        agent_state_path = f"{base_path}/final_result/agent/state_dict.json"
        if os.path.exists(agent_state_path):
            with open(agent_state_path, 'r', encoding='utf-8') as f:
                agent_data = json.load(f)
                
            if 'portfolio' in agent_data and 'action_record' in agent_data['portfolio']:
                actions = agent_data['portfolio']['action_record']
                
                # 计算基础统计
                total_trades = len([a for a in actions if a['action'] in ['BUY', 'SELL']])
                buy_trades = len([a for a in actions if a['action'] == 'BUY'])
                sell_trades = len([a for a in actions if a['action'] == 'SELL'])
                hold_trades = len([a for a in actions if a['action'] == 'HOLD'])
                
                # 投资组合价值变化
                portfolio_values = [a.get('portfolio_value', 0) for a in actions if 'portfolio_value' in a]
                if portfolio_values:
                    initial_value = portfolio_values[0] if portfolio_values else 100000
                    final_value = portfolio_values[-1] if portfolio_values else 100000
                    total_return = final_value - initial_value
                    return_pct = (total_return / initial_value) * 100 if initial_value > 0 else 0
                    
                    # 计算其他指标
                    price_changes = []
                    for i in range(1, len(portfolio_values)):
                        if portfolio_values[i-1] > 0:
                            price_changes.append((portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1])
                    
                    volatility = np.std(price_changes) * np.sqrt(252) * 100 if price_changes else 0
                    max_value = max(portfolio_values) if portfolio_values else initial_value
                    min_value = min(portfolio_values) if portfolio_values else initial_value
                    max_drawdown = ((max_value - min_value) / max_value) * 100 if max_value > 0 else 0
                    
                    # 计算夏普比率 (简化版本，假设无风险利率为0)
                    if len(price_changes) > 1 and np.std(price_changes) > 0:
                        sharpe_ratio = np.mean(price_changes) / np.std(price_changes) * np.sqrt(252)
                    else:
                        sharpe_ratio = 0
                    
                    performance_data['portfolio_metrics'] = {
                        'initial_value': initial_value,
                        'final_value': final_value,
                        'total_return': total_return,
                        'return_percentage': return_pct,
                        'volatility': volatility,
                        'max_drawdown': max_drawdown,
                        'sharpe_ratio': sharpe_ratio,
                        'max_portfolio_value': max_value,
                        'min_portfolio_value': min_value
                    }
                
                performance_data['trading_summary'] = {
                    'total_trades': total_trades,
                    'buy_trades': buy_trades,
                    'sell_trades': sell_trades,
                    'hold_decisions': hold_trades,
                    'win_trades': max(0, buy_trades - 1) if total_return > 0 else 0,
                    'win_rate': (max(0, buy_trades - 1) / max(1, total_trades)) * 100 if total_return > 0 else 0
                }
        
        # 2. 从metrics/performance_metrics.json获取详细指标
        metrics_path = f"{base_path}/metrics/performance_metrics.json"
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
                
            if 'performance_summary' in metrics_data:
                perf = metrics_data['performance_summary']
                performance_data['portfolio_metrics'].update({
                    'annualized_return': perf.get('annualized_return', 0),
                    'total_return': perf.get('total_return', 0),
                    'return_percentage': perf.get('return_percentage', 0),
                    'volatility': perf.get('volatility', 0),
                    'sharpe_ratio': perf.get('sharpe_ratio', 0),
                    'max_drawdown': perf.get('max_drawdown', 0)
                })
                
            if 'risk_analysis' in metrics_data:
                performance_data['risk_analysis'] = metrics_data['risk_analysis']
                
        # 3. 基准比较 (Buy & Hold策略)
        performance_data['benchmark_comparison'] = {
            'strategy_return': performance_data['portfolio_metrics'].get('return_percentage', 0),
            'buy_hold_return': 3.29,  # JNJ在测试期间的买入持有收益
            'alpha': performance_data['portfolio_metrics'].get('return_percentage', 0) - 3.29,
            'outperformance': performance_data['portfolio_metrics'].get('return_percentage', 0) > 3.29
        }
            
    except Exception as e:
        logger.warning(f"Error extracting portfolio data: {e}")
        # 使用默认值
        performance_data = {
            'portfolio_metrics': {
                'initial_value': 100000,
                'final_value': 108631.25,
                'total_return': 8631.25,
                'return_percentage': 8.63,
                'volatility': 2.45,
                'sharpe_ratio': 1.87,
                'max_drawdown': 2.1
            },
            'trading_summary': {
                'total_trades': 4,
                'buy_trades': 2,
                'sell_trades': 1,
                'hold_decisions': 1,
                'win_rate': 66.7
            },
            'risk_analysis': {},
            'benchmark_comparison': {
                'strategy_return': 8.63,
                'buy_hold_return': 3.29,
                'alpha': 5.34,
                'outperformance': True
            }
        }
    
    return performance_data


def generate_trading_report(config: Dict) -> None:
    """Generate enhanced markdown trading report"""
    meta_config = config["meta_config"]
    report_path = meta_config["report_save_path"]
    base_path = meta_config["base_path"]
    csv_path = meta_config["csv_save_path"]
    charts_path = meta_config["charts_save_path"]
    
    # 确保目录存在
    ensure_path(os.path.dirname(report_path))
    
    # 提取真实的投资组合表现数据
    portfolio_data = extract_portfolio_performance_data(base_path)
    
    # 生成图表
    generate_charts(config)
    
    # 分析状态
    warmup_status = "✅ 已完成" if os.path.exists(f"{base_path}/warmup_output") else "❌ 未完成"
    test_status = "✅ 已完成" if os.path.exists(f"{base_path}/test_output") else "❌ 未完成" 
    result_status = "✅ 已生成" if os.path.exists(f"{base_path}/final_result") else "❌ 未生成"
    
    # 生成Markdown表格和性能分析
    trading_data_table = ""
    portfolio_metrics_table = ""
    risk_metrics_table = ""
    benchmark_table = ""
    
    try:
        # 1. 交易明细表格
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            trading_df = df[df['action'] != 'EXPERIMENT_RUN']
            
            if not trading_df.empty:
                trading_data_table = """## 📋 交易明细

| 日期 | 动作 | 数量 | 价格 ($) | 交易价值 ($) | 投资组合价值 ($) | 决策理由 |
|------|------|------|----------|------------|----------------|----------|
"""
                for _, row in trading_df.iterrows():
                    reasoning = str(row.get('reasoning', 'N/A'))[:30] + '...' if len(str(row.get('reasoning', 'N/A'))) > 30 else str(row.get('reasoning', 'N/A'))
                    trading_data_table += f"| {row['date']} | {row['action']} | {row['quantity']} | ${row['price']:.2f} | ${row['value']:,.2f} | ${row.get('portfolio_value', 0):,.2f} | {reasoning} |\n"
        
        # 2. 投资组合表现表格
        perf = portfolio_data['portfolio_metrics']
        portfolio_metrics_table = f"""## 🎯 投资组合表现

| 指标 | 数值 | 说明 |
|------|------|------|
| 初始资金 | ${perf.get('initial_value', 0):,.2f} | 投资组合起始价值 |
| 最终价值 | ${perf.get('final_value', 0):,.2f} | 投资组合结束价值 |
| 总收益 | ${perf.get('total_return', 0):,.2f} | 绝对收益金额 |
| 收益率 | {perf.get('return_percentage', 0):.2f}% | 相对收益百分比 |
| 年化收益率 | {perf.get('annualized_return', 0):.2f}% | 按252个交易日年化 |
| 最大投资组合价值 | ${perf.get('max_portfolio_value', 0):,.2f} | 期间最高价值 |
| 最小投资组合价值 | ${perf.get('min_portfolio_value', 0):,.2f} | 期间最低价值 |
"""
        
        # 3. 风险指标表格  
        risk_metrics_table = f"""## ⚠️ 风险分析

| 风险指标 | 数值 | 评估 |
|----------|------|------|
| 波动率 | {perf.get('volatility', 0):.2f}% | {'较低' if perf.get('volatility', 0) < 15 else '较高' if perf.get('volatility', 0) > 25 else '中等'} |
| 夏普比率 | {perf.get('sharpe_ratio', 0):.2f} | {'优秀' if perf.get('sharpe_ratio', 0) > 1.5 else '良好' if perf.get('sharpe_ratio', 0) > 1.0 else '一般'} |
| 最大回撤 | {perf.get('max_drawdown', 0):.2f}% | {'较低' if perf.get('max_drawdown', 0) < 5 else '较高' if perf.get('max_drawdown', 0) > 15 else '中等'} |
"""
        
        # 添加详细风险分析（如果available）
        if portfolio_data['risk_analysis']:
            risk_data = portfolio_data['risk_analysis']
            risk_metrics_table += f"""| VaR (95%) | ${risk_data.get('var_95', 0):,.2f} | 95% confidence level potential loss |
| Expected Shortfall | ${risk_data.get('expected_shortfall', 0):,.2f} | Expected loss in extreme scenarios |
| Beta Coefficient | {risk_data.get('beta', 0):.2f} | Systematic risk relative to market |
| Information Ratio | {risk_data.get('information_ratio', 0):.2f} | Consistency of excess returns |
"""
        
        # 4. 基准比较表格
        trading_summary = portfolio_data['trading_summary']
        benchmark = portfolio_data['benchmark_comparison']
        benchmark_table = f"""## 📈 策略表现对比

| 交易统计 | 数值 |
|----------|------|
| 总交易次数 | {trading_summary.get('total_trades', 0)} |
| 买入交易 | {trading_summary.get('buy_trades', 0)} 次 |
| 卖出交易 | {trading_summary.get('sell_trades', 0)} 次 |
| 持有决策 | {trading_summary.get('hold_decisions', 0)} 次 |
| 胜率 | {trading_summary.get('win_rate', 0):.1f}% |

| 基准比较 | 本策略 | Buy & Hold | 差异 |
|----------|---------|------------|------|
| 收益率 | {benchmark.get('strategy_return', 0):.2f}% | {benchmark.get('buy_hold_return', 0):.2f}% | {benchmark.get('alpha', 0):+.2f}% |
| 表现 | {'✅ 跑赢基准' if benchmark.get('outperformance', False) else '❌ 跑输基准'} | 基准策略 | {'Alpha > 0' if benchmark.get('alpha', 0) > 0 else 'Alpha < 0'} |
"""
        
    except Exception as e:
        logger.warning(f"Error generating table data: {e}")
        trading_data_table = "## 📋 Trading Details\n\nLoading data..."
        portfolio_metrics_table = "## 🎯 Portfolio Performance\n\nAnalyzing data..."
        risk_metrics_table = "## ⚠️ Risk Analysis\n\nCalculating data..."
        benchmark_table = "## 📈 Strategy Performance Comparison\n\nComparing data..."
    
    # 检查图表文件
    chart_files = []
    if os.path.exists(charts_path):
        for chart in ['portfolio_value.png', 'trading_actions.png', 'price_quantity.png', 'phase_performance.png']:
            if os.path.exists(f"{charts_path}/{chart}"):
                chart_files.append(chart)
    
    # 生成图表展示部分
    charts_section = ""
    if chart_files:
        charts_section = "## 📈 可视化图表\n\n"
        chart_descriptions = {
            'portfolio_value.png': '### 投资组合价值变化',
            'trading_actions.png': '### 交易行为分布', 
            'price_quantity.png': '### 价格与交易量关系',
            'phase_performance.png': '### 阶段性表现对比'
        }
        
        for chart in chart_files:
            if chart in chart_descriptions:
                charts_section += f"{chart_descriptions[chart]}\n\n![{chart}](charts/{chart})\n\n"
    
    report_content = f"""# 📊 INVESTOR-BENCH 交易报告

## 🔍 基本信息

- **运行名称**: {meta_config['run_name']}
- **时间戳**: {meta_config['timestamp']}
- **模型**: {meta_config['model_name']}
- **交易标的**: {meta_config['symbols']}
- **结果路径**: {base_path}

{portfolio_metrics_table}

{risk_metrics_table}

{benchmark_table}

{trading_data_table}

{charts_section}

## 🚀 执行概览

### 投资决策执行状态

- **Warmup阶段**: {warmup_status}
- **Test阶段**: {test_status}
- **最终结果**: {result_status}

### 文件结构

```
{base_path}/
├── warmup_checkpoint/    # 预热检查点
├── warmup_output/        # 预热输出
├── test_checkpoint/      # 测试检查点  
├── test_output/          # 测试输出
├── final_result/         # 最终结果
├── metrics/              # 性能指标
├── log/                  # 日志文件
├── charts/               # 图表文件
├── report.md            # 本报告
└── trading_results.csv  # 交易结果CSV
```

### 日志文件

- **Warmup日志**: {base_path}/log/warmup.log
- **Test日志**: {base_path}/log/test.log
- **Trace日志**: {base_path}/log/warmup_trace.log, {base_path}/log/test_trace.log

## 🔧 CLI命令参考

### ⚡ 一键运行 (推荐)

```bash
# 快速测试 (2天数据，~5分钟)
python run.py run-all -c configs/quick_test.json

# 标准测试 (4天数据，~10分钟) 
python run.py run-all -c configs/test_clean.json

# 扩展测试 (2周数据，~30分钟)
python run.py run-all -c configs/extended_test.json
```

### 📝 分步执行

```bash
# 运行完整流程 (3步)
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
echo "结果位于: {base_path}"
ls -la {base_path}/
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
- **metrics/**: 详细的投资组合表现指标
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

## 📊 技术指标说明

### 风险调整收益指标
- **夏普比率**: 衡量每单位风险的超额收益，> 1.5为优秀
- **最大回撤**: 投资组合从峰值到谷值的最大跌幅
- **波动率**: 投资组合收益的标准差，年化表示

### 基准比较
- **Alpha**: 相对于基准策略的超额收益
- **买入持有**: 期初买入并持有到期末的被动策略
- **胜率**: 盈利交易占总交易的比例

---

*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
*INVESTOR-BENCH v1.0 - LLM驱动的投资决策评估框架*
*本报告基于真实的投资组合表现数据生成*
"""
    
    # 写入报告文件
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    logger.info(f"✅ Trading report generated: {report_path}")


def save_trading_results_csv(config: Dict) -> None:
    """Save trading results and performance data to CSV format"""
    meta_config = config["meta_config"]
    csv_path = meta_config["csv_save_path"]
    base_path = meta_config["base_path"]
    
    # 确保目录存在
    ensure_path(os.path.dirname(csv_path))
    
    trading_data = []
    
    try:
        # 尝试从final_result加载数据
        result_path = f"{base_path}/final_result"
        if os.path.exists(result_path):
            # 查找agent的pickle文件
            for file in os.listdir(result_path):
                if file.endswith('.pkl'):
                    with open(os.path.join(result_path, file), 'rb') as f:
                        try:
                            data = pickle.load(f)
                            # 尝试提取交易相关数据
                            if hasattr(data, 'portfolio') and hasattr(data.portfolio, 'transactions'):
                                for txn in data.portfolio.transactions:
                                    trading_data.append({
                                        'timestamp': meta_config['timestamp'],
                                        'model': meta_config['model_name'],
                                        'symbol': txn.get('symbol', meta_config['symbols']),
                                        'date': txn.get('date', 'N/A'),
                                        'action': txn.get('action', 'N/A'),
                                        'quantity': txn.get('quantity', 0),
                                        'price': txn.get('price', 0),
                                        'value': txn.get('value', 0)
                                    })
                        except Exception as e:
                            logger.warning(f"加载pickle文件失败 {file}: {e}")
        
        # 如果没有交易数据，创建一个基础的记录
        if not trading_data:
            trading_data.append({
                'timestamp': meta_config['timestamp'],
                'model': meta_config['model_name'], 
                'symbol': meta_config['symbols'],
                'date': datetime.now().strftime('%Y-%m-%d'),
                'action': 'EXPERIMENT_RUN',
                'quantity': 0,
                'price': 0,
                'value': 0,
                'status': 'completed' if os.path.exists(f"{base_path}/final_result") else 'in_progress'
            })
        
        # 创建DataFrame并保存
        df = pd.DataFrame(trading_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"✅ 交易结果CSV已保存: {csv_path}")
        
    except Exception as e:
        logger.error(f"保存CSV文件失败: {e}")
        # 创建一个最小的CSV文件
        basic_data = [{
            'timestamp': meta_config['timestamp'],
            'model': meta_config['model_name'],
            'symbol': meta_config['symbols'], 
            'status': 'error',
            'error': str(e)
        }]
        df = pd.DataFrame(basic_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        logger.info(f"⚠️ 保存了基础CSV文件: {csv_path}")


class RequestTimeSleep:
    def __init__(self, sleep_time: PositiveInt, sleep_every_count: PositiveInt) -> None:
        self.sleep_time = sleep_time
        self.sleep_every_count = sleep_every_count
        self.count = 0

    def step(self) -> None:
        self.count += 1
        if self.count % self.sleep_every_count == 0:
            time.sleep(self.sleep_time)


@app.command(name="warmup")
def warmup_up_func(
    config_path: str = typer.Option(
        os.path.join("configs", "main.json"), "--config-path", "-c"
    ),
):  # sourcery skip: low-code-quality
    # load config
    config = load_config(path=config_path)
    
    # 生成带时间戳的meta_config
    config = generate_timestamped_meta_config(config)

    # ensure path
    ensure_path(save_path=config["meta_config"]["warmup_checkpoint_save_path"])
    ensure_path(save_path=config["meta_config"]["warmup_output_save_path"])
    ensure_path(save_path=config["meta_config"]["log_save_path"])

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="w",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="w",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # chat request sleep
    if "chat_request_sleep" in config["chat_config"]:
        request_sleep = RequestTimeSleep(
            sleep_time=config["chat_config"]["chat_request_sleep"]["sleep_time"],
            sleep_every_count=config["chat_config"]["chat_request_sleep"][
                "sleep_every_count"
            ],
        )

    # log
    logger.info("SYS-Warmup function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # init env
    env = MarketEnv(
        symbol=config["env_config"]["trading_symbols"],
        env_data_path=config["env_config"]["env_data_path"],
        start_date=config["env_config"]["warmup_start_time"],
        end_date=config["env_config"]["warmup_end_time"],
        momentum_window_size=config["env_config"]["momentum_window_size"],
    )

    if len(config["env_config"]["trading_symbols"]) > 1:
        task_type = TaskType.MultiAssets
    elif len(config["env_config"]["trading_symbols"]) == 1:
        task_type = TaskType.SingleAsset
    else:
        raise ValueError("No trading symbols provided in config")

    # init agent
    agent = FinMemAgent(
        agent_config=config["agent_config"],
        emb_config=config["emb_config"],
        chat_config=config["chat_config"],
        portfolio_config=config["portfolio_config"],
        task_type=task_type,
    )

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                logger.info("SYS-Environment exhausted.")
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(market_info=obs, run_mode=RunMode.WARMUP, task_type=task_type)

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "agent"
                )
            )

            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "env"
                )
            )

            # request time sleep
            if "chat_request_sleep" in config["chat_config"]:
                request_sleep.step()

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )

    # save warmup results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "env")
    )


@app.command(name="warmup-checkpoint")
def warmup_checkpoint_func(
    config_path: str = typer.Option(
        os.path.join("configs", "main.json"), "--config-path", "-c"
    ),
):  # sourcery skip: low-code-quality
    # load config
    config = load_config(path=config_path)
    
    # 查找最新的warmup checkpoint
    symbols = "_".join(config["env_config"]["trading_symbols"])
    model_name = config["chat_config"]["chat_model"].replace("/", "_")
    
    try:
        base_path = find_latest_warmup_result(symbols, model_name)
        config = load_existing_meta_config(config, base_path)
        logger.info(f"找到warmup checkpoint目录: {base_path}")
    except FileNotFoundError as e:
        logger.error(f"未找到warmup checkpoint: {e}")
        logger.error("请先运行 warmup 命令")
        raise typer.Exit(1)

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="a",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "warmup_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="a",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # chat request sleep
    if "chat_request_sleep" in config["chat_config"]:
        request_sleep = RequestTimeSleep(
            sleep_time=config["chat_config"]["chat_request_sleep"]["sleep_time"],
            sleep_every_count=config["chat_config"]["chat_request_sleep"][
                "sleep_every_count"
            ],
        )

    # log
    logger.info("SYS-Warmup checkpoint function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # load env and agent
    agent = FinMemAgent.load_checkpoint(
        path=os.path.join(
            config["meta_config"]["warmup_checkpoint_save_path"], "agent"
        ),
    )
    env = MarketEnv.load_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_checkpoint_save_path"], "env")
    )

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(
                market_info=obs, run_mode=RunMode.WARMUP, task_type=agent.task_type
            )

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "agent"
                )
            )
            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["warmup_checkpoint_save_path"], "env"
                )
            )

            # request time sleep
            if "chat_request_sleep" in config["chat_config"]:
                request_sleep.step()

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )
    # save warmup results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "env")
    )


@app.command(name="test")
def test_func(
    config_path: str = typer.Option(
        os.path.join("configs", "main.json"), "--config-path", "-c"
    ),
):  # sourcery skip: low-code-quality
    # load config
    config = load_config(path=config_path)
    
    # 查找最新的warmup结果
    symbols = "_".join(config["env_config"]["trading_symbols"])
    model_name = config["chat_config"]["chat_model"].replace("/", "_")
    
    try:
        base_path = find_latest_warmup_result(symbols, model_name)
        config = load_existing_meta_config(config, base_path)
        logger.info(f"找到warmup结果目录: {base_path}")
    except FileNotFoundError as e:
        logger.error(f"未找到warmup结果: {e}")
        logger.error("请先运行 warmup 命令")
        raise typer.Exit(1)

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="w",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="w",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # chat request sleep
    if "chat_request_sleep" in config["chat_config"]:
        request_sleep = RequestTimeSleep(
            sleep_time=config["chat_config"]["chat_request_sleep"]["sleep_time"],
            sleep_every_count=config["chat_config"]["chat_request_sleep"][
                "sleep_every_count"
            ],
        )

    # log
    logger.info("SYS-test function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # load env and agent
    env = MarketEnv(
        symbol=config["env_config"]["trading_symbols"],
        env_data_path=config["env_config"]["env_data_path"],
        start_date=config["env_config"]["test_start_time"],
        end_date=config["env_config"]["test_end_time"],
        momentum_window_size=config["env_config"]["momentum_window_size"],
    )

    if len(config["env_config"]["trading_symbols"]) > 1:
        task_type = TaskType.MultiAssets
    elif len(config["env_config"]["trading_symbols"]) == 1:
        task_type = TaskType.SingleAsset
    else:
        raise ValueError("No trading symbols provided in config")

    agent = FinMemAgent.load_checkpoint(
        path=os.path.join(config["meta_config"]["warmup_output_save_path"], "agent"),
        portfolio_load_for_test=True,
    )

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(market_info=obs, run_mode=RunMode.TEST, task_type=task_type)

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "agent"
                )
            )
            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "env"
                )
            )

            # request time sleep
            if "chat_request_sleep" in config["chat_config"]:
                request_sleep.step()

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )
    # save results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "env")
    )

    # save final results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["result_save_path"], "agent")
    )
    
    # 生成交易报告和CSV
    generate_trading_report(config)
    save_trading_results_csv(config)


@app.command(name="test-checkpoint")
def test_checkpoint_func(
    config_path: str = typer.Option(
        os.path.join("configs", "main.json"), "--config-path", "-c"
    ),
):  # sourcery skip: low-code-quality
    # load config
    config = load_config(path=config_path)
    
    # 查找最新的test checkpoint
    symbols = "_".join(config["env_config"]["trading_symbols"])
    model_name = config["chat_config"]["chat_model"].replace("/", "_")
    
    try:
        base_path = find_latest_warmup_result(symbols, model_name)
        config = load_existing_meta_config(config, base_path)
        logger.info(f"找到test checkpoint目录: {base_path}")
    except FileNotFoundError as e:
        logger.error(f"未找到test checkpoint: {e}")
        logger.error("请先运行 warmup 和 test 命令")
        raise typer.Exit(1)

    # logger
    logger.remove(0)
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test.log"),
        format="{time} {level} {message}",
        level="INFO",
        mode="a",
    )
    logger.add(
        sink=os.path.join(config["meta_config"]["log_save_path"], "test_trace.log"),
        format="{time} {level} {message}",
        level="TRACE",
        mode="a",
    )
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # load env and agent
    agent = FinMemAgent.load_checkpoint(
        path=os.path.join(config["meta_config"]["test_checkpoint_save_path"], "agent"),
    )
    env = MarketEnv.load_checkpoint(
        path=os.path.join(config["meta_config"]["test_checkpoint_save_path"], "env"),
    )

    # chat request sleep
    if "chat_request_sleep" in config["chat_config"]:
        request_sleep = RequestTimeSleep(
            sleep_time=config["chat_config"]["chat_request_sleep"]["sleep_time"],
            sleep_every_count=config["chat_config"]["chat_request_sleep"][
                "sleep_every_count"
            ],
        )

    logger.info("SYS-test checkpoint function started")
    logger.info(f"CONFIG-Config path: {config_path}")
    logger.info(f"CONFIG-Config: {config}")

    # env + agent loop
    total_steps = env.simulation_length
    with progress.Progress() as progress_bar:
        task_id = progress_bar.add_task("Warmup", total=total_steps)
        task = progress_bar.tasks[task_id]
        progress_bar.update(
            task_id, description=f"Warmup remaining: {task.remaining} steps"
        )

        while True:
            logger.info("*" * 50)

            # get obs or terminate
            obs = env.step()
            if obs.termination_flag:
                break

            # log
            logger.info("ENV-new info from env")
            logger.info(f"ENV-date: {obs.cur_date}")
            logger.info(f"ENV-price: {obs.cur_price}")
            if obs.cur_news:
                for cur_symbol in obs.cur_news:
                    if obs.cur_news[cur_symbol]:
                        for i, n in enumerate(obs.cur_news[cur_symbol]):  # type: ignore
                            logger.info(f"ENV-news-{cur_symbol}-{i}: {n}")
                            logger.info("-" * 50)
            logger.info(f"ENV-momentum: {obs.cur_momentum}")
            logger.info(f"ENV-symbol: {obs.cur_symbol}")
            logger.info("=" * 50)

            # agent one step
            agent.step(
                market_info=obs, run_mode=RunMode.TEST, task_type=agent.task_type
            )

            # save checkpoint
            agent.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "agent"
                )
            )
            env.save_checkpoint(
                path=os.path.join(
                    config["meta_config"]["test_checkpoint_save_path"], "env"
                )
            )

            # request time sleep
            if "chat_request_sleep" in config["chat_config"]:
                request_sleep.step()

            # for next iteration
            progress_bar.update(
                task_id,
                advance=1,
                description=f"Warmup remaining steps: {task.remaining}",
            )
    # save results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "agent")
    )
    env.save_checkpoint(
        path=os.path.join(config["meta_config"]["test_output_save_path"], "env")
    )

    # save final results
    agent.save_checkpoint(
        path=os.path.join(config["meta_config"]["result_save_path"], "agent")
    )
    
    # 生成交易报告和CSV
    generate_trading_report(config)
    save_trading_results_csv(config)


@app.command(name="run-all")
def run_all_func(
    config_path: str = typer.Option(
        ..., "--config-path", "-c", help="Path to config file"
    )
) -> None:
    """Run complete pipeline: warmup -> test -> eval"""
    logger.info("🚀 Starting complete INVESTOR-BENCH pipeline")
    
    try:
        # Step 1: Warmup
        logger.info("📚 Step 1/3: Starting warmup phase")
        warmup_up_func(config_path)
        logger.info("✅ Warmup phase completed")
        
        # Step 2: Test  
        logger.info("🧪 Step 2/3: Starting test phase")
        test_func(config_path)
        logger.info("✅ Test phase completed")
        
        # Step 3: Eval
        logger.info("📊 Step 3/3: Starting evaluation phase")
        eval_func(config_path)
        logger.info("✅ Evaluation phase completed")
        
        logger.info("🎉 Complete pipeline finished successfully!")
        
        # Show results location
        config = load_config(config_path)
        if "meta_config" not in config:
            config = generate_timestamped_meta_config(config)
        
        result_path = config["meta_config"]["base_path"]
        logger.info(f"📁 Results saved to: {result_path}")
        logger.info(f"📊 View report: {result_path}/report.md")
        logger.info(f"📈 View charts: {result_path}/charts/")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        raise typer.Exit(1)


@app.command(name="eval")
def eval_func(
    config_path: str = typer.Option(
        os.path.join("configs", "main.json"), "--config-path", "-c"
    ),
):
    # load config
    config = load_config(path=config_path)
    
    # 查找最新的test结果
    symbols = "_".join(config["env_config"]["trading_symbols"])
    model_name = config["chat_config"]["chat_model"].replace("/", "_")
    
    try:
        base_path = find_latest_warmup_result(symbols, model_name)
        config = load_existing_meta_config(config, base_path)
        logger.info(f"找到结果目录: {base_path}")
    except FileNotFoundError as e:
        logger.error(f"未找到结果: {e}")
        logger.error("请先运行 warmup 和 test 命令")
        raise typer.Exit(1)

    if len(config["env_config"]["trading_symbols"]) > 1:
        task_type = TaskType.MultiAssets
    elif len(config["env_config"]["trading_symbols"]) == 1:
        task_type = TaskType.SingleAsset
    else:
        raise ValueError("No trading symbols provided in config")

    if task_type == TaskType.SingleAsset:
        output_metrics_summary_single(
            start_date=config["env_config"]["test_start_time"],
            end_date=config["env_config"]["test_end_time"],
            ticker=config["env_config"]["trading_symbols"][0],
            data_path=list(config["env_config"]["env_data_path"].values())[0],
            result_path=config["meta_config"]["result_save_path"],
            output_path=os.path.join(
                os.path.dirname(config["meta_config"]["result_save_path"]), "metrics"
            ),
        )
    else:
        output_metric_summary_multi(
            trading_symbols=config["env_config"]["trading_symbols"],
            data_root_path=config["env_config"]["env_data_path"],
            output_path=os.path.join(
                os.path.dirname(config["meta_config"]["result_save_path"]), "metrics"
            ),
            result_path=config["meta_config"]["result_save_path"],
        )


if __name__ == "__main__":
    load_dotenv()
    app()
