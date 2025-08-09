#!/usr/bin/env python3
"""
Enhanced evaluation pipeline based on original INVESTOR-BENCH framework
扩展原始框架以支持CSV保存和图表生成，但保持核心评估逻辑不变
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
from typing import List, Dict, Any
from datetime import datetime

# 导入原始框架的函数
sys.path.append('references/INVESTOR-BENCH-main/src')
try:
    from eval_pipeline import (
        input_data_restructure,
        reframe_data_files,
        daily_reward,
        total_reward,
        calculate_metrics,
        standard_deviation,
        annualized_volatility,
        calculate_sharpe_ratio,
        calculate_max_drawdown
    )
except ImportError:
    # 如果导入失败，我们需要复制核心函数
    print("Warning: 无法导入原始eval_pipeline，使用内置版本")
    
    def daily_reward(price_list, actions_list):
        """基于原始框架的日收益计算"""
        reward = []
        for i in range(len(price_list) - 1):
            r = actions_list[i] * np.log(price_list[i + 1] / price_list[i])
            reward.append(r)
        return reward

    def total_reward(price_list, actions_list):
        """基于原始框架的累计收益计算"""
        return sum(
            actions_list[i] * np.log(price_list[i + 1] / price_list[i])
            for i in range(len(price_list) - 1)
        )

def enhanced_single_asset_evaluation(
    start_date: str,
    end_date: str,
    ticker: str,
    data_path: str,
    result_path: str,
    output_base_path: str,
    meta_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    基于原始框架的增强单资产评估
    保持原始逻辑，但添加CSV和图表输出
    """
    
    print(f"🚀 开始基于原始框架的增强评估 - {ticker}")
    
    # 1. 使用原始框架加载和处理数据
    print("📊 加载市场数据...")
    full_dates_lst, yahoo_df = load_market_data(start_date, end_date, data_path)
    ticker_stock_price_lst = yahoo_df["Adj Close"].tolist()
    dates_list = yahoo_df.index.tolist()
    
    # 2. 加载模型决策数据（使用原始框架逻辑）
    print("🤖 加载模型决策数据...")
    actions_df = load_model_actions(start_date, end_date, result_path, full_dates_lst, ticker)
    actions_list = actions_df["direction"].tolist()
    
    # 3. 使用原始框架计算指标
    print("📈 计算性能指标...")
    trading_days = 252 if ticker in {"MSFT", "JNJ", "UVV", "HON", "TSLA", "AAPL", "NIO", "ETF"} else 365
    
    # 模型策略指标
    model_cum_return, model_sharpe, model_mdd, model_vol = calculate_metrics(
        ticker_stock_price_lst, actions_list, trading_days
    )
    
    # Buy&Hold基准指标  
    buyhold_cum_return, buyhold_sharpe, buyhold_mdd, buyhold_vol = calculate_metrics(
        ticker_stock_price_lst, [1] * len(ticker_stock_price_lst), trading_days
    )
    
    # 4. 生成增强的CSV报告
    print("💾 生成CSV报告...")
    csv_data = generate_enhanced_csv_data(
        dates_list, ticker_stock_price_lst, actions_list, 
        meta_config, ticker
    )
    
    # 5. 生成增强的图表
    print("📊 生成分析图表...")
    charts_data = generate_enhanced_charts_data(
        dates_list, ticker_stock_price_lst, actions_list,
        meta_config, ticker
    )
    
    # 6. 保存所有结果
    results = {
        'model_metrics': {
            'cumulative_return': model_cum_return,
            'sharpe_ratio': model_sharpe,
            'max_drawdown': model_mdd,
            'volatility': model_vol
        },
        'buyhold_metrics': {
            'cumulative_return': buyhold_cum_return,
            'sharpe_ratio': buyhold_sharpe,  
            'max_drawdown': buyhold_mdd,
            'volatility': buyhold_vol
        },
        'csv_data': csv_data,
        'charts_data': charts_data
    }
    
    # 保存到文件
    save_enhanced_results(results, output_base_path, ticker, meta_config)
    
    print("✅ 增强评估完成")
    return results

def load_market_data(start_date: str, end_date: str, data_path: str):
    """加载市场数据 - 基于原始框架逻辑"""
    try:
        with open(data_path, 'r') as f:
            data = json.load(f)
        
        dates = []
        prices = []
        
        for date_str, content in data.items():
            if content and 'prices' in content and content['prices'] is not None:
                date_obj = pd.to_datetime(date_str)
                if pd.to_datetime(start_date) <= date_obj <= pd.to_datetime(end_date):
                    dates.append(date_obj)
                    prices.append(content['prices'])
        
        df = pd.DataFrame({
            'Adj Close': prices
        }, index=dates)
        
        return dates, df
        
    except Exception as e:
        raise ValueError(f"无法加载市场数据: {e}")

def load_model_actions(start_date: str, end_date: str, result_path: str, full_dates_lst: List, ticker: str):
    """加载模型动作数据 - 基于原始框架逻辑"""
    from src.agent import FinMemAgent
    
    try:
        # 加载agent检查点
        action_path = os.path.join(result_path, "agent")
        agent = FinMemAgent.load_checkpoint(path=action_path)
        
        # 获取动作记录
        action_df = pd.DataFrame(agent.portfolio.get_action_record())
        action_df.drop(columns="price", inplace=True)
        action_df.rename(columns={"position": "direction"}, inplace=True)
        action_df["date"] = pd.to_datetime(action_df["date"])
        action_df["date"] = action_df["date"].dt.date
        
        # 过滤日期范围
        mask = (action_df["date"] >= pd.to_datetime(start_date).date()) & (
            action_df["date"] <= pd.to_datetime(end_date).date()
        )
        filtered_df = action_df.loc[mask]
        
        # 填补缺失日期（使用HOLD = 0）
        existing_dates = set(filtered_df["date"].tolist())
        full_dates_set = {pd.to_datetime(d).date() for d in full_dates_lst}
        missed_dates = list(full_dates_set - existing_dates)
        
        missed_data_df = pd.DataFrame({
            "date": missed_dates,
            "symbol": [ticker] * len(missed_dates),
            "direction": [0] * len(missed_dates),  # 缺失日期默认为HOLD
        })
        
        combined_df = pd.concat([filtered_df, missed_data_df]).sort_values("date").reset_index(drop=True)
        
        return combined_df
        
    except Exception as e:
        raise ValueError(f"无法加载模型动作数据: {e}")

def generate_enhanced_csv_data(dates_list, prices_list, actions_list, meta_config, ticker):
    """基于原始框架逻辑生成CSV数据"""
    
    # 计算基于方向预测的理论投资组合价值
    initial_capital = 100000
    portfolio_values = []
    cumulative_returns = []
    
    # 基于原始框架的逻辑：actions代表仓位方向
    for i, (date, price, action) in enumerate(zip(dates_list, prices_list, actions_list)):
        if i == 0:
            portfolio_values.append(initial_capital)
            cumulative_returns.append(0.0)
        else:
            # 计算日收益率 = position * ln(price_t / price_{t-1})
            daily_return = actions_list[i-1] * np.log(prices_list[i] / prices_list[i-1])
            
            # 累计组合价值
            prev_value = portfolio_values[-1]
            new_value = prev_value * (1 + daily_return)
            portfolio_values.append(new_value)
            
            # 累计收益率
            cum_return = (new_value / initial_capital - 1) * 100
            cumulative_returns.append(cum_return)
    
    # 生成CSV数据
    csv_data = []
    for i, (date, price, action, portfolio_value, cum_return) in enumerate(
        zip(dates_list, prices_list, actions_list, portfolio_values, cumulative_returns)
    ):
        # 转换action到可读格式
        if action == 1:
            action_str = "BUY"
            position_description = "Long (100%)"
        elif action == -1:
            action_str = "SELL" 
            position_description = "Short (100%)"
        else:
            action_str = "HOLD"
            position_description = "Neutral (0%)"
            
        csv_data.append({
            'timestamp': meta_config.get('timestamp', ''),
            'model': meta_config.get('model_name', ''),
            'symbol': ticker,
            'date': date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date),
            'action': action_str,
            'position_direction': action,  # -1, 0, 1
            'position_description': position_description,
            'asset_price': price,
            'portfolio_value': portfolio_value,
            'cumulative_return_pct': cum_return,
            'status': 'test'  # 或者从meta_config获取
        })
    
    return csv_data

def generate_enhanced_charts_data(dates_list, prices_list, actions_list, meta_config, ticker):
    """生成图表数据"""
    
    # 计算累计收益率曲线
    daily_returns = daily_reward(prices_list, actions_list)
    cumulative_returns_strategy = [0]  # 从0开始
    for daily_ret in daily_returns:
        cumulative_returns_strategy.append(cumulative_returns_strategy[-1] + daily_ret)
    
    # 转换为百分比
    cumulative_returns_strategy_pct = [r * 100 for r in cumulative_returns_strategy]
    
    # Buy&Hold基准
    buy_hold_returns = [(prices_list[i] / prices_list[0] - 1) * 100 for i in range(len(prices_list))]
    
    return {
        'dates': dates_list,
        'asset_prices': prices_list,
        'actions': actions_list,
        'strategy_cumulative_returns': cumulative_returns_strategy_pct,
        'buyhold_cumulative_returns': buy_hold_returns
    }

def save_enhanced_results(results, output_base_path, ticker, meta_config):
    """保存增强结果"""
    
    # 创建输出目录
    os.makedirs(output_base_path, exist_ok=True)
    charts_path = os.path.join(output_base_path, 'charts')
    os.makedirs(charts_path, exist_ok=True)
    
    # 1. 保存CSV
    csv_path = os.path.join(output_base_path, 'trading_results_enhanced.csv')
    csv_df = pd.DataFrame(results['csv_data'])
    csv_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"📁 CSV已保存: {csv_path}")
    
    # 2. 保存性能指标
    metrics_data = {
        'Model Strategy': results['model_metrics'],
        'Buy & Hold': results['buyhold_metrics']
    }
    metrics_df = pd.DataFrame(metrics_data).T
    metrics_path = os.path.join(output_base_path, 'performance_metrics.csv')
    metrics_df.to_csv(metrics_path, encoding='utf-8')
    print(f"📊 性能指标已保存: {metrics_path}")
    
    # 3. 生成图表
    generate_enhanced_charts(results['charts_data'], charts_path, ticker, meta_config)

def generate_enhanced_charts(charts_data, charts_path, ticker, meta_config):
    """生成增强图表"""
    
    # 1. 累计收益率对比图
    plt.figure(figsize=(14, 8))
    
    dates = charts_data['dates']
    strategy_returns = charts_data['strategy_cumulative_returns']
    buyhold_returns = charts_data['buyhold_cumulative_returns']
    actions = charts_data['actions']
    
    # 绘制策略收益率
    plt.plot(dates, strategy_returns, 
             marker='o', linewidth=2.5, markersize=4, color='#2E86AB', 
             label=f'{meta_config.get("model_name", "Model")} Strategy')
    
    # 绘制基准收益率
    plt.plot(dates, buyhold_returns, 
             linewidth=2.5, color='orange', alpha=0.8, 
             label=f'{ticker} Buy&Hold')
    
    # 标记交易信号
    buy_dates = [dates[i] for i, action in enumerate(actions) if action == 1]
    sell_dates = [dates[i] for i, action in enumerate(actions) if action == -1]
    buy_returns = [strategy_returns[i] for i, action in enumerate(actions) if action == 1]
    sell_returns = [strategy_returns[i] for i, action in enumerate(actions) if action == -1]
    
    if buy_dates:
        plt.scatter(buy_dates, buy_returns, 
                   color='green', s=100, marker='^', alpha=0.8, label='Buy Signal', zorder=5)
    
    if sell_dates:
        plt.scatter(sell_dates, sell_returns, 
                   color='red', s=100, marker='v', alpha=0.8, label='Sell Signal', zorder=5)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'{ticker} Cumulative Returns Comparison (Original Framework Logic)', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{charts_path}/returns_comparison_enhanced.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 累计收益率图表已保存: {charts_path}/returns_comparison_enhanced.png")
    
    # 2. 投资组合价值与资产价格对比图（双坐标轴）
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 计算理论投资组合价值
    initial_capital = 100000
    portfolio_values = [initial_capital * (1 + r/100) for r in strategy_returns]
    
    # 左坐标轴：投资组合价值
    color1 = '#2E86AB'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(dates, portfolio_values, 
             marker='o', linewidth=2.5, markersize=4, color=color1, 
             label='Portfolio Value (Theory)', alpha=0.8)
    ax1.axhline(y=initial_capital, color=color1, linestyle='--', alpha=0.5, 
                label=f'Initial Capital: ${initial_capital:,}')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.2)
    
    # 右坐标轴：资产价格
    ax2 = ax1.twinx()
    color2 = '#F39C12'
    ax2.set_ylabel('Asset Price ($)', color=color2, fontsize=12, fontweight='bold')
    
    # 归一化价格以对齐起点
    prices = charts_data['asset_prices']
    initial_price = prices[0]
    price_scale = initial_capital / initial_price
    normalized_prices = [p * price_scale for p in prices]
    
    ax2.plot(dates, normalized_prices,
            linewidth=2.5, color=color2, alpha=0.7,
            label=f'{ticker} Price (Normalized)')
    
    # 右轴标签显示实际价格
    ax2_ticks = ax2.get_yticks()
    ax2.set_yticklabels([f'${tick/price_scale:.1f}' for tick in ax2_ticks])
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # 标记交易信号
    if buy_dates:
        buy_portfolio_values = [portfolio_values[i] for i, action in enumerate(actions) if action == 1]
        ax1.scatter(buy_dates, buy_portfolio_values, 
                   color='green', s=100, marker='^', alpha=0.8, label='Buy Signal', 
                   zorder=5, edgecolor='darkgreen')
    
    if sell_dates:
        sell_portfolio_values = [portfolio_values[i] for i, action in enumerate(actions) if action == -1]
        ax1.scatter(sell_dates, sell_portfolio_values, 
                   color='red', s=100, marker='v', alpha=0.8, label='Sell Signal', 
                   zorder=5, edgecolor='darkred')
    
    plt.title(f'{ticker} Portfolio Performance vs Asset Price (Enhanced)', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.savefig(f"{charts_path}/portfolio_value_enhanced.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"📊 投资组合价值图表已保存: {charts_path}/portfolio_value_enhanced.png")

# 测试函数
def test_enhanced_evaluation():
    """测试增强评估功能"""
    
    config = {
        "env_config": {
            "test_start_time": "2020-03-12",
            "test_end_time": "2020-03-30",
            "trading_symbols": ["JNJ"]
        },
        "meta_config": {
            "model_name": "Qwen_Qwen3-8B",
            "timestamp": "250808_230347",
            "result_save_path": "results/250808_230347_Qwen_Qwen3-8B_JNJ"
        }
    }
    
    try:
        results = enhanced_single_asset_evaluation(
            start_date=config["env_config"]["test_start_time"],
            end_date=config["env_config"]["test_end_time"], 
            ticker=config["env_config"]["trading_symbols"][0],
            data_path="data/JNJ.json",
            result_path=os.path.join(config["meta_config"]["result_save_path"], "test_output"),
            output_base_path=os.path.join(config["meta_config"]["result_save_path"], "enhanced_results"),
            meta_config=config["meta_config"]
        )
        
        print("\n🎉 增强评估测试成功!")
        print(f"模型累计收益: {results['model_metrics']['cumulative_return']:.4f}")
        print(f"基准累计收益: {results['buyhold_metrics']['cumulative_return']:.4f}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_evaluation()