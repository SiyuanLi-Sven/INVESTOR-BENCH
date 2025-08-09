#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import json
import numpy as np

def load_benchmark_data(data_path: str, start_date, end_date):
    """Load benchmark asset price data"""
    try:
        if os.path.exists(data_path):
            with open(data_path, 'r') as f:
                data = json.load(f)
            
            dates = []
            prices = []
            
            for date_str, content in data.items():
                if content and 'prices' in content and content['prices'] is not None:
                    date_obj = pd.to_datetime(date_str)
                    if start_date <= date_obj <= end_date:
                        dates.append(date_obj)
                        prices.append(content['prices'])
            
            if dates:
                benchmark_df = pd.DataFrame({
                    'date': dates,
                    'price': prices
                }).sort_values('date').reset_index(drop=True)
                
                initial_price = benchmark_df['price'].iloc[0]
                benchmark_df['cumulative_return'] = (benchmark_df['price'] / initial_price - 1) * 100
                
                return benchmark_df
    except Exception as e:
        print(f"Warning: Could not load benchmark data: {e}")
    
    return pd.DataFrame()

def generate_returns_comparison_chart_fixed(trading_df: pd.DataFrame, benchmark_df: pd.DataFrame, charts_path: str, meta_config):
    """Generate cumulative returns comparison chart - FIXED VERSION"""
    plt.figure(figsize=(12, 8))
    
    # 重新计算正确的投资组合价值和累计收益率
    initial_value = 100000
    trading_df_copy = trading_df.copy()
    
    # 重新计算每个时点的投资组合真实价值
    corrected_portfolio_values = []
    current_cash = initial_value
    current_position = 0
    
    for _, row in trading_df_copy.iterrows():
        if row['action'] == 'BUY':
            shares_bought = row['quantity']
            cost = shares_bought * row['price']
            current_cash -= cost
            current_position += shares_bought
        elif row['action'] == 'SELL':
            shares_sold = row['quantity']
            proceeds = shares_sold * row['price']
            current_cash += proceeds
            current_position -= shares_sold
        # HOLD时不改变现金和持仓
        
        # 计算当前投资组合总价值 = 现金 + 持仓市值
        portfolio_value = current_cash + (current_position * row['price'])
        corrected_portfolio_values.append(portfolio_value)
    
    trading_df_copy['corrected_portfolio_value'] = corrected_portfolio_values
    trading_df_copy['portfolio_return'] = (trading_df_copy['corrected_portfolio_value'] / initial_value - 1) * 100
    
    # 绘制投资组合收益率
    plt.plot(trading_df_copy['date'], trading_df_copy['portfolio_return'], 
             marker='o', linewidth=2.5, markersize=4, color='#2E86AB', 
             label=f'{meta_config["model_name"]} Strategy')
    
    # 绘制基准收益率
    if not benchmark_df.empty:
        plt.plot(benchmark_df['date'], benchmark_df['cumulative_return'], 
                 linewidth=2.5, color='orange', alpha=0.8, 
                 label=f'{meta_config["symbols"]} Buy&Hold')
    
    # 标记买卖操作
    buy_trades = trading_df_copy[trading_df_copy['action'] == 'BUY']
    sell_trades = trading_df_copy[trading_df_copy['action'] == 'SELL']
    
    if not buy_trades.empty:
        plt.scatter(buy_trades['date'], buy_trades['portfolio_return'], 
                   color='green', s=100, marker='^', alpha=0.8, label='Buy', zorder=5)
    
    if not sell_trades.empty:
        plt.scatter(sell_trades['date'], sell_trades['portfolio_return'], 
                   color='red', s=100, marker='v', alpha=0.8, label='Sell', zorder=5)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title(f'{meta_config["symbols"]} Cumulative Returns Comparison', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Return (%)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{charts_path}/returns_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_portfolio_value_chart_fixed(trading_df: pd.DataFrame, benchmark_df: pd.DataFrame, charts_path: str, meta_config):
    """Generate portfolio value vs benchmark price comparison chart with dual y-axis - FIXED VERSION"""
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 重新计算正确的投资组合价值
    initial_value = 100000
    trading_df_copy = trading_df.copy()
    
    corrected_portfolio_values = []
    current_cash = initial_value
    current_position = 0
    
    for _, row in trading_df_copy.iterrows():
        if row['action'] == 'BUY':
            shares_bought = row['quantity']
            cost = shares_bought * row['price']
            current_cash -= cost
            current_position += shares_bought
        elif row['action'] == 'SELL':
            shares_sold = row['quantity']
            proceeds = shares_sold * row['price']
            current_cash += proceeds
            current_position -= shares_sold
        
        portfolio_value = current_cash + (current_position * row['price'])
        corrected_portfolio_values.append(portfolio_value)
    
    trading_df_copy['corrected_portfolio_value'] = corrected_portfolio_values
    
    # 左坐标轴: 投资组合价值
    color1 = '#2E86AB'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', color=color1, fontsize=12, fontweight='bold')
    line1 = ax1.plot(trading_df_copy['date'], trading_df_copy['corrected_portfolio_value'], 
                     marker='o', linewidth=2.5, markersize=4, color=color1, 
                     label='Portfolio Value', alpha=0.8)
    
    # 投资组合初始价值参考线
    ax1.axhline(y=initial_value, color=color1, linestyle='--', alpha=0.5, 
                label=f'Initial Capital: ${initial_value:,}')
    
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.2)
    
    # 右坐标轴: 资产价格
    if not benchmark_df.empty:
        ax2 = ax1.twinx()
        color2 = '#F39C12'
        ax2.set_ylabel('Asset Price ($)', color=color2, fontsize=12, fontweight='bold')
        
        # 计算价格缩放，使起始价格在图上对齐
        initial_price = benchmark_df['price'].iloc[0]
        price_scale = initial_value / initial_price
        
        # 为了在图上对齐起点，我们绘制归一化后的价格
        normalized_prices = benchmark_df['price'] * price_scale
        line2 = ax2.plot(benchmark_df['date'], normalized_prices,
                        linewidth=2.5, color=color2, alpha=0.7,
                        label=f'{meta_config["symbols"]} Price (Normalized)')
        
        # 右轴标签显示实际价格
        actual_ticks = ax2.get_yticks()
        ax2.set_yticklabels([f'${tick/price_scale:.1f}' for tick in actual_ticks])
        ax2.tick_params(axis='y', labelcolor=color2)
    
    # 标记交易信号
    buy_trades = trading_df_copy[trading_df_copy['action'] == 'BUY']
    sell_trades = trading_df_copy[trading_df_copy['action'] == 'SELL']
    
    if not buy_trades.empty:
        ax1.scatter(buy_trades['date'], buy_trades['corrected_portfolio_value'], 
                   color='green', s=100, marker='^', alpha=0.8, label='Buy Signal', 
                   zorder=5, edgecolor='darkgreen')
    
    if not sell_trades.empty:
        ax1.scatter(sell_trades['date'], sell_trades['corrected_portfolio_value'], 
                   color='red', s=100, marker='v', alpha=0.8, label='Sell Signal', 
                   zorder=5, edgecolor='darkred')
    
    # 标题和格式化
    plt.title(f'{meta_config["symbols"]} Portfolio Performance vs Asset Price', 
              fontsize=16, fontweight='bold', pad=20)
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    if not benchmark_df.empty:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)
    else:
        ax1.legend(loc='upper left', framealpha=0.9)
    
    # 格式化x轴
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.savefig(f"{charts_path}/portfolio_value.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    print("🚀 开始生成修复后的投资分析图表...")
    
    # 读取交易数据
    trading_df = pd.read_csv('results/250808_230347_Qwen_Qwen3-8B_JNJ/trading_results.csv')
    trading_df['date'] = pd.to_datetime(trading_df['date'])
    
    # 加载基准数据
    benchmark_data = load_benchmark_data('data/JNJ.json', 
                                        pd.to_datetime('2020-03-12'), 
                                        pd.to_datetime('2020-03-30'))
    
    # 创建元配置
    meta_config = {
        'model_name': 'Qwen_Qwen3-8B', 
        'symbols': 'JNJ'
    }
    
    # 创建输出文件夹
    charts_path = 'results/250808_230347_Qwen_Qwen3-8B_JNJ/charts_fixed'
    os.makedirs(charts_path, exist_ok=True)
    
    try:
        print("📈 正在生成修复后的累计回报率对比图...")
        generate_returns_comparison_chart_fixed(trading_df, benchmark_data, charts_path, meta_config)
        print("✅ returns_comparison.png 已生成")
        
        print("📊 正在生成修复后的投资组合价值双轴图...")
        generate_portfolio_value_chart_fixed(trading_df, benchmark_data, charts_path, meta_config)
        print("✅ portfolio_value.png 已生成")
        
        print(f"\n🎯 所有修复后的图表已保存到: {charts_path}/")
        print("主要修复:")
        print("  1. 累计回报率现在正确反映持仓价值变化")
        print("  2. 组合价值图使用专业的双坐标轴设计")
        print("  3. 金融数据计算更加准确和专业")
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()