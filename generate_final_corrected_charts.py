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

def generate_corrected_returns_comparison_chart(trading_df: pd.DataFrame, benchmark_df: pd.DataFrame, charts_path: str, meta_config):
    """Generate corrected cumulative returns comparison chart"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # 直接使用原始数据的累计收益率
    initial_value = 100000
    trading_df_copy = trading_df.copy()
    trading_df_copy['portfolio_return'] = (trading_df_copy['portfolio_value'] / initial_value - 1) * 100
    
    # 上图：完整对比图
    ax1.plot(trading_df_copy['date'], trading_df_copy['portfolio_return'], 
             marker='o', linewidth=2.5, markersize=4, color='#2E86AB', 
             label=f'{meta_config["model_name"]} Strategy')
    
    if not benchmark_df.empty:
        ax1.plot(benchmark_df['date'], benchmark_df['cumulative_return'], 
                 linewidth=2.5, color='orange', alpha=0.8, 
                 label=f'{meta_config["symbols"]} Buy&Hold')
    
    # 标记买卖操作
    buy_trades = trading_df_copy[trading_df_copy['action'] == 'BUY']
    sell_trades = trading_df_copy[trading_df_copy['action'] == 'SELL']
    
    if not buy_trades.empty:
        ax1.scatter(buy_trades['date'], buy_trades['portfolio_return'], 
                   color='green', s=100, marker='^', alpha=0.8, label='Buy', zorder=5)
    
    if not sell_trades.empty:
        ax1.scatter(sell_trades['date'], sell_trades['portfolio_return'], 
                   color='red', s=100, marker='v', alpha=0.8, label='Sell', zorder=5)
    
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_title(f'{meta_config["symbols"]} Cumulative Returns Comparison', 
                  fontsize=16, fontweight='bold')
    ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # 下图：放大显示策略的细微变化
    ax2.plot(trading_df_copy['date'], trading_df_copy['portfolio_return'], 
             marker='o', linewidth=3, markersize=6, color='#2E86AB', 
             label=f'{meta_config["model_name"]} Strategy (Zoomed)')
    
    if not buy_trades.empty:
        ax2.scatter(buy_trades['date'], buy_trades['portfolio_return'], 
                   color='green', s=120, marker='^', alpha=0.9, zorder=5)
    
    if not sell_trades.empty:
        ax2.scatter(sell_trades['date'], sell_trades['portfolio_return'], 
                   color='red', s=120, marker='v', alpha=0.9, zorder=5)
    
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_title('Model Strategy Returns (Detailed View)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Return (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # 设置y轴范围以突出显示策略的变化
    strategy_min = trading_df_copy['portfolio_return'].min()
    strategy_max = trading_df_copy['portfolio_return'].max()
    y_range = max(abs(strategy_min), abs(strategy_max)) * 1.2
    ax2.set_ylim(-y_range, y_range)
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{charts_path}/returns_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_corrected_portfolio_value_chart(trading_df: pd.DataFrame, benchmark_df: pd.DataFrame, charts_path: str, meta_config):
    """Generate corrected portfolio value vs benchmark price comparison chart"""
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # 左坐标轴: 投资组合价值（直接使用原始数据）
    color1 = '#2E86AB'
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Portfolio Value ($)', color=color1, fontsize=12, fontweight='bold')
    ax1.plot(trading_df['date'], trading_df['portfolio_value'], 
             marker='o', linewidth=2.5, markersize=4, color=color1, 
             label='Portfolio Value', alpha=0.8)
    
    # 投资组合初始价值参考线
    initial_value = 100000
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
        ax2.plot(benchmark_df['date'], normalized_prices,
                linewidth=2.5, color=color2, alpha=0.7,
                label=f'{meta_config["symbols"]} Price (Normalized)')
        
        # 右轴显示实际价格
        ax2_ticks = ax2.get_yticks()
        ax2.set_yticklabels([f'${tick/price_scale:.1f}' for tick in ax2_ticks])
        ax2.tick_params(axis='y', labelcolor=color2)
    
    # 标记交易信号
    buy_trades = trading_df[trading_df['action'] == 'BUY']
    sell_trades = trading_df[trading_df['action'] == 'SELL']
    
    if not buy_trades.empty:
        ax1.scatter(buy_trades['date'], buy_trades['portfolio_value'], 
                   color='green', s=100, marker='^', alpha=0.8, label='Buy Signal', 
                   zorder=5, edgecolor='darkgreen')
    
    if not sell_trades.empty:
        ax1.scatter(sell_trades['date'], sell_trades['portfolio_value'], 
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
    
    plt.xticks(rotation=45)
    fig.tight_layout()
    plt.savefig(f"{charts_path}/portfolio_value.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    print("🚀 生成最终修正的投资分析图表...")
    
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
    charts_path = 'results/250808_230347_Qwen_Qwen3-8B_JNJ/charts_final_corrected'
    os.makedirs(charts_path, exist_ok=True)
    
    # 打印数据统计
    initial_value = 100000
    portfolio_return = (trading_df['portfolio_value'] / initial_value - 1) * 100
    print(f"📊 策略收益率统计:")
    print(f"   最终收益率: {portfolio_return.iloc[-1]:.4f}%")
    print(f"   最大收益率: {portfolio_return.max():.4f}%")
    print(f"   最小收益率: {portfolio_return.min():.4f}%")
    print(f"   收益率标准差: {portfolio_return.std():.4f}%")
    print(f"   基准最终收益率: {benchmark_data['cumulative_return'].iloc[-1]:.2f}%")
    
    try:
        print("📈 生成修正的累计回报率对比图（双面板）...")
        generate_corrected_returns_comparison_chart(trading_df, benchmark_data, charts_path, meta_config)
        print("✅ returns_comparison.png 已生成")
        
        print("📊 生成修正的投资组合价值双轴图...")
        generate_corrected_portfolio_value_chart(trading_df, benchmark_data, charts_path, meta_config)
        print("✅ portfolio_value.png 已生成")
        
        print(f"\n🎯 所有修正后的图表已保存到: {charts_path}/")
        print("主要修正:")
        print("  1. 准确显示了模型策略的小幅收益波动（-0.05%范围内）")
        print("  2. 采用双面板设计：上图全景对比，下图放大显示策略细节")
        print("  3. 使用原始数据，确保计算准确性")
        print("  4. 专业的金融图表设计和配色")
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()