#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

def debug_chart_returns():
    """Debug why the returns chart shows flat line"""
    
    # 读取原始数据
    df = pd.read_csv('results/250808_230347_Qwen_Qwen3-8B_JNJ/trading_results.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("=== 原始数据 ===")
    print(df[['date', 'action', 'price', 'portfolio_value']].round(2))
    
    # 按照我之前修复的代码重新计算
    print("\n=== 我的修复代码重新计算的组合价值 ===")
    initial_value = 100000
    corrected_portfolio_values = []
    current_cash = initial_value
    current_position = 0
    
    for _, row in df.iterrows():
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
    
    df['my_corrected_portfolio_value'] = corrected_portfolio_values
    df['my_portfolio_return'] = (df['my_corrected_portfolio_value'] / initial_value - 1) * 100
    
    print(df[['date', 'portfolio_value', 'my_corrected_portfolio_value', 'my_portfolio_return']].round(4))
    
    # 直接用原始数据计算累计收益率
    print("\n=== 直接用原始portfolio_value计算累计收益率 ===")
    df['original_return'] = (df['portfolio_value'] / initial_value - 1) * 100
    
    print(df[['date', 'portfolio_value', 'original_return']].round(4))
    
    # 检查为什么图表显示为直线
    print(f"\n=== 收益率范围分析 ===")
    print(f"我计算的收益率范围: {df['my_portfolio_return'].min():.4f}% 到 {df['my_portfolio_return'].max():.4f}%")
    print(f"原始数据收益率范围: {df['original_return'].min():.4f}% 到 {df['original_return'].max():.4f}%")
    print(f"收益率标准差: {df['original_return'].std():.4f}%")
    
    # 生成测试图表看看问题在哪里
    plt.figure(figsize=(12, 6))
    plt.plot(df['date'], df['original_return'], 'o-', label='Original Return', linewidth=2)
    plt.plot(df['date'], df['my_portfolio_return'], 's--', label='My Calculated Return', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Cumulative Returns Comparison - Debug')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('debug_returns_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n调试图表已保存: debug_returns_comparison.png")
    
    return df

if __name__ == "__main__":
    result = debug_chart_returns()