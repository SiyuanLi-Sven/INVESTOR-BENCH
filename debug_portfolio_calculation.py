#!/usr/bin/env python3
import pandas as pd
import numpy as np

def debug_portfolio_calculation():
    """Debug portfolio value and returns calculation"""
    
    # 读取原始数据
    df = pd.read_csv('results/250808_230347_Qwen_Qwen3-8B_JNJ/trading_results.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("=== 原始数据分析 ===")
    print(df[['date', 'action', 'quantity', 'price', 'portfolio_value', 'current_position', 'cash_remaining']])
    
    print("\n=== 重新计算正确的组合价值 ===")
    initial_value = 100000
    current_cash = initial_value
    current_position = 0
    
    corrected_data = []
    
    for i, row in df.iterrows():
        print(f"\n日期: {row['date'].strftime('%Y-%m-%d')}")
        print(f"操作: {row['action']}, 数量: {row['quantity']}, 价格: {row['price']:.2f}")
        print(f"操作前 - 现金: {current_cash:.2f}, 持仓: {current_position}")
        
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
        # HOLD时不变
        
        # 计算正确的组合价值
        position_value = current_position * row['price']
        portfolio_value = current_cash + position_value
        
        print(f"操作后 - 现金: {current_cash:.2f}, 持仓: {current_position}, 持仓市值: {position_value:.2f}")
        print(f"组合总价值: {portfolio_value:.2f}")
        print(f"原数据组合价值: {row['portfolio_value']:.2f}")
        print(f"差异: {portfolio_value - row['portfolio_value']:.2f}")
        
        corrected_data.append({
            'date': row['date'],
            'action': row['action'],
            'price': row['price'],
            'current_cash': current_cash,
            'current_position': current_position,
            'position_value': position_value,
            'portfolio_value': portfolio_value,
            'original_portfolio_value': row['portfolio_value']
        })
    
    corrected_df = pd.DataFrame(corrected_data)
    
    print("\n=== 计算日收益率和累计收益率 ===")
    # 计算日收益率 (基于组合价值变化)
    corrected_df['daily_return'] = corrected_df['portfolio_value'].pct_change().fillna(0)
    
    # 计算累计收益率 (从初始价值开始)
    corrected_df['cumulative_return'] = (corrected_df['portfolio_value'] / initial_value - 1) * 100
    
    # 也可以通过日收益率累乘计算
    corrected_df['cumulative_return_alt'] = ((1 + corrected_df['daily_return']).cumprod() - 1) * 100
    
    print(corrected_df[['date', 'daily_return', 'cumulative_return', 'cumulative_return_alt']].round(4))
    
    return corrected_df

if __name__ == "__main__":
    result = debug_portfolio_calculation()