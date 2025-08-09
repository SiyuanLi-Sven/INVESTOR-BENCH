#!/usr/bin/env python3
import pandas as pd
import numpy as np

def analyze_single_asset_portfolio():
    """深度分析单资产组合收益率计算"""
    
    # 读取交易数据
    df = pd.read_csv('results/250808_230347_Qwen_Qwen3-8B_JNJ/trading_results.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("=== 单资产组合深度分析 ===")
    print(f"初始资金: $100,000")
    print(f"资产: JNJ")
    print(f"交易期间: {df['date'].min()} 到 {df['date'].max()}")
    
    print("\n=== 详细交易分析 ===")
    current_cash = 100000
    current_position = 0
    
    analysis_data = []
    
    for i, row in df.iterrows():
        print(f"\n第{i+1}天 - {row['date'].strftime('%Y-%m-%d')}")
        print(f"股价: ${row['price']:.2f}")
        print(f"操作: {row['action']}")
        
        if row['action'] == 'BUY':
            shares_bought = row['quantity']
            cost = shares_bought * row['price']
            current_cash -= cost
            current_position += shares_bought
            print(f"买入 {shares_bought} 股，成本 ${cost:.2f}")
        elif row['action'] == 'SELL':
            shares_sold = row['quantity']
            proceeds = shares_sold * row['price']
            current_cash += proceeds
            current_position -= shares_sold
            print(f"卖出 {shares_sold} 股，收入 ${proceeds:.2f}")
        
        # 计算组合价值
        position_value = current_position * row['price']
        portfolio_value = current_cash + position_value
        
        # 计算权重
        cash_weight = current_cash / portfolio_value if portfolio_value != 0 else 0
        stock_weight = position_value / portfolio_value if portfolio_value != 0 else 0
        
        print(f"持仓: {current_position} 股")
        print(f"现金: ${current_cash:.2f}")
        print(f"持仓市值: ${position_value:.2f}")
        print(f"组合总价值: ${portfolio_value:.2f}")
        print(f"现金权重: {cash_weight:.1%}, 股票权重: {stock_weight:.1%}")
        print(f"原数据组合价值: ${row['portfolio_value']:.2f}")
        
        analysis_data.append({
            'date': row['date'],
            'price': row['price'],
            'action': row['action'],
            'position': current_position,
            'cash': current_cash,
            'position_value': position_value,
            'portfolio_value': portfolio_value,
            'cash_weight': cash_weight,
            'stock_weight': stock_weight,
            'original_portfolio_value': row['portfolio_value']
        })
    
    analysis_df = pd.DataFrame(analysis_data)
    
    print("\n=== 关键问题分析 ===")
    
    # 1. 检查是否完全持有股票
    final_position = analysis_df['position'].iloc[-1]
    final_cash = analysis_df['cash'].iloc[-1]
    final_price = analysis_df['price'].iloc[-1]
    
    print(f"最终持仓: {final_position} 股")
    print(f"最终现金: ${final_cash:.2f}")
    print(f"如果完全投资股票应有股数: {100000 / analysis_df['price'].iloc[0]:.2f}")
    
    # 2. 计算如果一直持有股票的收益率
    initial_price = analysis_df['price'].iloc[0]
    final_price = analysis_df['price'].iloc[-1]
    buy_hold_return = (final_price / initial_price - 1) * 100
    
    print(f"\n=== Buy&Hold基准对比 ===")
    print(f"初始股价: ${initial_price:.2f}")
    print(f"最终股价: ${final_price:.2f}")
    print(f"Buy&Hold收益率: {buy_hold_return:.2f}%")
    
    # 3. 分析实际组合收益率
    actual_portfolio_return = (analysis_df['portfolio_value'].iloc[-1] / 100000 - 1) * 100
    print(f"实际组合收益率: {actual_portfolio_return:.4f}%")
    
    # 4. 分析收益率差异的原因
    print(f"\n=== 收益率差异分析 ===")
    print(f"Buy&Hold vs 实际组合收益率差异: {buy_hold_return - actual_portfolio_return:.4f}%")
    
    # 5. 检查交易成本和现金拖累
    print(f"\n=== 现金拖累分析 ===")
    avg_cash_weight = analysis_df['cash_weight'].mean()
    avg_stock_weight = analysis_df['stock_weight'].mean()
    print(f"平均现金权重: {avg_cash_weight:.1%}")
    print(f"平均股票权重: {avg_stock_weight:.1%}")
    
    # 6. 详细的权重变化
    print(f"\n=== 权重变化详情 ===")
    print(analysis_df[['date', 'stock_weight', 'cash_weight', 'action']].round(4))
    
    return analysis_df

if __name__ == "__main__":
    result = analyze_single_asset_portfolio()