#!/usr/bin/env python3
# 测试修复后的图表生成

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
import sys
import json

# 导入必要的函数
sys.path.append('src')
exec(open('run.py').read())

def main():
    # 读取交易数据
    trading_df = pd.read_csv('results/250808_230347_Qwen_Qwen3-8B_JNJ/trading_results.csv')
    trading_df['date'] = pd.to_datetime(trading_df['date'])
    
    # 加载基准数据
    benchmark_data = load_benchmark_data('data/JNJ.json', pd.to_datetime('2020-03-12'), pd.to_datetime('2020-03-30'))
    
    # 创建元配置
    meta_config = {
        'model_name': 'Qwen_Qwen3-8B', 
        'symbols': 'JNJ'
    }
    
    # 创建输出文件夹
    charts_path = 'results/250808_230347_Qwen_Qwen3-8B_JNJ/charts_fixed'
    os.makedirs(charts_path, exist_ok=True)
    
    try:
        print("正在生成修复后的累计回报率对比图...")
        generate_returns_comparison_chart(trading_df, benchmark_data, charts_path, meta_config)
        print("✅ returns_comparison.png 已生成")
        
        print("正在生成修复后的投资组合价值图...")
        generate_portfolio_value_chart(trading_df, benchmark_data, charts_path, meta_config)
        print("✅ portfolio_value.png 已生成")
        
        print(f"\n所有图表已保存到: {charts_path}")
        
    except Exception as e:
        print(f"❌ 生成图表时出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()