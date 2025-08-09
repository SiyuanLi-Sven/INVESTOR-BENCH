#!/usr/bin/env python3
"""
修正评估框架，使其与原始框架完全一致
基于方向预测计算收益率，而不是基于实际交易数量
"""

import sys
import os
sys.path.append('src')
sys.path.append('references/INVESTOR-BENCH-main/src')

# 导入原始框架的评估函数
from references.INVESTOR_BENCH_main.src.eval_pipeline import (
    output_metrics_summary_single,
    input_data_restructure,
    reframe_data_files,
    metrics_summary
)

def fix_evaluation_to_match_original_framework():
    """修正评估，使用原始框架的逻辑"""
    
    print("🔧 修正评估框架，使其与原始INVESTOR-BENCH完全一致")
    print("=" * 60)
    
    # 使用原始框架的配置
    config = {
        "env_config": {
            "test_start_time": "2020-03-12",
            "test_end_time": "2020-03-30", 
            "trading_symbols": ["JNJ"]
        },
        "meta_config": {
            "result_save_path": "results/250808_230347_Qwen_Qwen3-8B_JNJ/test_output"
        }
    }
    
    # 数据路径
    data_path = "data/JNJ.json"
    output_path = "results/250808_230347_Qwen_Qwen3-8B_JNJ/metrics_original_framework"
    
    print(f"📊 使用原始框架的评估逻辑:")
    print(f"   - 基于方向预测 (-1, 0, 1)")
    print(f"   - 理论上100%仓位投资")
    print(f"   - 评估模型的方向预测能力")
    
    try:
        # 调用原始框架的评估函数
        output_metrics_summary_single(
            start_date=config["env_config"]["test_start_time"],
            end_date=config["env_config"]["test_end_time"],
            ticker=config["env_config"]["trading_symbols"][0],
            data_path=data_path,
            result_path=config["meta_config"]["result_save_path"],
            output_path=output_path
        )
        
        print("✅ 原始框架评估完成")
        print(f"📁 结果保存在: {output_path}")
        
    except Exception as e:
        print(f"❌ 评估过程中出错: {e}")
        import traceback
        traceback.print_exc()

def compare_evaluation_results():
    """对比原始框架评估结果与当前错误的结果"""
    
    print("\n🔍 对比评估结果:")
    print("=" * 40)
    
    print("❌ 当前错误的方法:")
    print("   - 基于实际交易数量 (1-2股)")
    print("   - 现金权重99.9%，股票权重0.1%") 
    print("   - 收益率: -0.026%")
    print("   - 与Buy&Hold差异: 6.09%")
    
    print("\n✅ 原始框架正确方法:")
    print("   - 基于方向预测 (-1, 0, 1)")
    print("   - 理论上100%仓位")
    print("   - 评估方向预测准确性")
    print("   - 应该得到合理的收益率对比")

if __name__ == "__main__":
    fix_evaluation_to_match_original_framework()
    compare_evaluation_results()