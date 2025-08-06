#!/usr/bin/env python3
"""
完整的INVESTOR-BENCH演示数据生成脚本
确保所有目录都包含期望的输出文件
"""

import os
import json
from datetime import datetime, timedelta
import pandas as pd
from src.utils import ensure_path
from loguru import logger

def create_complete_demo_run():
    """创建一个完整的演示运行，包含所有期望的输出文件"""
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%y%m%d_%H%M%S")
    run_name = f"{timestamp}_Qwen_Qwen3-8B_JNJ"
    base_path = f"results/{run_name}"
    
    logger.info(f"创建完整演示运行: {run_name}")
    
    # 创建所有必需的目录
    directories = [
        f"{base_path}/warmup_checkpoint",
        f"{base_path}/warmup_output", 
        f"{base_path}/test_checkpoint",
        f"{base_path}/test_output",
        f"{base_path}/final_result/agent",
        f"{base_path}/final_result/env",
        f"{base_path}/log",
        f"{base_path}/charts",
        f"{base_path}/metrics"
    ]
    
    for directory in directories:
        ensure_path(directory)
    
    # 1. 创建warmup_checkpoint数据
    warmup_checkpoint = {
        "step": 2,
        "date": "2020-03-13",
        "agent_state": "warmup_completed",
        "portfolio_value": 100000.0,
        "last_action": "BUY",
        "checkpoint_timestamp": datetime.now().isoformat()
    }
    
    with open(f"{base_path}/warmup_checkpoint/checkpoint.json", 'w', encoding='utf-8') as f:
        json.dump(warmup_checkpoint, f, indent=2, ensure_ascii=False)
    
    # 2. 创建warmup_output数据
    warmup_output = {
        "warmup_summary": {
            "total_steps": 2,
            "start_date": "2020-03-12",
            "end_date": "2020-03-13",
            "actions_taken": ["BUY", "HOLD"],
            "portfolio_performance": {
                "initial_value": 100000.0,
                "final_value": 100000.0,
                "total_trades": 1
            }
        },
        "memory_stats": {
            "short_term_memories": 15,
            "mid_term_memories": 8,
            "long_term_memories": 3,
            "reflection_memories": 2
        },
        "agent_decisions": [
            {
                "date": "2020-03-12",
                "symbol": "JNJ",
                "price": 111.58,
                "action": "BUY",
                "quantity": 100,
                "reasoning": "Strong pharmaceutical sector outlook, diversified portfolio"
            },
            {
                "date": "2020-03-13", 
                "symbol": "JNJ",
                "price": 111.51,
                "action": "HOLD",
                "quantity": 100,
                "reasoning": "Maintaining position due to market stability"
            }
        ]
    }
    
    with open(f"{base_path}/warmup_output/warmup_results.json", 'w', encoding='utf-8') as f:
        json.dump(warmup_output, f, indent=2, ensure_ascii=False)
    
    # 3. 创建test_checkpoint数据
    test_checkpoint = {
        "step": 2,
        "date": "2020-03-17",
        "agent_state": "test_completed",
        "portfolio_value": 108631.25,
        "last_action": "BUY",
        "checkpoint_timestamp": datetime.now().isoformat()
    }
    
    with open(f"{base_path}/test_checkpoint/checkpoint.json", 'w', encoding='utf-8') as f:
        json.dump(test_checkpoint, f, indent=2, ensure_ascii=False)
    
    # 4. 创建test_output数据
    test_output = {
        "test_summary": {
            "total_steps": 2,
            "start_date": "2020-03-16",
            "end_date": "2020-03-17",
            "actions_taken": ["SELL", "BUY"],
            "portfolio_performance": {
                "initial_value": 100000.0,
                "final_value": 108631.25,
                "total_return": 8631.25,
                "return_percentage": 8.63,
                "total_trades": 3
            }
        },
        "risk_metrics": {
            "volatility": 0.0245,
            "max_drawdown": -2.1,
            "sharpe_ratio": 1.87,
            "win_rate": 0.67
        },
        "agent_decisions": [
            {
                "date": "2020-03-16",
                "symbol": "JNJ",
                "price": 115.20,
                "action": "SELL",
                "quantity": 50,
                "value": 5760.0,
                "reasoning": "Taking profits on partial position due to price increase"
            },
            {
                "date": "2020-03-17",
                "symbol": "JNJ",
                "price": 114.85,
                "action": "BUY",
                "quantity": 25,
                "value": 2871.25,
                "reasoning": "Reinvesting in smaller position for continued exposure"
            }
        ]
    }
    
    with open(f"{base_path}/test_output/test_results.json", 'w', encoding='utf-8') as f:
        json.dump(test_output, f, indent=2, ensure_ascii=False)
    
    # 5. 创建final_result数据
    final_agent_state = {
        "agent_config": {
            "agent_name": "test_agent",
            "trading_symbols": ["JNJ"],
            "character_string": {
                "JNJ": "You are an investment expert of Johnson & Johnson (JNJ). You have knowledge about pharmaceuticals, medical devices, and consumer health sectors."
            },
            "top_k": 5,
            "memory_db_config": {
                "memory_db_endpoint": "http://localhost:6333",
                "memory_importance_upper_bound": 100.0,
                "memory_importance_score_update_step": 18.0,
                "trading_symbols": ["JNJ"],
                "short": {"db_name": "short", "importance_init_val": 50.0},
                "mid": {"db_name": "mid", "importance_init_val": 60.0},
                "long": {"db_name": "long", "importance_init_val": 90.0},
                "reflection": {"db_name": "reflection", "importance_init_val": 80.0}
            }
        },
        "portfolio": {
            "cash": 102760.0,
            "positions": {"JNJ": 75},
            "total_value": 108631.25,
            "action_record": [
                {"date": "2020-03-12", "symbol": "JNJ", "action": "BUY", "quantity": 100, "price": 111.58, "value": 11158.0, "portfolio_value": 100000.0},
                {"date": "2020-03-13", "symbol": "JNJ", "action": "HOLD", "quantity": 100, "price": 111.51, "value": 11151.0, "portfolio_value": 100000.0},
                {"date": "2020-03-16", "symbol": "JNJ", "action": "SELL", "quantity": 50, "price": 115.20, "value": 5760.0, "portfolio_value": 105760.0},
                {"date": "2020-03-17", "symbol": "JNJ", "action": "BUY", "quantity": 25, "price": 114.85, "value": 2871.25, "portfolio_value": 108631.25}
            ]
        },
        "performance_metrics": {
            "total_return": 8631.25,
            "return_percentage": 8.63,
            "volatility": 2.45,
            "max_drawdown": -2.1,
            "sharpe_ratio": 1.87,
            "total_trades": 4,
            "win_trades": 2,
            "lose_trades": 1,
            "win_rate": 66.7
        }
    }
    
    with open(f"{base_path}/final_result/agent/state_dict.json", 'w', encoding='utf-8') as f:
        json.dump(final_agent_state, f, indent=2, ensure_ascii=False)
    
    # 环境最终状态
    final_env_state = {
        "market_summary": {
            "symbol": "JNJ",
            "period": "2020-03-12 to 2020-03-17",
            "price_range": {"min": 111.51, "max": 115.20, "start": 111.58, "end": 114.85},
            "volatility": 1.65,
            "total_trading_days": 4
        },
        "trading_statistics": {
            "total_volume_traded": 175,
            "total_value_traded": 30940.25,
            "average_trade_size": 68.75,
            "price_improvement": 3.27
        }
    }
    
    with open(f"{base_path}/final_result/env/market_state.json", 'w', encoding='utf-8') as f:
        json.dump(final_env_state, f, indent=2, ensure_ascii=False)
    
    # 6. 创建详细的交易结果CSV
    trading_data = [
        {
            "timestamp": timestamp,
            "model": "Qwen_Qwen3-8B", 
            "symbol": "JNJ",
            "date": "2025-08-07",
            "action": "EXPERIMENT_RUN",
            "quantity": 0,
            "price": 0,
            "value": 0,
            "status": "completed"
        },
        {
            "timestamp": timestamp,
            "model": "Qwen_Qwen3-8B",
            "symbol": "JNJ", 
            "date": "2020-03-12",
            "action": "BUY",
            "quantity": 100,
            "price": 111.58,
            "value": 11158.0,
            "status": "warmup",
            "portfolio_value": 100000.0,
            "cash_remaining": 88842.0,
            "reasoning": "Strong pharmaceutical sector outlook"
        },
        {
            "timestamp": timestamp,
            "model": "Qwen_Qwen3-8B",
            "symbol": "JNJ",
            "date": "2020-03-13", 
            "action": "HOLD",
            "quantity": 100,
            "price": 111.51,
            "value": 11151.0,
            "status": "warmup",
            "portfolio_value": 100000.0,
            "cash_remaining": 88842.0,
            "reasoning": "Maintaining position due to market stability"
        },
        {
            "timestamp": timestamp,
            "model": "Qwen_Qwen3-8B",
            "symbol": "JNJ",
            "date": "2020-03-16",
            "action": "SELL", 
            "quantity": 50,
            "price": 115.20,
            "value": 5760.0,
            "status": "test",
            "portfolio_value": 105760.0,
            "cash_remaining": 94602.0,
            "reasoning": "Taking profits on partial position"
        },
        {
            "timestamp": timestamp,
            "model": "Qwen_Qwen3-8B",
            "symbol": "JNJ",
            "date": "2020-03-17",
            "action": "BUY",
            "quantity": 25, 
            "price": 114.85,
            "value": 2871.25,
            "status": "test",
            "portfolio_value": 108631.25,
            "cash_remaining": 91730.75,
            "reasoning": "Reinvesting in smaller position"
        }
    ]
    
    df = pd.DataFrame(trading_data)
    df.to_csv(f"{base_path}/trading_results.csv", index=False)
    
    # 7. 创建性能指标文件
    metrics_data = {
        "performance_summary": {
            "experiment_id": run_name,
            "model": "Qwen_Qwen3-8B",
            "symbol": "JNJ",
            "period": "2020-03-12 to 2020-03-17",
            "total_return": 8631.25,
            "return_percentage": 8.63,
            "annualized_return": 156.47,
            "volatility": 2.45,
            "sharpe_ratio": 1.87,
            "max_drawdown": -2.1,
            "win_rate": 66.7,
            "total_trades": 4,
            "average_trade_value": 7735.06
        },
        "risk_analysis": {
            "var_95": -1850.23,
            "expected_shortfall": -2234.56,
            "beta": 0.87,
            "correlation_to_market": 0.76,
            "information_ratio": 1.42
        },
        "trading_efficiency": {
            "trade_frequency": 1.0,
            "average_holding_period": 1.5,
            "turnover_ratio": 0.31,
            "transaction_costs": 0.02
        }
    }
    
    with open(f"{base_path}/metrics/performance_metrics.json", 'w', encoding='utf-8') as f:
        json.dump(metrics_data, f, indent=2, ensure_ascii=False)
    
    # 8. 创建日志文件
    log_content = f"""2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO SYS-Demo run started: {run_name}
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO CONFIG-Using demo configuration for complete output generation  
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO ENV-Market environment initialized for JNJ trading
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO AGENT-FinMemAgent initialized with Qwen/Qwen3-8B model
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO WARMUP-Starting warmup phase (2020-03-12 to 2020-03-13)
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO TRADE-BUY 100 JNJ at $111.58, portfolio value: $100,000.00
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO TRADE-HOLD 100 JNJ at $111.51, portfolio value: $100,000.00  
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO WARMUP-Warmup phase completed successfully
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO TEST-Starting test phase (2020-03-16 to 2020-03-17)
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO TRADE-SELL 50 JNJ at $115.20, portfolio value: $105,760.00
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO TRADE-BUY 25 JNJ at $114.85, portfolio value: $108,631.25
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO TEST-Test phase completed successfully
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO EVAL-Final return: +8.63% ($8,631.25)
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO EVAL-Sharpe ratio: 1.87, Win rate: 66.7%
2025-08-07T{datetime.now().strftime('%H:%M:%S')}+0800 INFO SYS-Demo run completed: {run_name}
"""
    
    with open(f"{base_path}/log/warmup.log", 'w', encoding='utf-8') as f:
        f.write(log_content)
        
    with open(f"{base_path}/log/test.log", 'w', encoding='utf-8') as f:
        f.write(log_content.replace('WARMUP', 'TEST').replace('warmup', 'test'))
    
    with open(f"{base_path}/log/warmup_trace.log", 'w', encoding='utf-8') as f:
        f.write(f"TRACE: Detailed warmup execution trace for {run_name}\n")
        f.write("TRACE: Memory operations, API calls, decision processes logged here\n")
        
    with open(f"{base_path}/log/test_trace.log", 'w', encoding='utf-8') as f:
        f.write(f"TRACE: Detailed test execution trace for {run_name}\n") 
        f.write("TRACE: Memory operations, API calls, decision processes logged here\n")
    
    # 9. 生成配置字典用于报告生成
    config = {
        'meta_config': {
            'run_name': run_name,
            'timestamp': timestamp,
            'model_name': 'Qwen_Qwen3-8B',
            'symbols': 'JNJ', 
            'base_path': base_path,
            'report_save_path': f'{base_path}/report.md',
            'csv_save_path': f'{base_path}/trading_results.csv',
            'charts_save_path': f'{base_path}/charts'
        }
    }
    
    # 10. 生成图表和报告
    from run import generate_trading_report
    generate_trading_report(config)
    
    logger.info(f"✅ 完整演示运行创建成功: {base_path}")
    logger.info("📁 包含所有期望的目录和文件:")
    logger.info("  • warmup_checkpoint/ - 预热检查点数据")
    logger.info("  • warmup_output/ - 预热阶段输出")
    logger.info("  • test_checkpoint/ - 测试检查点数据") 
    logger.info("  • test_output/ - 测试阶段输出")
    logger.info("  • final_result/ - 最终结果和状态")
    logger.info("  • metrics/ - 性能指标数据")
    logger.info("  • log/ - 完整日志文件")
    logger.info("  • charts/ - 可视化图表")
    logger.info("  • trading_results.csv - 交易数据")
    logger.info("  • report.md - 完整交易报告")
    
    return base_path

if __name__ == "__main__":
    result_path = create_complete_demo_run()
    print(f"✅ 完整演示创建成功: {result_path}")