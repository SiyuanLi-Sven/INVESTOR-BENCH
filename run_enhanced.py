#!/usr/bin/env python3
"""
INVESTOR-BENCH 增强版 - 结合完整记忆系统和现代化输出风格
支持真正的warmup→test状态传递，同时提供用户友好的输出和API调用方式
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
import argparse

import orjson
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from rich import progress

# 导入现有的完整系统
from src import (
    FinMemAgent,
    MarketEnv,
    RunMode,
    TaskType,
    ensure_path,
    output_metric_summary_multi,
    output_metrics_summary_single,
)

# 加载环境变量
load_dotenv()

class EnhancedInvestorBench:
    """增强版INVESTOR-BENCH，结合完整记忆系统和现代化输出"""
    
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now()
        
        # 创建带时间戳的结果目录
        timestamp = self.start_time.strftime("%y%m%d_%H%M%S")
        model_name = self._clean_model_name(args.model)
        self.run_dir = Path("results") / f"{timestamp}_{model_name}_{args.symbol}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        log_file = self.run_dir / "run.log"
        logger.add(log_file, level="TRACE", format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}")
        
        # 初始化OpenAI客户端
        self.client = self._init_openai_client()
        
        # API统计
        self.api_stats = {
            "chat_calls": 0,
            "embedding_calls": 0,
            "total_tokens": 0,
            "errors": 0
        }
        
        logger.info(f"🚀 INVESTOR-BENCH增强版启动 - 运行目录: {self.run_dir}")
        
    def _clean_model_name(self, model: str) -> str:
        """清理模型名称用于文件路径"""
        return model.replace("/", "_").replace(":", "_").replace(" ", "_")
    
    def _init_openai_client(self) -> OpenAI:
        """初始化OpenAI客户端"""
        api_key = self.args.api_key or os.getenv('OPENAI_API_KEY')
        api_base = self.args.api_base or os.getenv('OPENAI_API_BASE', 'https://api.siliconflow.cn/v1')
        
        if not api_key:
            raise ValueError("❌ API密钥未设置！请通过--api-key参数或OPENAI_API_KEY环境变量提供")
        
        logger.info(f"🔧 OpenAI客户端配置:")
        logger.info(f"   模型: {self.args.model}")
        logger.info(f"   API Base: {api_base}")
        logger.info(f"   API Key: {api_key[:10]}...")
        
        return OpenAI(api_key=api_key, base_url=api_base)
    
    def _find_latest_warmup_result(self) -> str:
        """查找最近的warmup结果目录"""
        results_dir = Path("results")
        if not results_dir.exists():
            raise FileNotFoundError("❌ 找不到results目录")
        
        # 查找所有包含当前股票的结果目录，但排除当前运行目录
        pattern = f"*_{self._clean_model_name(self.args.model)}_{self.args.symbol}"
        matching_dirs = [d for d in results_dir.glob(pattern) if d.name != self.run_dir.name]
        
        if not matching_dirs:
            raise FileNotFoundError(f"❌ 找不到{self.args.symbol}的warmup结果")
        
        # 查找包含warmup_output的目录，按时间戳排序
        warmup_dirs = []
        for dir_path in matching_dirs:
            warmup_agent_path = dir_path / "warmup_output" / "agent"
            if warmup_agent_path.exists():
                warmup_dirs.append(dir_path)
        
        if not warmup_dirs:
            raise FileNotFoundError(f"❌ 找不到{self.args.symbol}的有效warmup结果")
        
        # 按时间戳排序，取最新的
        latest_dir = max(warmup_dirs, key=lambda x: x.name[:13])
        warmup_agent_path = latest_dir / "warmup_output" / "agent"
        
        return str(warmup_agent_path)
    
    def load_config(self) -> Dict[str, Any]:
        """根据命令行参数动态生成配置"""
        
        # 基础配置模板
        config = {
            "chat_config": {
                "chat_model": self.args.model,
                "lora": False,
                "chat_model_type": "chat",
                "chat_model_inference_engine": "openai",
                "chat_system_message": "You are a helpful assistant.",
                "chat_endpoint": f"{self.client.base_url}/chat/completions",
                "chat_parameters": {
                    "temperature": 0.6,
                    "max_tokens": 1024
                },
                "chat_request_sleep": {
                    "sleep_time": 2,
                    "sleep_every_count": 10
                },
                "chat_max_new_token": 1000,
                "chat_request_timeout": 300,  # 设置为5分钟，而不是1000秒
                "chat_vllm_endpoint": "http://0.0.0.0:8000"
            },
            "emb_config": {
                "emb_model_name": "Qwen/Qwen3-Embedding-4B",
                "request_endpoint": f"{self.client.base_url}/embeddings",
                "emb_size": 2560,
                "embedding_timeout": 120  # 设置为2分钟，而不是600秒
            },
            "env_config": {
                "trading_symbols": [self.args.symbol],
                "warmup_start_time": self.args.start_date,
                "warmup_end_time": self.args.end_date,
                "test_start_time": self.args.test_start_date or "2020-10-01",
                "test_end_time": self.args.test_end_date or "2021-05-06",
                "momentum_window_size": 3,
                "env_data_path": {
                    self.args.symbol: f"data/{self.args.symbol.lower()}.json"
                }
            },
            "portfolio_config": {
                "trading_symbols": [self.args.symbol],
                "type": "single-asset",
                "look_back_window_size": 3
            },
            "agent_config": {
                "agent_name": "agent",
                "trading_symbols": [self.args.symbol],
                "character_string": {
                    self.args.symbol: f"You are an investment expert specializing in {self.args.symbol}."
                },
                "top_k": 5,
                "memory_db_config": {
                    "memory_db_endpoint": "http://localhost:6333",
                    "memory_importance_upper_bound": 100.0,
                    "memory_importance_score_update_step": 18.0,
                    "trading_symbols": [self.args.symbol],
                    "short": {
                        "db_name": "short",
                        "importance_init_val": 50.0,
                        "decay_recency_factor": 3.0,
                        "decay_importance_factor": 0.92,
                        "clean_up_recency_threshold": 0.05,
                        "clean_up_importance_threshold": 5.0,
                        "jump_upper_threshold": 55.0
                    },
                    "mid": {
                        "db_name": "mid", 
                        "importance_init_val": 60.0,
                        "decay_recency_factor": 90.0,
                        "decay_importance_factor": 0.96,
                        "clean_up_recency_threshold": 0.05,
                        "clean_up_importance_threshold": 5.0,
                        "jump_lower_threshold": 55.0,
                        "jump_upper_threshold": 85.0
                    },
                    "long": {
                        "db_name": "long",
                        "importance_init_val": 90.0,
                        "decay_recency_factor": 365.0,
                        "decay_importance_factor": 0.96,
                        "clean_up_recency_threshold": 0.05,
                        "clean_up_importance_threshold": 5.0,
                        "jump_lower_threshold": 85.0
                    },
                    "reflection": {
                        "db_name": "reflection",
                        "importance_init_val": 80.0,
                        "decay_recency_factor": 365.0,
                        "decay_importance_factor": 0.98,
                        "clean_up_recency_threshold": 0.05,
                        "clean_up_importance_threshold": 5.0,
                        "similarity_threshold": 0.95
                    }
                }
            },
            "meta_config": {
                "run_name": "enhanced_exp",
                "momentum_window_size": 3,
                "warmup_checkpoint_save_path": str(self.run_dir / "warmup_checkpoint"),
                "warmup_output_save_path": str(self.run_dir / "warmup_output"),
                "test_checkpoint_save_path": str(self.run_dir / "test_checkpoint"),
                "test_output_save_path": str(self.run_dir / "test_output"),
                "result_save_path": str(self.run_dir / "final_result"),
                "log_save_path": str(self.run_dir / "log")
            }
        }
        
        return config
    
    def run_warmup(self) -> bool:
        """运行warmup阶段，采用新版本的输出风格"""
        logger.info(f"🎯 开始Warmup阶段 - 学习期")
        
        config = self.load_config()
        
        # 确保路径存在
        for path_key in ["warmup_checkpoint_save_path", "warmup_output_save_path", "log_save_path"]:
            ensure_path(save_path=config["meta_config"][path_key])
        
        # 初始化环境和代理
        env = MarketEnv(
            env_data_path=config["env_config"]["env_data_path"],
            start_date=config["env_config"]["warmup_start_time"],
            end_date=config["env_config"]["warmup_end_time"],
            symbol=config["env_config"]["trading_symbols"],
            momentum_window_size=config["env_config"]["momentum_window_size"]
        )
        
        # 确定任务类型
        if len(config["env_config"]["trading_symbols"]) > 1:
            task_type = TaskType.MultiAssets
        else:
            task_type = TaskType.SingleAsset
            
        agent = FinMemAgent(
            agent_config=config["agent_config"],
            chat_config=config["chat_config"],
            emb_config=config["emb_config"],
            portfolio_config=config["portfolio_config"],
            task_type=task_type
        )
        
        # 运行warmup循环
        total_steps = env.simulation_length
        logger.info(f"📊 Warmup数据范围: {total_steps}天, {config['env_config']['warmup_start_time']} 至 {config['env_config']['warmup_end_time']}")
        
        step_count = 0
        with progress.Progress() as progress_bar:
            task_id = progress_bar.add_task("🎓 Warmup学习中", total=total_steps)
            
            while True:
                # 获取市场观察
                obs = env.step()
                if obs.termination_flag:
                    break
                
                step_count += 1
                logger.info(f"📅 处理第{step_count}/{total_steps}天: {obs.cur_date}")
                
                # 代理学习（Warmup模式不返回交易决策，只学习）
                logger.info(f"🧠 分析 {self.args.symbol} - {obs.cur_date}")
                agent.step(market_info=obs, run_mode=RunMode.WARMUP, task_type=task_type)
                
                # 获取当前价格
                current_price = obs.cur_price.get(self.args.symbol, 0) if obs.cur_price else 0
                
                # 现代化输出风格 - Warmup阶段只学习不交易
                logger.info(f"🎓 Warmup学习: 正在学习市场信息和新闻")
                logger.info(f"📝 当前价格: ${current_price:.2f}")
                logger.info(f"💡 学习进度: {step_count}/{total_steps} ({step_count/total_steps*100:.1f}%)")
                logger.info("-" * 60)
                
                progress_bar.update(task_id, advance=1)
        
        # 保存checkpoint
        logger.info("💾 保存Warmup结果...")
        agent.save_checkpoint(
            path=os.path.join(config["meta_config"]["warmup_checkpoint_save_path"], "agent")
        )
        env.save_checkpoint(
            path=os.path.join(config["meta_config"]["warmup_checkpoint_save_path"], "env")
        )
        
        # 保存输出
        agent.save_checkpoint(
            path=os.path.join(config["meta_config"]["warmup_output_save_path"], "agent")
        )
        
        logger.info("✅ Warmup阶段完成！")
        logger.info(f"📁 Warmup结果保存在: {config['meta_config']['warmup_output_save_path']}")
        
        return True
    
    def run_test(self, warmup_result_path: str = None) -> bool:
        """运行test阶段，从warmup结果加载状态"""
        logger.info(f"🎯 开始Test阶段 - 实战期")
        
        config = self.load_config()
        
        # 如果指定了warmup路径，使用指定路径；否则查找最近的warmup结果
        if warmup_result_path:
            warmup_agent_path = os.path.join(warmup_result_path, "warmup_output", "agent")
        else:
            # 查找最近的warmup结果
            warmup_agent_path = self._find_latest_warmup_result()
            
        if not os.path.exists(warmup_agent_path):
            raise FileNotFoundError(f"❌ 找不到Warmup结果！请先运行warmup阶段。路径: {warmup_agent_path}")
        
        logger.info(f"📂 从Warmup结果加载AI状态: {warmup_agent_path}")
        
        # 确保路径存在
        for path_key in ["test_checkpoint_save_path", "test_output_save_path", "result_save_path"]:
            ensure_path(save_path=config["meta_config"][path_key])
        
        # 加载环境和代理
        env = MarketEnv(
            env_data_path=config["env_config"]["env_data_path"],
            start_date=config["env_config"]["test_start_time"],
            end_date=config["env_config"]["test_end_time"],
            symbol=config["env_config"]["trading_symbols"],
            momentum_window_size=config["env_config"]["momentum_window_size"]
        )
        
        # 关键：从warmup输出加载代理状态
        agent = FinMemAgent.load_checkpoint(
            path=warmup_agent_path,
            portfolio_load_for_test=True,
        )
        
        # 运行test循环
        total_steps = env.simulation_length
        logger.info(f"📊 Test数据范围: {total_steps}天, {config['env_config']['test_start_time']} 至 {config['env_config']['test_end_time']}")
        
        step_count = 0
        with progress.Progress() as progress_bar:
            task_id = progress_bar.add_task("🎯 Test回测中", total=total_steps)
            
            while True:
                # 获取市场观察
                obs = env.step()
                if obs.termination_flag:
                    break
                
                step_count += 1
                logger.info(f"📅 处理第{step_count}/{total_steps}天: {obs.cur_date}")
                
                # 代理决策（基于warmup学到的知识）
                logger.info(f"🧠 分析 {self.args.symbol} - {obs.cur_date}")
                # 确定任务类型
                if len(config["env_config"]["trading_symbols"]) > 1:
                    task_type = TaskType.MultiAssets
                else:
                    task_type = TaskType.SingleAsset
                action = agent.step(obs, RunMode.TEST, task_type)
                
                # 获取当前价格
                current_price = obs.cur_price.get(self.args.symbol, 0) if obs.cur_price else 0
                
                if action and hasattr(action, 'action'):
                    # 现代化输出风格
                    decision_map = {0: "HOLD", 1: "BUY", 2: "SELL"}  
                    decision_str = decision_map.get(action.action, "UNKNOWN")
                    logger.info(f"🎯 Test决策: {decision_str} (基于Warmup学习)")
                else:
                    logger.info(f"🎯 Test决策: 正在处理中... (基于Warmup学习)")
                
                logger.info(f"📝 当前价格: ${current_price:.2f}")
                logger.info(f"💼 投资组合价值: (待实现)")
                logger.info(f"📈 测试进度: {step_count}/{total_steps} ({step_count/total_steps*100:.1f}%)")
                logger.info("-" * 60)
                
                progress_bar.update(task_id, advance=1)
        
        # 保存结果
        logger.info("💾 保存Test结果...")
        agent.save_checkpoint(
            path=os.path.join(config["meta_config"]["test_checkpoint_save_path"], "agent")
        )
        env.save_checkpoint(
            path=os.path.join(config["meta_config"]["test_checkpoint_save_path"], "env")
        )
        
        logger.info("✅ Test阶段完成！")
        logger.info(f"📁 Test结果保存在: {config['meta_config']['test_output_save_path']}")
        
        return True
    
    def generate_report(self) -> None:
        """生成最终报告"""
        logger.info("📊 生成投资分析报告...")
        
        config = self.load_config()
        
        # 生成投资绩效报告（需要有test阶段的结果）
        if self.args.mode in ["test", "both"]:
            try:
                # 确定任务类型
                if len(config["env_config"]["trading_symbols"]) == 1:
                    logger.info(f"📈 生成{self.args.symbol}的投资绩效报告...")
                    output_metrics_summary_single(
                        start_date=config["env_config"]["test_start_time"],
                        end_date=config["env_config"]["test_end_time"],
                        ticker=config["env_config"]["trading_symbols"][0],
                        data_path=list(config["env_config"]["env_data_path"].values())[0],
                        result_path=config["meta_config"]["result_save_path"],
                        output_path=str(self.run_dir / "metrics"),
                    )
                    logger.info(f"✅ 投资绩效报告已生成: {self.run_dir / 'metrics'}")
                else:
                    logger.info(f"📈 生成多资产投资绩效报告...")
                    output_metric_summary_multi(
                        trading_symbols=config["env_config"]["trading_symbols"],
                        data_root_path=config["env_config"]["env_data_path"],
                        output_path=str(self.run_dir / "metrics"),
                    )
                    logger.info(f"✅ 多资产投资绩效报告已生成: {self.run_dir / 'metrics'}")
            except Exception as e:
                logger.warning(f"⚠️ 投资报告生成失败: {str(e)}")
        
        # 保存元数据
        metadata = {
            "run_info": {
                "timestamp": self.start_time.isoformat(),
                "command_line": " ".join(sys.argv),
                "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
                "version": "enhanced_v1.0"
            },
            "config": {
                "symbol": self.args.symbol,
                "model": self.args.model,
                "mode": self.args.mode,
                "start_date": self.args.start_date,
                "end_date": self.args.end_date
            },
            "api_usage": self.api_stats
        }
        
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            import json
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 元数据已保存: {metadata_file}")

def main():
    parser = argparse.ArgumentParser(description="INVESTOR-BENCH 增强版 - 真正的Warmup→Test流程")
    
    # 必需参数
    parser.add_argument("mode", choices=["warmup", "test", "both"], help="运行模式")
    parser.add_argument("--symbol", required=True, help="股票代码")
    parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="结束日期 (YYYY-MM-DD)")
    
    # 可选参数
    parser.add_argument("--test-start-date", help="Test阶段开始日期 (YYYY-MM-DD)")
    parser.add_argument("--test-end-date", help="Test阶段结束日期 (YYYY-MM-DD)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="LLM模型名称")
    parser.add_argument("--api-key", help="OpenAI API密钥")
    parser.add_argument("--api-base", help="API基础URL")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    try:
        system = EnhancedInvestorBench(args)
        
        if args.mode == "warmup":
            system.run_warmup()
        elif args.mode == "test":
            system.run_test()
        elif args.mode == "both":
            logger.info("🚀 运行完整的Warmup→Test流程")
            system.run_warmup()
            logger.info("⏭️ Warmup完成，开始Test阶段...")
            system.run_test()
        
        system.generate_report()
        logger.info("🎉 运行完成！")
        logger.info(f"📁 结果保存在: {system.run_dir}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 用户中断了程序执行")
        return
    except ImportError as e:
        logger.error(f"❌ 依赖包导入失败: {str(e)}")
        logger.error("💡 请检查是否安装了所有必需的依赖包")
        return
    except ValueError as e:
        if "API" in str(e) or "key" in str(e):
            logger.error(f"❌ API配置错误: {str(e)}")
            logger.error("💡 请检查API密钥和URL配置")
        else:
            logger.error(f"❌ 配置错误: {str(e)}")
        return
    except ConnectionError as e:
        logger.error(f"❌ 网络连接失败: {str(e)}")
        logger.error("💡 请检查网络连接和API服务状态")
        return
    except TimeoutError as e:
        logger.error(f"❌ 请求超时: {str(e)}")
        logger.error("💡 API响应时间过长，请稍后重试或检查API服务状态")
        return
    except FileNotFoundError as e:
        logger.error(f"❌ 文件未找到: {str(e)}")
        logger.error("💡 请检查数据文件是否存在")
        return
    except Exception as e:
        logger.error(f"❌ 运行失败: {str(e)}")
        logger.error("💡 如果问题持续存在，请检查日志文件或联系技术支持")
        if args.verbose:
            import traceback
            logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()