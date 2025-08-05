#!/usr/bin/env python3
"""
INVESTOR-BENCH - 基于LLM的智能投资回测系统
支持OpenAI兼容API的统一命令行接口

使用方法:
python investor_bench.py --symbol JNJ --start-date 2020-07-02 --end-date 2020-07-10 --mode warmup
python investor_bench.py --symbol JNJ --model gpt-4 --api-key sk-xxx --api-base https://api.openai.com/v1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Any
import traceback

import numpy as np
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
import httpx

# 加载环境变量
load_dotenv()

class InvestorBench:
    """INVESTOR-BENCH主控制器"""
    
    def __init__(self, args):
        self.args = args
        self.start_time = datetime.now()
        
        # 设置运行目录 - 带时间戳和模型名称
        timestamp = self.start_time.strftime("%y%m%d_%H%M%S")
        model_name = self._clean_model_name(args.model)
        self.run_dir = Path("results") / f"{timestamp}_{model_name}_{args.symbol}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置详细的日志
        log_file = self.run_dir / "run.log"
        logger.add(log_file, level="TRACE", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
        
        # 初始化OpenAI客户端
        self.client = self._init_openai_client()
        
        # 初始化API统计
        self.api_stats = {
            "chat_calls": 0,
            "embedding_calls": 0,
            "total_tokens": 0,
            "errors": 0
        }
        
        # 简化的记忆存储
        self.memories = []
        
        logger.info(f"🚀 INVESTOR-BENCH启动 - 运行目录: {self.run_dir}")
        
    def _clean_model_name(self, model: str) -> str:
        """清理模型名称用于文件路径"""
        return model.replace("/", "_").replace(":", "_").replace(" ", "_")
    
    def _init_openai_client(self) -> OpenAI:
        """初始化OpenAI客户端"""
        # 优先级：命令行参数 > 环境变量 > 默认值
        api_key = self.args.api_key or os.getenv('OPENAI_API_KEY')
        api_base = self.args.api_base or os.getenv('OPENAI_API_BASE', 'https://api.siliconflow.cn/v1')
        
        if not api_key:
            raise ValueError("❌ API密钥未设置！请通过--api-key参数或OPENAI_API_KEY环境变量提供")
        
        logger.info(f"🔧 OpenAI客户端配置:")
        logger.info(f"   模型: {self.args.model}")
        logger.info(f"   API Base: {api_base}")
        logger.info(f"   API Key: {api_key[:10]}...")
        
        return OpenAI(api_key=api_key, base_url=api_base)
    
    def call_llm(self, prompt: str, system_message: str = None, max_tokens: int = 1024) -> Dict[str, Any]:
        """调用LLM API with统计和错误处理"""
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        try:
            self.api_stats["chat_calls"] += 1
            
            response = self.client.chat.completions.create(
                model=self.args.model,
                messages=messages,
                temperature=0.6,
                max_tokens=max_tokens,
                response_format={"type": "json_object"}
            )
            
            # 更新统计
            if hasattr(response, 'usage') and response.usage:
                self.api_stats["total_tokens"] += response.usage.total_tokens
            
            content = response.choices[0].message.content
            return json.loads(content)
            
        except Exception as e:
            self.api_stats["errors"] += 1
            logger.error(f"❌ LLM调用失败: {e}")
            return {"error": str(e), "action": "HOLD", "confidence": 0.0, "reasoning": "API调用失败"}
    
    def call_embedding(self, text: str) -> List[float]:
        """调用Embedding API"""
        try:
            self.api_stats["embedding_calls"] += 1
            
            # 使用固定的SiliconFlow embedding
            embed_client = OpenAI(
                api_key=os.getenv('OPENAI_API_KEY'),
                base_url='https://api.siliconflow.cn/v1'
            )
            
            response = embed_client.embeddings.create(
                model="Qwen/Qwen3-Embedding-4B",
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding调用失败: {e}")
            return [0.0] * 2560  # 返回默认维度的零向量
    
    def load_market_data(self) -> List[Dict]:
        """加载市场数据"""
        data_file = Path("data") / f"{self.args.symbol.lower()}.json"
        
        if not data_file.exists():
            raise FileNotFoundError(f"❌ 数据文件不存在: {data_file}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 转换数据格式：从{date: {data}} 转为 [{date, data}]
        data = []
        for date_str, day_info in raw_data.items():
            data.append({
                'date': date_str,
                'symbol': self.args.symbol,
                'price': day_info.get('prices', 0),
                'news': day_info.get('news', []),
                'filing_k': day_info.get('10k', []),
                'filing_q': day_info.get('10q', [])
            })
        
        # 按日期排序
        data.sort(key=lambda x: x['date'])
        
        # 过滤日期范围
        filtered_data = []
        start_date = datetime.strptime(self.args.start_date, '%Y-%m-%d').date()
        end_date = datetime.strptime(self.args.end_date, '%Y-%m-%d').date()
        
        for day_data in data:
            day_date = datetime.strptime(day_data['date'], '%Y-%m-%d').date()
            if start_date <= day_date <= end_date:
                filtered_data.append(day_data)
        
        logger.info(f"📊 加载数据: {len(filtered_data)}天, {start_date} 至 {end_date}")
        return filtered_data
    
    def investment_decision(self, day_data: Dict) -> Dict[str, Any]:
        """制定投资决策"""
        symbol = day_data.get('symbol', self.args.symbol)
        price = day_data.get('price', 0)
        news = day_data.get('news', [])
        date_str = day_data['date']
        
        # 构建上下文
        news_text = "\\n".join([f"- {item}" for item in news[:5]])  # 最多5条新闻
        memory_context = "\\n".join([f"记忆: {mem}" for mem in self.memories[-3:]])  # 最近3条记忆
        
        # 构建决策prompt
        prompt = f"""
请基于以下信息为{symbol}制定投资决策：

## 当前市场信息
- 股票代码: {symbol}
- 当前价格: ${price:.2f}
- 分析日期: {date_str}
- 运行模式: {self.args.mode}

## 今日新闻
{news_text if news_text.strip() else "无重要新闻"}

## 历史记忆
{memory_context if memory_context.strip() else "暂无历史记忆"}

## 专业背景
你是{symbol}的专业投资分析师，请基于上述信息做出投资决策。

## 要求
请返回标准JSON格式:
{{
    "action": "BUY, SELL, 或 HOLD",
    "confidence": 0.0到1.0的置信度数值,
    "reasoning": "详细的决策理由说明",
    "key_factors": ["影响决策的关键因素1", "关键因素2", "关键因素3"],
    "memory_update": "需要记住的重要信息(可选)"
}}
"""
        
        system_message = f"你是专业的{symbol}投资分析师。在{self.args.mode}模式下提供投资建议。始终返回有效JSON。"
        
        logger.info(f"🧠 分析 {symbol} - {date_str}")
        decision = self.call_llm(prompt, system_message, max_tokens=1024)
        
        # 更新记忆
        if decision.get("memory_update"):
            self.memories.append(f"{date_str}: {decision['memory_update']}")
            if len(self.memories) > 10:  # 限制记忆条数
                self.memories.pop(0)
        
        return decision
    
    def run_backtest(self):
        """运行回测"""
        logger.info(f"🎯 开始回测 - 模式: {self.args.mode}")
        
        # 加载市场数据
        market_data = self.load_market_data()
        
        if not market_data:
            raise ValueError("❌ 没有找到符合日期范围的数据")
        
        results = []
        portfolio_value = 10000.0  # 初始资金
        position = 0  # 持仓数量
        
        for i, day_data in enumerate(market_data):
            logger.info(f"📅 处理第{i+1}/{len(market_data)}天: {day_data['date']}")
            
            # 制定投资决策
            decision = self.investment_decision(day_data)
            
            # 模拟执行交易
            price = day_data.get('price', 0)
            action = decision.get('action', 'HOLD')
            
            if action == 'BUY' and position == 0:
                position = portfolio_value / price
                portfolio_value = 0
                logger.info(f"💰 买入 {position:.2f}股 @ ${price:.2f}")
            elif action == 'SELL' and position > 0:
                portfolio_value = position * price
                position = 0
                logger.info(f"💸 卖出获得 ${portfolio_value:.2f}")
            
            # 计算当前价值
            current_value = portfolio_value + (position * price)
            
            result = {
                "date": day_data['date'],
                "symbol": self.args.symbol,
                "price": price,
                "decision": decision,
                "position": position,
                "portfolio_value": current_value,
                "daily_return": 0 if i == 0 else (current_value / results[-1]["portfolio_value"] - 1)
            }
            
            results.append(result)
            
            # 输出进度
            confidence = decision.get('confidence', 0)
            reasoning = decision.get('reasoning', 'N/A')
            logger.info(f"🎯 决策: {action} (置信度: {confidence:.2f})")
            logger.info(f"📝 投资组合价值: ${current_value:.2f}")
            logger.info(f"💡 理由: {reasoning[:100]}...")
            logger.info("-" * 60)
        
        return results
    
    def calculate_trading_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """计算交易指标"""
        if len(results) < 2:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "volatility": 0.0,
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "total_trades": 0
            }
        
        # 提取价格和投资组合价值数据
        prices = [r["price"] for r in results]
        portfolio_values = [r["portfolio_value"] for r in results]
        actions = [r["decision"].get("action", "HOLD") for r in results]
        
        # 计算每日收益率
        daily_returns = []
        for i in range(1, len(portfolio_values)):
            daily_return = (portfolio_values[i] / portfolio_values[i-1]) - 1
            daily_returns.append(daily_return)
        
        # 基本指标
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1
        trading_days = len(results)
        annualized_return = ((1 + total_return) ** (252 / trading_days)) - 1 if trading_days > 0 else 0
        
        # 波动率
        volatility = np.std(daily_returns) * np.sqrt(252) if daily_returns else 0
        
        # 夏普比率 (假设无风险利率为0)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # 最大回撤
        max_drawdown = self._calculate_max_drawdown(portfolio_values)
        
        # 交易统计
        buy_actions = actions.count("BUY")
        sell_actions = actions.count("SELL")
        total_trades = buy_actions + sell_actions
        
        # 胜率计算 (简化版本)
        profitable_days = sum(1 for r in daily_returns if r > 0)
        win_rate = profitable_days / len(daily_returns) if daily_returns else 0
        
        return {
            "total_return": total_return,
            "annualized_return": annualized_return,
            "volatility": volatility,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "buy_actions": buy_actions,
            "sell_actions": sell_actions
        }
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        if len(portfolio_values) < 2:
            return 0.0
            
        peak = portfolio_values[0]
        max_drawdown = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                
        return max_drawdown
    
    def generate_reports(self, results: List[Dict]):
        """生成分析报告"""
        logger.info("📊 生成投资分析报告...")
        
        # 计算交易指标
        trading_metrics = self.calculate_trading_metrics(results)
        logger.info("📈 交易指标计算完成")
        
        # 保存详细结果JSON
        results_file = self.run_dir / "results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 生成CSV报告
        self._generate_csv_report(results)
        
        # 生成Markdown报告
        self._generate_markdown_report(results, trading_metrics)
        
        logger.info(f"✅ 报告已生成在 {self.run_dir}")
    
    def _generate_csv_report(self, results: List[Dict]):
        """生成CSV格式报告"""
        import csv
        
        csv_file = self.run_dir / "trading_results.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if not results:
                return
                
            fieldnames = ['date', 'symbol', 'price', 'action', 'confidence', 'position', 'portfolio_value', 'daily_return']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                writer.writerow({
                    'date': result['date'],
                    'symbol': result['symbol'],
                    'price': result['price'],
                    'action': result['decision'].get('action', 'HOLD'),
                    'confidence': result['decision'].get('confidence', 0),
                    'position': result['position'],
                    'portfolio_value': result['portfolio_value'],
                    'daily_return': result['daily_return']
                })
    
    def _generate_markdown_report(self, results: List[Dict], trading_metrics: Dict[str, float]):
        """生成Markdown格式报告"""
        if not results:
            return
            
        # 计算统计数据
        total_days = len(results)
        actions = [r["decision"].get("action", "HOLD") for r in results]
        buy_count = actions.count("BUY")
        sell_count = actions.count("SELL")
        hold_count = actions.count("HOLD")
        
        initial_value = results[0]["portfolio_value"]
        final_value = results[-1]["portfolio_value"]
        total_return = (final_value / initial_value - 1) * 100
        
        avg_confidence = sum([r["decision"].get("confidence", 0) for r in results]) / total_days
        
        # 生成报告
        report = f"""# INVESTOR-BENCH 投资回测报告

## 📊 基本信息

| 项目 | 值 |
|------|---|
| 股票代码 | {self.args.symbol} |
| 分析期间 | {results[0]['date']} 至 {results[-1]['date']} |
| 总交易日 | {total_days} |
| 运行模式 | {self.args.mode} |
| 使用模型 | {self.args.model} |
| 运行时间 | {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} |

## 💰 投资表现

| 指标 | 值 |
|------|---|
| 初始资金 | ${initial_value:,.2f} |
| 最终价值 | ${final_value:,.2f} |
| 总收益率 | {total_return:+.2f}% |
| 年化收益率 | {trading_metrics['annualized_return']*100:+.2f}% |
| 平均置信度 | {avg_confidence:.2f} |

## 📊 风险指标

| 指标 | 值 |
|------|---|
| 波动率 (年化) | {trading_metrics['volatility']*100:.2f}% |
| 夏普比率 | {trading_metrics['sharpe_ratio']:.3f} |
| 最大回撤 | {trading_metrics['max_drawdown']*100:.2f}% |
| 胜率 | {trading_metrics['win_rate']*100:.1f}% |
| 总交易次数 | {trading_metrics['total_trades']} |

## 🎯 决策分布

| 决策类型 | 次数 | 占比 |
|---------|------|------|
| BUY | {buy_count} | {buy_count/total_days*100:.1f}% |
| HOLD | {hold_count} | {hold_count/total_days*100:.1f}% |
| SELL | {sell_count} | {sell_count/total_days*100:.1f}% |

## 📈 详细交易记录

| 日期 | 价格 | 决策 | 置信度 | 组合价值 | 日收益率 |
|------|------|------|--------|----------|----------|
"""
        
        for result in results:
            decision = result["decision"]
            report += f"| {result['date']} | ${result['price']:.2f} | {decision.get('action', 'HOLD')} | {decision.get('confidence', 0):.2f} | ${result['portfolio_value']:,.2f} | {result['daily_return']*100:+.2f}% |\\n"
        
        report += f"""
## 🧠 AI决策洞察

### 高置信度决策 (>0.8)
"""
        high_conf_decisions = [r for r in results if r["decision"].get("confidence", 0) > 0.8]
        for result in high_conf_decisions[:5]:  # 最多显示5个
            decision = result["decision"]
            report += f"- **{result['date']}**: {decision.get('action')} (置信度: {decision.get('confidence', 0):.2f})\\n"
            report += f"  理由: {decision.get('reasoning', 'N/A')[:150]}...\\n\\n"
        
        report += f"""
## 🔧 技术信息

### API使用统计
- Chat API调用: {self.api_stats['chat_calls']}次
- Embedding API调用: {self.api_stats['embedding_calls']}次  
- 总Token消耗: {self.api_stats['total_tokens']}
- API错误次数: {self.api_stats['errors']}

### 系统配置
- OpenAI SDK版本: {self.client._version if hasattr(self.client, '_version') else '未知'}
- API Base: {self.client.base_url}
- 运行目录: {self.run_dir}

---
*报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        # 保存报告
        report_file = self.run_dir / "analysis_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
    
    def save_metadata(self):
        """保存运行元数据"""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        metadata = {
            "run_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "run_directory": str(self.run_dir),
                "command_line": " ".join(sys.argv)
            },
            "configuration": {
                "symbol": self.args.symbol,
                "start_date": self.args.start_date,
                "end_date": self.args.end_date,
                "mode": self.args.mode,
                "model": self.args.model,
                "api_base": str(self.client.base_url) if self.client else None,
                "embedding_model": "Qwen/Qwen3-Embedding-4B"
            },
            "api_statistics": self.api_stats,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "working_directory": os.getcwd()
            },
            "memory_records": len(self.memories),
            "version": "2.0.0"
        }
        
        metadata_file = self.run_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"💾 元数据已保存: {metadata_file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="INVESTOR-BENCH - 基于LLM的智能投资回测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用环境变量中的API配置
  python investor_bench.py --symbol JNJ --start-date 2020-07-02 --end-date 2020-07-10 --mode warmup
  
  # 指定特定的API参数
  python investor_bench.py --symbol AAPL --model gpt-4 --api-key sk-xxx --api-base https://api.openai.com/v1
  
  # 使用本地部署的模型（通过OpenAI SDK）
  python investor_bench.py --symbol TSLA --model llama-3.1-8b --api-base http://localhost:8000/v1 --api-key fake
        """
    )
    
    # 必需参数
    parser.add_argument("--symbol", required=True, help="股票代码 (如: JNJ, AAPL)")
    parser.add_argument("--start-date", required=True, help="开始日期 (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="结束日期 (YYYY-MM-DD)")
    
    # 可选参数
    parser.add_argument("--mode", default="backtest", choices=["warmup", "test", "backtest"], 
                        help="运行模式 (默认: backtest)")
    parser.add_argument("--model", default="Qwen/Qwen3-8B", help="LLM模型名称")
    parser.add_argument("--api-key", help="OpenAI API密钥 (或设置OPENAI_API_KEY环境变量)")
    parser.add_argument("--api-base", help="API基础URL (或设置OPENAI_API_BASE环境变量)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    
    args = parser.parse_args()
    
    try:
        # 初始化系统
        system = InvestorBench(args)
        
        # 运行回测
        results = system.run_backtest()
        
        # 生成报告
        system.generate_reports(results)
        
        # 保存元数据
        system.save_metadata()
        
        logger.info("🎉 回测完成！")
        logger.info(f"📁 结果保存在: {system.run_dir}")
        print(f"\\n✅ 回测完成！结果保存在: {system.run_dir}")
        
    except Exception as e:
        logger.error(f"❌ 系统错误: {e}")
        logger.error(traceback.format_exc())
        print(f"\\n❌ 运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()