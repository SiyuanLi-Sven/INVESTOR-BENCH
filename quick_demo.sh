#!/bin/bash
# INVESTOR-BENCH 快速演示脚本
# 运行完整的两阶段JNJ实验

echo "🚀 INVESTOR-BENCH 快速演示"
echo "=========================="
echo ""

# 检查环境变量
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ 请先设置API密钥："
    echo "export OPENAI_API_KEY='sk-your-api-key'"
    echo "export OPENAI_API_BASE='https://api.siliconflow.cn/v1'"
    echo ""
    exit 1
fi

echo "✅ API配置检测完成"
echo "模型: Qwen/Qwen3-8B"
echo "API Base: ${OPENAI_API_BASE:-https://api.siliconflow.cn/v1}"
echo "API Key: ${OPENAI_API_KEY:0:10}..."
echo ""

echo "📅 运行JNJ完整两阶段实验"
echo "=========================="

echo ""
echo "🎓 阶段1: Warmup (AI学习期)"
echo "日期: 2020-07-02 至 2020-07-10 (9天)"
echo "目的: 让AI学习JNJ的历史表现和市场特性"
echo ""

python investor_bench.py \
    --symbol JNJ \
    --start-date 2020-07-02 \
    --end-date 2020-07-10 \
    --mode warmup \
    --verbose

if [ $? -ne 0 ]; then
    echo "❌ Warmup阶段失败"
    exit 1
fi

echo ""
echo "🎯 阶段2: Test (AI实战期)"
echo "日期: 2020-10-01 至 2020-10-15 (短期测试)"
echo "目的: 基于学到的知识进行实际投资决策"
echo ""

python investor_bench.py \
    --symbol JNJ \
    --start-date 2020-10-01 \
    --end-date 2020-10-15 \
    --mode test \
    --verbose

if [ $? -ne 0 ]; then
    echo "❌ Test阶段失败"
    exit 1
fi

echo ""
echo "🎉 演示完成！"
echo "=============="
echo ""
echo "📊 查看结果："
echo "ls -la results/ | head -5"
ls -la results/ | head -5

echo ""
echo "📈 查看最新报告："
LATEST_DIR=$(ls -t results/ | head -1)
if [ -n "$LATEST_DIR" ] && [ -f "results/$LATEST_DIR/analysis_report.md" ]; then
    echo "cat results/$LATEST_DIR/analysis_report.md | head -20"
    cat "results/$LATEST_DIR/analysis_report.md" | head -20
    echo "..."
    echo ""
    echo "💡 完整报告位置: results/$LATEST_DIR/analysis_report.md"
fi

echo ""
echo "🚀 下一步："
echo "1. 查看完整的CLI命令参考: docs/cli-quick-reference.md"
echo "2. 运行更长期的实验，将test结束日期改为 2021-05-06"
echo "3. 尝试其他股票: MSFT, BTC, ETH"
echo ""
echo "感谢使用INVESTOR-BENCH! 🎯"