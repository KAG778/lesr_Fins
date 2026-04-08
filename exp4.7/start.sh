#!/bin/bash
# Exp4.7 一键启动脚本

cd /home/wangmeiyi/AuctionNet/lesr

echo "=========================================="
echo "Exp4.7 金融交易LESR优化实验"
echo "=========================================="

# 激活环境
echo "激活lesr环境..."
source /home/wangmeiyi/miniconda3/etc/profile.d/conda.sh
conda activate lesr

# 设置路径
export PYTHONPATH=/home/wangmeiyi/AuctionNet/lesr/exp4.7:/home/wangmeiyi/AuctionNet/lesr:$PYTHONPATH

# 检查配置
echo "检查配置..."
if grep -q "sk-your" exp4.7/config.yaml; then
    echo "❌ 请先在config.yaml中设置API密钥!"
    exit 1
fi

echo "✓ 配置检查通过"
echo ""

# 创建日志目录
mkdir -p exp4.7/logs
mkdir -p exp4.7/results

echo "开始运行实验..."
echo "日志: exp4.7/logs/run.log"
echo "=========================================="

# 运行实验
python exp4.7/main_simple.py 2>&1 | tee exp4.7/logs/run.log
