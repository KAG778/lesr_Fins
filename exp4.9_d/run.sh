#!/bin/bash
# Exp4.7 运行脚本

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."

# 激活环境
source /home/wangmeiyi/miniconda3/etc/profile.d/conda.sh
conda activate AuctionNet

# 设置环境变量
export PYTHONPATH=/home/wangmeiyi/AuctionNet/lesr/FINSABER:$PYTHONPATH

# 自动加载保存的API密钥
if [ -f "exp4.7/.api_keys.txt" ]; then
    eval $(grep -v '^#' exp4.7/.api_keys.txt | xargs)
    echo "✓ 已加载保存的API密钥"
fi

# 检查API密钥
if [ -z "$OPENAI_API_KEY" ]; then
    echo "错误: OPENAI_API_KEY 环境变量未设置"
    echo ""
    echo "请使用以下方式之一设置密钥:"
    echo "  1. 运行密钥配置工具: python exp4.7/setup_keys.py"
    echo "  2. 手动设置: export OPENAI_API_KEY=your_key"
    echo "  3. 创建 .api_keys.txt 文件"
    exit 1
fi

# 显示配置信息
echo "配置信息:"
echo "  模型: gpt-4o-mini"
if [ -n "$OPENAI_BASE_URL" ]; then
    echo "  Base URL: $OPENAI_BASE_URL"
else
    echo "  Base URL: 官方API"
fi
echo ""

# 运行实验
python exp4.7/main.py --config exp4.7/config.yaml "$@"
