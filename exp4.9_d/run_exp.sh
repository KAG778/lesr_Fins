#!/bin/bash
# Exp4.7 启动脚本

cd /home/wangmeiyi/AuctionNet/lesr

# 激活环境
source /home/wangmeiyi/miniconda3/etc/profile.d/conda.sh
conda activate lesr

# 设置路径
export PYTHONPATH=/home/wangmeiyi/AuctionNet/lesr/FINSABER:/home/wangmeiyi/AuctionNet/lesr/exp4.7:/home/wangmeiyi/AuctionNet/lesr:$PYTHONPATH

# 运行实验
python exp4.7/main.py --config exp4.7/config.yaml "$@"
