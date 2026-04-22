#!/bin/bash
# 并行启动 10 个实验 (5 窗口 × 2 方法) 到 4 块 GPU 上
# 每个 GPU 分配 2-3 个实验

set -e
cd /home/wangmeiyi/AuctionNet/lesr/exp4.15
mkdir -p logs

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
echo "=========================================="
echo "并行 10 实验启动 - ${TIMESTAMP}"
echo "GPU 分配: 4 × A100-PCIE-40GB"
echo "=========================================="

# GPU 0: W2_LESR, W2_Baseline, W3_LESR
echo "[GPU 0] 启动 W2 LESR, W2 Baseline, W3 LESR"
CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/main_simple.py --config configs/config_W2.yaml > logs/W2_lesr_parallel_${TIMESTAMP}.log 2>&1 &
PID_W2_LESR=$!
echo "  W2 LESR PID=$PID_W2_LESR"

CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/run_baseline.py --config configs/config_W2.yaml > logs/W2_baseline_parallel_${TIMESTAMP}.log 2>&1 &
PID_W2_BASE=$!
echo "  W2 Baseline PID=$PID_W2_BASE"

CUDA_VISIBLE_DEVICES=0 nohup python3 scripts/main_simple.py --config configs/config_W3.yaml > logs/W3_lesr_parallel_${TIMESTAMP}.log 2>&1 &
PID_W3_LESR=$!
echo "  W3 LESR PID=$PID_W3_LESR"

# GPU 1: W3_Baseline, W4_LESR, W4_Baseline
echo "[GPU 1] 启动 W3 Baseline, W4 LESR, W4 Baseline"
CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/run_baseline.py --config configs/config_W3.yaml > logs/W3_baseline_parallel_${TIMESTAMP}.log 2>&1 &
PID_W3_BASE=$!
echo "  W3 Baseline PID=$PID_W3_BASE"

CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/main_simple.py --config configs/config_W4.yaml > logs/W4_lesr_parallel_${TIMESTAMP}.log 2>&1 &
PID_W4_LESR=$!
echo "  W4 LESR PID=$PID_W4_LESR"

CUDA_VISIBLE_DEVICES=1 nohup python3 scripts/run_baseline.py --config configs/config_W4.yaml > logs/W4_baseline_parallel_${TIMESTAMP}.log 2>&1 &
PID_W4_BASE=$!
echo "  W4 Baseline PID=$PID_W4_BASE"

# GPU 2: W5_LESR, W5_Baseline, W6_LESR
echo "[GPU 2] 启动 W5 LESR, W5 Baseline, W6 LESR"
CUDA_VISIBLE_DEVICES=2 nohup python3 scripts/main_simple.py --config configs/config_W5.yaml > logs/W5_lesr_parallel_${TIMESTAMP}.log 2>&1 &
PID_W5_LESR=$!
echo "  W5 LESR PID=$PID_W5_LESR"

CUDA_VISIBLE_DEVICES=2 nohup python3 scripts/run_baseline.py --config configs/config_W5.yaml > logs/W5_baseline_parallel_${TIMESTAMP}.log 2>&1 &
PID_W5_BASE=$!
echo "  W5 Baseline PID=$PID_W5_BASE"

CUDA_VISIBLE_DEVICES=2 nohup python3 scripts/main_simple.py --config configs/config_W6.yaml > logs/W6_lesr_parallel_${TIMESTAMP}.log 2>&1 &
PID_W6_LESR=$!
echo "  W6 LESR PID=$PID_W6_LESR"

# GPU 3: W6_Baseline
echo "[GPU 3] 启动 W6 Baseline"
CUDA_VISIBLE_DEVICES=3 nohup python3 scripts/run_baseline.py --config configs/config_W6.yaml > logs/W6_baseline_parallel_${TIMESTAMP}.log 2>&1 &
PID_W6_BASE=$!
echo "  W6 Baseline PID=$PID_W6_BASE"

echo ""
echo "=========================================="
echo "10 个实验已全部并行启动!"
echo "=========================================="
echo ""
echo "进程列表:"
echo "  GPU 0: W2_LESR($PID_W2_LESR) W2_BASE($PID_W2_BASE) W3_LESR($PID_W3_LESR)"
echo "  GPU 1: W3_BASE($PID_W3_BASE) W4_LESR($PID_W4_LESR) W4_BASE($PID_W4_BASE)"
echo "  GPU 2: W5_LESR($PID_W5_LESR) W5_BASE($PID_W5_BASE) W6_LESR($PID_W6_LESR)"
echo "  GPU 3: W6_BASE($PID_W6_BASE)"
echo ""
echo "监控命令:"
echo "  python3 scripts/monitor_5_windows.py"
echo "  nvidia-smi"
