#!/bin/bash
# ============================================================
# DouMH 训练脚本 (GPU 优化版)
#
# 核心优化:
#   1. Actor NN 推理移至 GPU (去除 actor_device_cpu)
#      - CPU 专注游戏逻辑 + 蒙特卡洛采样
#      - GPU 4090D 处理 Actor 推理 + Learner 训练，时分复用
#   2. 模型架构可配置 (d_model / nhead / layers / dim_ff)
#   3. 模型架构可配置，默认预留 ~6GB 显存余量防止 OOM
#
# 用法:
#   bash train.sh                                    # 默认参数
#   bash train.sh --total_frames 50000000            # 自定义训练帧数
#   bash train.sh --load_model                       # 从 checkpoint 恢复
#   D_MODEL=256 NHEAD=8 bash train.sh                # 轻量模型 (快速测试)
#   ACTOR_MODE=cpu bash train.sh                     # 回退: CPU Actor
# ============================================================
set -e

# ---------- 环境激活 ----------
PYTHON="f:/python/Python312/python.exe"
export TMP="h:/tmp2"
export TEMP="h:/tmp2"
# PyTorch CUDA allocator: eliminate fragmentation at source
# roundup_power2_divisions:16  — round all allocs to 16 power-of-2 buckets
# max_split_size_mb:128        — don't cache blocks >128MB
# expandable_segments          — Linux only, further defrag (PyTorch ≥ 2.0)
if [ "$OS" != "Windows_NT" ]; then
    export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:16"
else
    export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128,roundup_power2_divisions:16"
fi

# CUDA_LAUNCH_BLOCKING=1 使所有 CUDA 启动同步执行，错误可立即抛出而非静默挂起
# (已禁用 — 同步执行导致严重性能下降，尤其与 LOR 额外输出头叠加时)
# export CUDA_LAUNCH_BLOCKING=1

# ---------- 验证环境 ----------
$PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用!'; print(f'[OK] PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')"

# ---------- 工作目录 ----------
cd "h:/bot/DouEnMH-design/DouEnMH/DouEnMH-new"

# ============================================================
# 可配置参数 (通过环境变量覆盖)
# ============================================================

# --- 训练规模 ---
XPID="${XPID:-douEnMH_v1.0}"                  # 实验名称
TOTAL_FRAMES="${TOTAL_FRAMES:-2000}" # 总训练帧数 (3亿)
NUM_ACTORS="${NUM_ACTORS:-2}"             # Actor 进程数
SAVE_INTERVAL="${SAVE_INTERVAL:-10}"      # checkpoint 保存间隔 (分钟)
BATCH_SIZE="${BATCH_SIZE:-2}"            # Learner 批次大小 (越大显存越多)
UNROLL_LENGTH="${UNROLL_LENGTH:-5}"       # 轨迹展开长度 (越大显存越多)
LEARNING_RATE="${LEARNING_RATE:-0.0003}"  # 学习率
GRAD_ACCUM="${GRAD_ACCUM:-1}"             # 梯度累积步数 (1=不累积)
EXP_EPSILON="${EXP_EPSILON:-0.3}"         # Epsilon 探索率 (叫分阶段需要高探索)
BID_EXP_EPSILON="${BID_EXP_EPSILON:-0.3}"   # 叫分独立探索率

# --- 叫分课程 ---
BID_CURRICULUM_GAMES="${BID_CURRICULUM_GAMES:-5000}"  # P0叫0分衰减局数 (0=关闭)

# --- 日志控制 ---
LOG_GAME_STEPS="${LOG_GAME_STEPS:-0}"        # 0=关闭每局步数日志, 1=开启 (多actor时刷屏严重)

# --- 模型架构 (安全余量, 预计 ~18GB 显存) ---
D_MODEL="${D_MODEL:-16}"                 # Transformer 嵌入维度
NHEAD="${NHEAD:-4}"                       # 多头注意力头数
NUM_ENCODER_LAYERS="${NUM_ENCODER_LAYERS:-4}"  # Encoder 层数
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-256}"     # FFN 隐藏.3层维度

# --- Actor 模式: gpu (推荐) / cpu (回退) ---
ACTOR_MODE="${ACTOR_MODE:-gpu}"


# ============================================================
# 打印配置
# ============================================================
echo "============================================"
echo " DouMH 训练 (GPU 优化版)"
echo "============================================"
echo " 实验名称:       $XPID"
echo " 总帧数:         $TOTAL_FRAMES"
echo " Actor 模式:     $ACTOR_MODE"
echo " Actor 数量:     $NUM_ACTORS"
echo " 模型配置:       d_model=$D_MODEL nhead=$NHEAD layers=$NUM_ENCODER_LAYERS dim_ff=$DIM_FEEDFORWARD"
echo " 批次大小:       $BATCH_SIZE x $UNROLL_LENGTH"
echo " 学习率:         $LEARNING_RATE"
echo " 梯度累积:       $GRAD_ACCUM"
echo " 保存间隔:       ${SAVE_INTERVAL}min"
echo " 输出目录:       checkpoints/$XPID/"
echo "============================================"

# ---------- 构建参数列表 ----------
ARGS=(
    --xpid "$XPID"
    --total_frames "$TOTAL_FRAMES"
    --num_actors "$NUM_ACTORS"
    --training_device 0
    --gpu_devices 0
    --save_interval "$SAVE_INTERVAL"
    --batch_size "$BATCH_SIZE"
    --unroll_length "$UNROLL_LENGTH"
    --learning_rate "$LEARNING_RATE"
    --gradient_accumulation_steps "$GRAD_ACCUM"
    --exp_epsilon "$EXP_EPSILON"
    --bid_exp_epsilon "$BID_EXP_EPSILON"
    --bid_curriculum_games "$BID_CURRICULUM_GAMES"
    --d_model "$D_MODEL"
    --nhead "$NHEAD"
    --num_encoder_layers "$NUM_ENCODER_LAYERS"
    --dim_feedforward "$DIM_FEEDFORWARD"
)

# Actor 设备配置
if [ "$ACTOR_MODE" = "cpu" ]; then
    ARGS+=(--actor_device_cpu)
    echo "  [Mode] Actor 推理在 CPU, GPU 仅用于训练"
else
    ARGS+=(--num_actor_devices 1)
    echo "  [Mode] Actor 推理在 GPU, CPU 仅处理游戏逻辑"
fi

[ "$LOG_GAME_STEPS" = "1" ] && ARGS+=(--log_game_steps)

# ---------- 启动训练 ----------
echo ""
$PYTHON train.py "${ARGS[@]}" "$@"

echo ""
echo "训练完成。模型保存在 checkpoints/$XPID/"
