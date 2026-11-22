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


# ---------- 验证环境 ----------
$PYTHON -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用!'; print(f'[OK] PyTorch {torch.__version__}, GPU: {torch.cuda.get_device_name(0)}')"

# ---------- 工作目录 ----------
cd "h:/bot/DouEnMH-design/DouEnMH/DouEnMH-new"

# ============================================================
# 可配置参数 (通过环境变量覆盖)
# ============================================================

# --- 训练规模 ---
XPID="${XPID:-DouEnMH_v1.2}"  # 实验名称
TOTAL_FRAMES="${TOTAL_FRAMES:-3000000000}" # 总训练帧数 (30亿)
NUM_ACTORS="${NUM_ACTORS:-8}"             # Actor 进程数
SAVE_INTERVAL="${SAVE_INTERVAL:-15}"      # checkpoint 保存间隔 (分钟)
BATCH_SIZE="${BATCH_SIZE:-21}"            # Learner 批次大小 (越大显存越多)
UNROLL_LENGTH="${UNROLL_LENGTH:-100}"       # 轨迹展开长度 (越大显存越多)
LEARNING_RATE="${LEARNING_RATE:-0.0001}"  # 学习率
GRAD_ACCUM="${GRAD_ACCUM:-3}"             # 梯度累积步数 (1=不累积)
EXP_EPSILON="${EXP_EPSILON:-0.01}"          # 出牌探索率终值 (对齐论文)
EXP_EPSILON_START="${EXP_EPSILON_START:-0.03}"  # 出牌探索率初值
BID_EXP_EPSILON="${BID_EXP_EPSILON:-0.05}"   # 叫分探索率终值
BID_EXP_EPSILON_START="${BID_EXP_EPSILON_START:-0.3}"  # 叫分探索率初值
EPSILON_DECAY_GAMES="${EPSILON_DECAY_GAMES:-20000}"  # 每 actor 衰减局数

# --- 叫分课程 ---
BID_CURRICULUM_GAMES="${BID_CURRICULUM_GAMES:-500000}"   # P0叫0分衰减局数
BID_BATCH_SIZE="${BID_BATCH_SIZE:-8}"        # 叫分位置专用 batch (小样本位置)

# --- 神谕辅助头 ---
ORACLE_DECAY_RATIO="${ORACLE_DECAY_RATIO:-0.3}"      # 神谕 λ 衰减覆盖的训练比例 (0.2=前20%帧数)
ORACLE_LAMBDA_LANDLORD="${ORACLE_LAMBDA_LANDLORD:-1.0}"  # n_active 神谕 λ 初值
ORACLE_LAMBDA_PEASANT="${ORACLE_LAMBDA_PEASANT:-0.6}"    # n_passive/ΔN 神谕 λ 初值
ORACLE_LAMBDA_MIN="${ORACLE_LAMBDA_MIN:-0.05}"       # 神谕 λ 下限

# --- 日志控制 ---
LOG_GAME_STEPS="${LOG_GAME_STEPS:-0}"        # 0=关闭每局步数日志, 1=开启 (多actor时刷屏严重)

# --- 模型架构 (安全余量, 预计 ~18GB 显存) ---
D_MODEL="${D_MODEL:-384}"                 # Transformer 嵌入维度
NHEAD="${NHEAD:-8}"                       # 多头注意力头数
NUM_ENCODER_LAYERS="${NUM_ENCODER_LAYERS:-6}"  # Encoder 层数
DIM_FEEDFORWARD="${DIM_FEEDFORWARD:-1280}"     # FFN 隐藏.3层维度

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
echo " 架构:           V2 Cross-Attention"
echo " Actor 模式:     $ACTOR_MODE"
echo " Actor 数量:     $NUM_ACTORS"
echo " 模型配置:       d_model=$D_MODEL nhead=$NHEAD layers=$NUM_ENCODER_LAYERS dim_ff=$DIM_FEEDFORWARD"
echo " 出牌 batch:     $BATCH_SIZE 叫分 batch: $BID_BATCH_SIZE"
echo " unroll_length:  $UNROLL_LENGTH"
echo " 学习率:         $LEARNING_RATE"
echo " 梯度累积:       $GRAD_ACCUM"
echo " 出牌探索率:     $EXP_EPSILON_START → $EXP_EPSILON (over ${EPSILON_DECAY_GAMES} games)"
echo " 叫分探索率:     $BID_EXP_EPSILON_START → $BID_EXP_EPSILON"
echo " 神谕 λ:        L=${ORACLE_LAMBDA_LANDLORD} P=${ORACLE_LAMBDA_PEASANT}→${ORACLE_LAMBDA_MIN} over ${ORACLE_DECAY_RATIO}×frames"
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
    --exp_epsilon_start "$EXP_EPSILON_START"
    --bid_exp_epsilon "$BID_EXP_EPSILON"
    --bid_exp_epsilon_start "$BID_EXP_EPSILON_START"
    --epsilon_decay_games "$EPSILON_DECAY_GAMES"
    --bid_curriculum_games "$BID_CURRICULUM_GAMES"
    --bid_batch_size "$BID_BATCH_SIZE"
    --oracle_decay_ratio "$ORACLE_DECAY_RATIO"
    --oracle_lambda_landlord "$ORACLE_LAMBDA_LANDLORD"
    --oracle_lambda_peasant "$ORACLE_LAMBDA_PEASANT"
    --oracle_lambda_min "$ORACLE_LAMBDA_MIN"
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
