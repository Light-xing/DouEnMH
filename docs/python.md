# python/ 训练端

本文件说明 `python/` 训练端的依赖与环境、目录职责、训练/续训/评测/推理命令、模型结构、DMC 训练流程、`arguments.py` 全部参数，以及 checkpoint 的格式与加载规则。所有结论均以当前代码为准（关键实现位置在各节标注）。

---

## 一、依赖与环境

`python/` 是 DouZero DMC 框架的 PyTorch 实现：多个 actor 进程在 GPU 上自对弈产生样本，单 learner 线程优化 6 个价值网络（3 个叫分模型 + 3 个出牌模型）。

### 1.1 必需依赖

| 依赖 | 要求 | 说明 |
|------|------|------|
| Python | ≥ 3.8 | `douzero/env/opponent_modeling.py` 使用 `math.comb`（3.8 引入）；已验证 3.12 可运行 |
| PyTorch | ≥ 1.9 | `requirements.txt` 声明 `torch>=1.9.0`；代码使用 `torch.inference_mode()`（`douzero/dmc/utils.py`）与 `nn.Mish`/`TransformerEncoder(batch_first=True)`（`douzero/dmc/models.py`），实际训练在 2.x 上验证 |
| numpy | 任意近期版本 | 环境编码、特征组装、DP 均基于 `np.ndarray` |
| CUDA | GPU 训练时需要 | actor 在 GPU 推理、learner 在 GPU 训练；纯 CPU 训练需同时传 `--actor_device_cpu --training_device cpu` |

### 1.2 可选依赖（缺失时自动降级）

| 依赖 | 用途 | 缺失时的行为 |
|------|------|--------------|
| numba | `douzero/env/numba_utils.py` 用 `@njit(cache=True)` 加速 `cards2array_v2`、`_cards_to_rank_counts`、`_cards2tensor_v2`，模块导入时自动 patch 到 `env.py` / `opponent_modeling.py` / `dmc/utils.py` | 导入失败时 `_HAS_NUMBA=False`，patch 跳过，全部走纯 Python 实现；结果一致，只是编码更慢。`Numba_DISABLE=1` 可强制关闭 |
| pandas | `auto_test.py` 结果写 CSV、读历史 CSV | `auto_test.py` 在模块顶部 `import pandas`，缺失时该脚本无法启动；训练/评测不受影响 |
| matplotlib | `auto_test.py` 画 win rate / score 曲线 | 同上，缺失时 `auto_test.py` 无法启动 |
| rlcard | 评测中的 `rlcard` 对手（`douzero/evaluation/rlcard_agent.py`） | 该模块在文件顶部 `from rlcard... import CARD_TYPE`，只有把某个席位指定为 `rlcard` 时才会被导入；缺失时其他对手（`random` / `.ckpt`）照常可用 |

### 1.3 神谕 C 扩展（可选，强烈建议）

`douzero/env/oracle_core.c` 是神谕 DP 的 C 实现，编译成 `douzero/env/oracle_core.dll` 后由 `douzero/env/oracle.py` 通过 `ctypes.CDLL` 加载（文件名固定为 `oracle_core.dll`，位于 `oracle.py` 同目录）。

- 存在 DLL 时：每局首次 DP 约 10ms 量级，并带跨调用 memo 缓存；`env.reset()` 时清空。
- 不存在 DLL 或 `ctypes` 加载失败时：自动回退到纯 Python DP（`_dp_unified_py`），每局首次 DP 约 700ms 量级（见 `oracle.py` 的 `dp_unified` docstring），训练明显变慢但功能完整、结果等价。
- `ORACLE_PLACEHOLDER=1` 环境变量是调试开关：跳过所有 DP，返回固定占位值（`oracle.py` 顶部），只用于跑通流程。
- 编译方法见“八、常见问题”。

### 1.4 环境变量

| 变量 | 默认 | 作用 |
|------|------|------|
| `DOUENMH_ENABLE_C5` | `'1'`（开启） | 控制 X 向量末尾 4 维 C5 对手行为风格特征；设为 `'0'` 时关闭，X 从 56 维变为 52 维（`opponent_modeling.ENABLE_C5` 与 `get_x_dim()`）。**训练、续训、导出的取值必须一致**，否则加载 checkpoint 会因维度不匹配报错 |
| `Numba_DISABLE` | `'0'` | 设为 `'1'` 时不安装 numba 加速 patch |
| `ORACLE_PLACEHOLDER` | `'0'` | 设为 `'1'` 时神谕 DP 返回占位值（调试用） |
| `PYTORCH_CUDA_ALLOC_CONF` | 未设置时由 `train.py` 补齐 | 控制 CUDA allocator 碎片：`max_split_size_mb:128`、`roundup_power2_divisions:16`，非 Windows 再加 `expandable_segments:True`。两个训练脚本也会显式设置 |

---

## 二、目录结构

`python/` 下每个文件/目录的职责：

| 路径 | 职责 |
|------|------|
| `train.py` | 训练入口：先补 `PYTORCH_CUDA_ALLOC_CONF`，解析 `arguments.py`，按 `--gpu_devices` 设置 `CUDA_VISIBLE_DEVICES`，调用 `douzero.dmc.dmc.train()` |
| `evaluate.py` | 单次批量评测：用固定发牌数据跑多进程对局，打印 WP/ADP 统计 |
| `auto_test.py` | 自动评测循环：等在训练目录里发现新 checkpoint 就评测一次，写 CSV 与曲线 |
| `infer.py` | 轻量单局推理：加载指定 `.ckpt`，对一手牌/一次叫分输出 top-k 动作与 Q 值 |
| `game_eval.py` | 评测专用的 GameEnv 副本（`evaluation/` 以顶层模块 `game_eval` 导入它；`get_infoset()` 返回 `deepcopy`），无独立命令行入口 |
| `eval_data.pkl` | 固定评测牌局：1000 条 `{first, second, third, three_landlord_cards}`，牌为 env 编码 |
| `requirements.txt` | 依赖说明（必需项 + 可选项 + C 扩展/C5 说明） |
| `LICENSE` | 训练端代码的许可协议文本 |
| `douzero/__init__.py`、`douzero/dmc/__init__.py`、`douzero/env/__init__.py`、`douzero/evaluation/__init__.py`、`douzero/radam/__init__.py` | 各包的包标记：`dmc` 导出 `parser`，`env` 导出 `Env`，`radam` 导出 `RAdam/PlainRAdam/AdamW`，其余为空 |
| `baseline/` | 基线权重：`best/`、`test/` 下的 `.ckpt`，以及 `SLModel/`（监督学习叫分/出牌模型，供评测的 `Supervised` 选项使用） |
| `douzero_checkpoints/` | 默认训练输出目录（`--savedir` 默认值），内含各实验子目录 |
| `scripts/train.sh` | 开发/冒烟训练脚本：小模型、小批量、2 actor、2000 帧 |
| `scripts/train_full.sh` | 正式训练脚本：384 维 / 8 头 / 6 层、batch 21 × unroll 100、8 actor、30 亿帧 |
| `douzero/dmc/arguments.py` | 全部命令行参数定义 |
| `douzero/dmc/dmc.py` | DMC 主流程：建共享模型、起 actor 进程、learner 线程、checkpoint、加载/续训 |
| `douzero/dmc/env_utils.py` | 训练环境封装 `Environment`：`initial()`/`step()` 自动重置对局，把观测搬到 GPU |
| `douzero/dmc/models.py` | 模型定义：`GeneralModelBid`、`GeneralModelTransformer`(V1)、`GeneralModelTransformerV2`、`Model` 六模型封装 |
| `douzero/dmc/utils.py` | actor 主循环 `act()`、优化器构造（优化器实现见 `douzero/radam/`）、探索率衰减、actor 端 `_cards2tensor_v2` 编码 |
| `douzero/dmc/file_writer.py` | 训练日志（`out.log`/`logs.csv`/`fields.csv`/`meta.json`） |
| `douzero/env/env.py` | 训练环境 `Env` + 观测编码（`cards2array_v2`、Z/X 组装、叫分观测） |
| `douzero/env/game.py` | 斗地主规则引擎 `GameEnv` 与 `InfoSet`（叫分、出牌、合法动作、春天、炸弹、结算） |
| `douzero/env/move_detector.py` | 牌型识别 `get_move_type()`（15 种牌型） |
| `douzero/env/move_generator.py` | 按手牌生成所有可出牌型 |
| `douzero/env/move_selector.py` | 按对手上一手过滤同型更大的牌 |
| `douzero/env/utils.py` | 牌型常量（`TYPE_0_PASS`…`TYPE_15_WRONG`）与链长下限 |
| `douzero/env/oracle.py` | 神谕 DP：`N_active`/`N_preserve`、被动步数启发式、5 维特征与 3 维 LOR 标签 |
| `douzero/env/oracle_core.c` | 神谕 DP 的 C 实现（编成 DLL 后经 ctypes 调用） |
| `douzero/env/_compile.bat` | Windows/MSVC 编译 `oracle_core.c` 的脚本（`VCVARS` 或 vswhere 定位工具链） |
| `douzero/env/opponent_modeling.py` | 对手建模：C3 OHS/OBP、C4 CTP、C5 SWI/TP，以及 `get_x_dim()` |
| `douzero/env/numba_utils.py` | numba 加速实现与运行时 patch |
| `douzero/evaluation/simulation.py` | `evaluate(...)`：多进程跑固定牌局并汇总 WP/ADP |
| `douzero/evaluation/autosimu.py` | `auto_test.py` 用的 `evaluate(...)`：额外返回指定位置的胜率与得分 |
| `douzero/evaluation/deep_agent.py` | `DeepAgent`（加载 `.ckpt` 出牌/叫分模型）、`SupervisedModel`（baseline 叫分） |
| `douzero/evaluation/random_agent.py` | 随机合法动作对手 |
| `douzero/evaluation/rlcard_agent.py` | rlcard 规则对手（可选依赖） |
| `douzero/radam/radam.py` | RAdam 优化器实现（本项目全部位置使用它） |

---

## 三、快速开始

以下命令默认工作目录是 `python/`（`train.py`、`eval_data.pkl`、`douzero_checkpoints/` 都是相对该目录解析的）；两个训练脚本会自动定位仓库根并 `cd` 到 `python/`，因此也可在仓库根用 `bash python/scripts/train.sh` 调用。

### 3.1 从头训练

```bash
cd python

# 开发/冒烟配置（小模型，几秒到几分钟跑通）
bash scripts/train.sh

# 正式配置（等价于 train_full.sh 拼出的完整参数）
bash scripts/train_full.sh
```

`train_full.sh` 实际执行的等价命令（可据此在别的调度系统里直接调用）：

```bash
cd python
python train.py \
  --xpid DouEnMH_v1.2 --total_frames 3000000000 --num_actors 8 \
  --training_device 0 --gpu_devices 0 --num_actor_devices 1 \
  --save_interval 15 --batch_size 21 --unroll_length 100 \
  --learning_rate 0.0001 --gradient_accumulation_steps 3 \
  --exp_epsilon 0.01 --exp_epsilon_start 0.03 \
  --bid_exp_epsilon 0.05 --bid_exp_epsilon_start 0.3 \
  --epsilon_decay_games 20000 --bid_curriculum_games 500000 --bid_batch_size 8 \
  --oracle_decay_ratio 0.3 --oracle_lambda_landlord 1.0 \
  --oracle_lambda_peasant 0.6 --oracle_lambda_min 0.05 \
  --d_model 384 --nhead 8 --num_encoder_layers 6 --dim_feedforward 1280
```

训练输出在 `douzero_checkpoints/<xpid>/`（`--savedir` 默认 `douzero_checkpoints`）。启动阶段会先做一次 CUDA 自检（两个脚本里的 `$PYTHON -c "import torch; assert torch.cuda.is_available()"`），CUDA 不可用时脚本立即失败。

### 3.2 从 checkpoint 续训

```bash
cd python
bash scripts/train_full.sh --load_model
```

- 续训读取的是 `douzero_checkpoints/<xpid>/model.tar`（包含 6 个模型权重、6 个优化器状态、`stats`、`frames`、`position_frames`、`flags`）；单个位置的 `{position}_{frame}.ckpt` 只是权重，不能用于续训。
- `--xpid` 必须与要恢复的实验一致，否则找不到 `model.tar`。**找不到时 `--load_model` 不会报错，会静默从头训练**（`dmc.py` 的判断条件是 `flags.load_model and os.path.exists(checkpointpath)`），续训前请确认文件存在。
- 模型结构（`--use_v2`、`--d_model`、`--nhead`、`--num_encoder_layers`、`--dim_feedforward`、`DOUENMH_ENABLE_C5`）必须与保存时一致；`missing keys`/`unexpected keys` 型的不匹配默认硬报错，可用 `--allow_partial_load` 让这些位置从零开始（其余位置正常恢复）。注意 `--allow_partial_load` 只豁免 key 集合不匹配；`DOUENMH_ENABLE_C5` 改变会让 X 维度变化，表现为 **size mismatch**，仍会硬报错（见 8.9）。
- `--disable_checkpoint` 可完全关闭保存（包括结束时的最终 checkpoint）。

### 3.3 评测：`evaluate.py`

```bash
cd python
python evaluate.py \
  --player_1_playcard douzero_checkpoints/DouEnMH_v1.2/landlord_611140100.ckpt \
  --player_2_playcard douzero_checkpoints/DouEnMH_v1.2/landlord_down_611140100.ckpt \
  --player_3_playcard douzero_checkpoints/DouEnMH_v1.2/landlord_up_611140100.ckpt \
  --eval_data eval_data.pkl --num_workers 2 --gpu_device 0
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--player_1_bid` / `--player_2_bid` / `--player_3_bid` | `random` | 三个叫分席位：`random`、`Supervised`（baseline SL 模型）或某个 `.ckpt` 路径 |
| `--player_1_playcard` / `--player_2_playcard` / `--player_3_playcard` | 无（必填） | 地主 / 地主下家 / 地主上家的出牌模型路径，或 `random`/`Supervised`/`rlcard` |
| `--eval_data` | `eval_data.pkl` | 固定牌局 pickle（1000 条） |
| `--num_workers` | `2` | 对局进程数，牌局按 worker 均分 |
| `--gpu_device` | `0` | 设置 `CUDA_VISIBLE_DEVICES` |

过程中按局打印 `----- 第N局 -----`（流局额外打印 `Draw`），末尾是汇总统计：`landlord distribution`（first/second/third 做地主的比例）、`WP results`（各位置胜率；`landlord : Farmers`）、`ADP results`（各位置平均回报）、`number of draw`（流局数）。汇总的分母是“分出胜负的局数”（不含流局）。脚本在模块加载时固定随机种子 78419（`random`/`numpy`/`torch`）。

注意：`evaluation/deep_agent.py` 的 `_load_model()` 只按 V1 结构加载出牌模型（读 `state_embed.weight`、层数从 `state_encoder.layers.*` 数、`nhead` 固定 8），因此 `evaluate.py`/`auto_test.py` 不能评测 V2 出牌权重；非 8 头的 V1 权重虽然能加载成功（`nhead` 不影响参数形状），但注意力头切分方式与训练时不同，结果不可信。

### 3.4 自动评测：`auto_test.py`

`auto_test.py` 没有命令行参数，配置写死在 `__main__`：

```python
checkpoint_dir = "douzero_checkpoints/douzero/"
output_csv     = "landlord_eval_out.csv"
output_all_csv = "landlord_eval_out_all.csv"
best_txt       = "landlord_best.txt"
dire           = "landlord/"
position       = 'landlord'
```

使用方式：

```bash
cd python
mkdir -p landlord        # 输出目录必须先存在（CSV 与图片都写到该目录）
python auto_test.py
```

行为：每 60 秒扫描一次 `checkpoint_dir`，取文件名 `{position}_{frame}.ckpt` 中 frame 最大的模型；若该 frame 已在 CSV 里则跳过。对手出牌模型固定取同目录下最新的 `landlord_down` / `landlord_up`（找不到则用 `random`），叫分全部随机；用 `eval_data.pkl` 的 1000 条牌局、2 个 worker 跑一局评测。每次新模型会：

- 往 `landlord/landlord_eval_out.csv` 追加一行（只保留最后 200 行）并全量追加到 `landlord/landlord_eval_out_all.csv`；
- 更新 `landlord/landlord_best.txt`（win_rate 最高与 score 最高的 model_id）；
- 重画 `landlord/win_rate.jpg`、`landlord/score.jpg`。

该脚本是常驻循环，需要 pandas 与 matplotlib。

### 3.5 单局推理：`infer.py`

```bash
cd python

# 出牌推理（地主）
python infer.py --position landlord \
  --model douzero_checkpoints/DouEnMH_v1.2/landlord_611140100.ckpt \
  --hand "3,3,5,6,7,7,7,9,10,10,11,12,13,14,17,20,30" --history "[]" --top-k 3

# 叫分推理
python infer.py --position first \
  --model douzero_checkpoints/DouEnMH_v1.2/first_611140100.ckpt \
  --hand "3,4,5,6,7,8,9,10,11,12,13,14,17,20,30,3,4" \
  --bid-info "0,0,0"
```

参数：

| 参数 | 默认 | 说明 |
|------|------|------|
| `--position` | 必填 | `first/second/third/landlord/landlord_up/landlord_down` |
| `--model` | 必填 | `.ckpt` 路径；叫分位置按 `model_dict` 建 `GeneralModelBid`；出牌位置按权重 key 自动判断 V1/V2 并从权重推断 `d_model`、层数、`dim_feedforward` |
| `--hand` | 必填 | 手牌（env 编码，逗号分隔，如 `3,3,5,20`） |
| `--history` | 空 | 最近出牌序列 `[[role, [cards]], ...]`（env 编码） |
| `--actions` | 空 | 完整出牌序列，格式同 `card_play_action_seq`；给出时覆盖 `--history`（历史特征取最近 35 手，只给最近两手会得到不同特征） |
| `--legal` | 空 | 显式指定候选动作（JSON 数组）；跟牌局面建议传，否则按“自由领出”枚举全部可出牌型并告警 |
| `--other-hand` / `--played-landlord` / `--played-up` / `--played-down` / `--three-cards` | 空 | 对手手牌并集、三家累计已出牌、三张底牌 |
| `--bid-info` | `0,0,0` | 三家叫分值，**必须是 3 个整数**（叫分模型的 X 是 3 维） |
| `--bomb-num` | `0` | 已出炸弹数 |
| `--cards-left-landlord` / `--cards-left-up` / `--cards-left-down` | `20/17/17` | 三家剩余牌数 |
| `--spring` | 关 | 是否春天 |
| `--top-k` | `5` | 输出前 K 个动作 |
| `--device` | `cpu` | `cpu`/`cuda`（CUDA 不可用时自动回落 CPU） |
| `--use-v2` | 关 | 强制按 V2 结构加载（通常不需要，权重 key 会自动识别） |

输出为按 Q 值降序的 top-k 表与最优动作。注意 `infer.py` 无法从权重推断 `nhead`，固定按 8 处理（源码 `getattr(args, 'nhead', None) or 8`），非 8 头的出牌模型请勿用它。

---

## 四、模型架构

6 个位置各有一个独立模型（`douzero/dmc/models.py` 的 `model_dict` / `model_dict_v2` 与 `Model` 封装）：

| 位置 | 模型类 | 输入 | 输出 |
|------|--------|------|------|
| `first` / `second` / `third`（叫分） | `GeneralModelBid` | `z=(B,2,120)`、`x=(B,3)` | 3 个叫分动作的 Q |
| `landlord` / `landlord_up` / `landlord_down`（出牌） | `GeneralModelTransformer`(V1) 或 `GeneralModelTransformerV2`（`--use_v2`） | V1：`z=(B,41,120)`、`x=(B,52/56)`；V2：`state_z=(1,5,120)`、`action_z=(N,1,120)`、`history_z=(1,35,120)`、`x=(N,52/56)` | 6 个头 |

### 4.1 叫分模型 `GeneralModelBid`

ResNet 风格 1D 卷积 + SE 通道注意力 + MLP：

- 输入 `z` 是 2 行 120 维：第 0 行是叫分动作的广播（120 维都填叫分值），第 1 行是手牌 8×15 矩阵；`x` 是 3 维原始叫分值。
- `BasicBlockM`：`Conv1d(kernel=3)+BN+Mish` → `Conv1d+BN` → SE 通道注意力 → 残差 → Mish。
- 3 级下采样（4→8→16 通道，长度 120→60→30→15），展平 16×15=240 维，与 `x`(3) 拼接成 243。
- MLP：243→256→256→128→3，最后拆成 `win_rate / win / lose` 三头；`win_rate=tanh(...)`，`Q = w̃·win + (1-w̃)·lose`，`w̃=(win_rate+1)/2`。
- 训练时用 `return_value=True` 只拿三头；推理时按 Q 取 argmax，并按 `flags.bid_exp_epsilon` 做 ε-greedy（探索率由 actor 按局数衰减后写回 `flags`）。

### 4.2 出牌模型 V1 `GeneralModelTransformer`（默认）

`--use_v2` 未开启时使用。双塔 Transformer：

- **State 塔**：取 `z` 的第 0–5 行（1 个候选动作行 + 5 个静态卡牌行）共 6 个 token，经 `Conv1d(120→d_model, kernel=1)` 逐点投影后送 `num_encoder_layers` 层（默认 6）`TransformerEncoder`，无位置编码（无序），对 6 个 token 做均值池化。
- **Action 塔**：取 `z` 的第 6–40 行（35 手出牌历史），同样 `Conv1d` 投影后加正弦位置编码，送 `TransformerEncoder`，均值池化。
- 拼接 `[state(d_model), action(d_model), x(X_dim)]` → 1024 → 512 → 128 → 6 头。
- 默认 `d_model=256`、`nhead=8`、`num_encoder_layers=6`、`dim_feedforward=1024`（`arguments.py` 默认值；`train_full.sh` 覆盖为 384/8/6/1280）。

### 4.3 出牌模型 V2 `GeneralModelTransformerV2`（`--use_v2`）

Cross-Attention 结构，把 V1 中“每个候选动作重复编码一次 state/history”改为只编码一次：

- `state_proj`：`state_z=(1,5,120)` 投影成 5 个 token，经 state encoder（层数 = `num_encoder_layers`）。
- `action_proj`：`action_z=(N,1,120)` 每个候选动作投影成 1 个 token，经 2 层轻量 Transformer（单 token 时自注意力等价于 FFN）。
- `cross_attn`：单层 `MultiheadAttention`，Q=动作 token，K/V=state token（动作 attend 状态）。
- `history_proj`：`history_z=(1,35,120)` 投影 + 位置编码，经 history encoder，再对 35 个 token 均值池化。
- 拼接 `[action_aware(d_model), history(d_model), x(X_dim)]` → 1024 → 512 → 128 → 6 头。

注意 V2 的 cross-attention 只让动作 attend 静态状态，历史塔输出是均值池化后直接拼接（不参与 attention）。

### 4.4 6 个头与 Q 值

出牌模型最后一层输出 6 个数，顺序固定：

| 头 | 含义 | 训练目标 |
|----|------|----------|
| `LWR` | 预测胜率（tanh 后 ∈[-1,1]） | `target_wp`（赢 +1 / 输 -1 / 流局 0） |
| `LPW` | 胜局时的 ADP 头 | `target_adp`（只在该分支的权重下回归） |
| `LPL` | 负局时的 ADP 头 | 同上 |
| `LOR_a` | 神谕辅助头：主动步数 `N_active` | LOR 标签第 0 维 |
| `LOR_p` | 神谕辅助头：被动步数 `N_passive` | LOR 标签第 1 维 |
| `LOR_d` | 神谕辅助头：控牌敏感度 `ΔN` | LOR 标签第 2 维 |

Q 值公式（模型前向与 C++ 推理端一致）：

```
w̃  = (tanh(LWR) + 1) / 2
Q  = w̃ · LPW + (1 - w̃) · LPL
```

`LOR_a / LOR_p / LOR_d` **只进入训练损失，不进入 Q**；动作选择只用 3 个主头的组合值做 argmax。

### 4.5 默认超参

以 `scripts/train_full.sh` 为准：

| 项 | 正式训练（train_full.sh） | 开发/冒烟（train.sh） | 含义 |
|----|--------------------------|------------------------|------|
| `xpid` | `DouEnMH_v1.2` | `douEnMH_v1.0` | 实验名 = 输出子目录名 |
| `total_frames` | `3000000000` | `2000` | 总训练帧数（6 个位置合计） |
| `num_actors` | `8` | `2` | 每个 actor 设备上的进程数 |
| `num_actor_devices` | `1` | `1` | actor 设备数（脚本固定传 1） |
| `save_interval` | `15`（分钟） | `10`（分钟） | checkpoint 保存周期 |
| `batch_size` | `21` | `2` | Learner 出牌批次大小（以 unroll 段为单位） |
| `bid_batch_size` | `8` | 未传，用默认 `8` | 叫分位置专用 batch |
| `unroll_length` | `100` | `5` | 每段轨迹长度 |
| `learning_rate` | `0.0001` | `0.0003` | RAdam 学习率 |
| `gradient_accumulation_steps` | `3` | `1` | 梯度累积步数 |
| `exp_epsilon` | `0.01` | `0.3` | 出牌探索率终值 |
| `exp_epsilon_start` | `0.03` | 未传（默认 0.05） | 出牌探索率初值 |
| `bid_exp_epsilon` | `0.05` | `0.3` | 叫分探索率终值 |
| `bid_exp_epsilon_start` | `0.3` | 未传（默认 0.5） | 叫分探索率初值 |
| `epsilon_decay_games` | `20000` | 未传（默认 20000） | 每 actor 的衰减局数 |
| `bid_curriculum_games` | `500000` | `5000` | P0 叫 0 分偏置衰减局数 |
| `oracle_decay_ratio` | `0.3` | 未传（默认 0.2） | 神谕 λ 衰减覆盖的训练帧比例 |
| `oracle_lambda_landlord` | `1.0` | 未传（默认 1.0） | `LOR_a` λ 初值 |
| `oracle_lambda_peasant` | `0.6` | 未传（默认 0.6） | `LOR_p`/`LOR_d` λ 初值 |
| `oracle_lambda_min` | `0.05` | 未传（默认 0.03） | λ 下限 |
| `d_model` | `384` | `16` | Transformer 维度 |
| `nhead` | `8` | `4` | 注意力头数 |
| `num_encoder_layers` | `6` | `4` | Encoder 层数 |
| `dim_feedforward` | `1280` | `256` | FFN 维度 |
| `actor_device_cpu` | 不传（GPU 推理） | 不传 | `ACTOR_MODE=cpu` 时传 |
| `log_game_steps` | `0`（关闭） | `0` | 每局步数日志 |

未在脚本中出现的参数（如 `--decay`、`--max_grad_norm`、`--epsilon`、`--use_v2`）取 `arguments.py` 默认值。

---

## 五、训练流程

### 5.1 进程与数据流

`douzero/dmc/dmc.py` 的 `train(flags)`：

1. **建共享模型**：对每个 actor 设备 `Model(device, ...)`，六个子模型 `share_memory()` 后 `eval()`；`torch.multiprocessing` 的共享策略设为 `file_system`。模型在 GPU 上时每个模型单独 `.to(cuda)` 并清缓存，避免 6 个模型的拷贝峰值叠加。
2. **建优化器**：`create_optimizers()` 为 6 个位置各建一个 RAdam（见 5.3）。
3. **建队列**：每个设备 × 6 个位置各一个 `ctx.Queue()`；`ctx = mp.get_context('spawn')`，`stop_event = ctx.Event()`。
4. **起 learner 线程**：`batch_and_learn()` 在主进程里轮询所有位置队列（见 5.2）。
5. **起 actor 进程**：`ctx.Process(target=act, args=(i, device, batch_queues[device], models[device], flags, stop_event))`，每个 actor 间隔 0.5s 启动，避免 CUDA 上下文初始化并发峰值。
6. **主循环**：每 5 秒检查一次 `frames < total_frames`、是否到 `save_interval`、GPU 分配率是否 >90%（紧急 `empty_cache`），并打 FPS/各位置 FPS/统计日志。
7. **收尾**：`stop_event.set()` → learner 线程退出 → 写最终 checkpoint → join/terminate actor → `torch.cuda.ipc_collect()`。

### 5.2 actor 与 learner

**actor（`douzero/dmc/utils.py::act`）**：

- 创建 `Env`（规则引擎包装）与 `Environment`（自动 reset 的 wrapper），预热一遍 CUDA 上下文（按 V1/V2 输入形状各跑一次 dummy forward）。
- 循环：若合法动作 >1 个，用 `model.forward(position, obs['z_batch'], obs['x_batch'], flags=flags, ...)` 得到动作索引（模型内部按 `flags.exp_epsilon` 做 ε-greedy），否则取唯一动作；再套叫分课程（5.4）。
- 把“动作卡牌矩阵 + 观测 Z 矩阵”拼成该位置的 `obs_z`，取 `obs_x_no_action` 后按实际执行动作注入 C2 神谕段（X 偏移 29），并记录该动作的 3 维 LOR 标签；按位置累积到缓冲区。每位置累计超过 `unroll_length` 条就打包成一段（`done/episode_return/target_adp/target_wp/target_lor/obs_z/obs_x_batch`）放进该位置队列。
- 一局结束时：把该局回报回填到各位置尚未结算的样本（`target_adp`、`target_wp`；`--decay` 默认 1.0，即整局共用同一回报），完成标记 `done=True`。流局回报为 0、WP 回报为 0。

**learner（`dmc.py::batch_and_learn` + `learn`）**：

- 先把 6 个位置队列里的段全部非阻塞（`get(timeout=0.01)`）排空到内存缓冲区，再按缓冲区长度从大到小依次处理。
- 每个位置每次最多训练 `K = gradient_accumulation_steps` 个 batch：出牌位置用 `--batch_size`、叫分位置用 `--bid_batch_size` 个 unroll 段，`torch.cat` 成一个 batch。
- `learn()`：前向 → 三项主损失 + 三项 LOR 损失 → `loss = loss / accum_total` → `backward()`；只有累积的最后一步做 `clip_grad_norm_(max_grad_norm)` + `optimizer.step()` + `zero_grad(set_to_none=True)`。
- 记账：`frames += unroll_length * batch_size * 本次训练的 batch 数`（六位置合计），同时累加 `position_frames[position]`；每次训练后写一行日志（`frames` + 12 个统计项）。因此 `frames` 可能略超 `total_frames`（主循环每 5 秒才检查一次）。
- 主损失（`compute_loss` 为 MSE）：

```
loss1 = MSE(LWR, target_wp)
w̃     = (tanh(LWR)+1)/2
loss2 = MSE(w̃·LPW, target_adp)
loss3 = MSE((1-w̃)·LPL, target_adp)
loss  = loss1 + loss2 + loss3 + (LOR 三项加权和)
```

- 每局回报 `episode_return` 由 `douzero/env/env.py` 的 `_get_reward*` 给出：底分 × 炸弹/春天倍数除以 24，地主一侧再除以 2，使地主与农民回报量级对称；`target_wp` 只取胜负符号（±1）。

### 5.3 优化器（含 RAdam 何时用）

- 6 个位置**全部使用 RAdam**（`douzero/radam/radam.py`，`create_optimizers`），没有任何位置用 Adam/AdamW/SGD；`--learning_rate` 与 `--epsilon` 直接传给 RAdam（`betas=(0.9,0.999)`、`weight_decay=0` 为 RAdam 默认值）。
- 每个位置一个独立优化器，状态随 `model.tar` 保存/恢复；学习率不随训练进度调度（只有探索率、叫分课程、神谕 λ 在调度）。

### 5.4 叫分课程

目的：训练早期让 P0（`first`）大概率叫 0 分，把叫分机会让给 P1/P2，避免叫分模型一开始就抢占地主。

- 代码在 actor 里，与模型输出解耦：当 `bid_curriculum_games > 0`、位置是 `first`、且该 actor 已完成局数 `game_count < bid_curriculum_games` 时，以 `p_zero = 1 - game_count / bid_curriculum_games` 的概率把动作强制改成合法动作里的 `[0]`（不叫）。
- 概率随局数线性衰减到 0，超过局数后不再干预；`bid_curriculum_games=0` 关闭。
- 课程按 **actor 本地 `game_count`** 计算（不是全局帧数），多 actor 时各进程独立衰减。

### 5.5 ε 衰减（出牌与叫分分别衰减）

- 每个 actor 用本地 `game_count`（完成局数）线性衰减：`ε = start + (end - start) * game_count / epsilon_decay_games`，`game_count >= epsilon_decay_games` 或 `epsilon_decay_games=0` 时取 end。
- 衰减结果写回该 actor 的 `flags.exp_epsilon`（出牌）与 `flags.bid_exp_epsilon`（叫分），模型前向的 ε-greedy 直接读这两个值。
- 代码在循环外先固定 `eps_end = flags.exp_epsilon` / `bid_eps_end = flags.bid_exp_epsilon`，否则衰减结果会覆盖终点值、退化为不动点。

### 5.6 神谕 λ 衰减与 LOR 辅助损失

- LOR 监督标签来自 `oracle.compute_oracle_labels_from_raw()`：`N_active`、对手感知的 `N_passive`、`ΔN`，均做饱和归一化（`x/(x+k)`，k 分别为 10/15/8）。出牌阶段逐动作计算（出牌后剩余手牌），actor 取实际执行动作的标签。
- λ 线性衰减（`_get_oracle_lambda`）：`λ(t) = max(λ_min, λ_init · (1 - t / T_decay))`，其中 `T_decay = int(total_frames * oracle_decay_ratio)`，`t` 是当前全局 `frames`。
- 三个头分别加权：`LOR_a` 用 `oracle_lambda_landlord`；`LOR_p` 与 `LOR_d` 都用 `oracle_lambda_peasant`；下限统一为 `oracle_lambda_min`。
- LOR 损失只加在出牌位置（V1/V2 模型有 oracle 头）；叫分模型没有 oracle 头，`result` 里没有 `'oracle'`，跳过。

### 5.7 各位置采样与统计

- 6 个位置各自有独立模型、独立优化器、独立队列、独立统计；一局内 3 个叫分席位各只出 1 次叫分动作，而 3 个出牌席位会出很多手，因此**叫分位置的数据量远小于出牌位置**，learner 用更小的 `bid_batch_size` 且按队列长度排序处理，避免叫分数据长期积压。
- 日志统计项（`log` 到 `out.log` / `logs.csv`）：`frames` 与 `mean_episode_return_{position}`、`loss_{position}`（6 个位置各两项）。`mean_episode_return` 的算法是：每次学习步把该 batch 中已结束 episode 的回报取平均后压入长度 50 的 `deque`，统计值再取这个 deque 的均值（即最近至多 50 个学习步的平均回报）。
- 每 5 秒打印一次全局 FPS、最近 24 个采样点的平均 FPS、三个出牌位置的 FPS 和完整 stats。
- 训练目录下的日志文件：`out.log`（文本日志）、`logs.csv`（每 batch 一行，宽表自动扩列）、`fields.csv`（列名）、`meta.json`（参数元数据）。

---

## 六、`arguments.py` 参数表

共 37 个参数（`python/douzero/dmc/arguments.py`），按源码分节列出。类型与默认值照抄源码。

### 6.1 General Settings

| 参数 | 类型 / 默认 | 说明 |
|------|-------------|------|
| `--xpid` | str / `douzero` | 实验 id，也是 `{savedir}/{xpid}/` 子目录名 |
| `--save_interval` | int / `10` | 每隔多少**分钟**保存一次 checkpoint |
| `--objective` | str / `adp`（只能选 `adp`） | 训练目标，固定用 ADP |

### 6.2 Training Settings

| 参数 | 类型 / 默认 | 说明 |
|------|-------------|------|
| `--actor_device_cpu` | flag / 关 | actor 推理放 CPU（GPU 只做训练） |
| `--gpu_devices` | str / `'0'` | 设置 `CUDA_VISIBLE_DEVICES`（逗号分隔） |
| `--num_actor_devices` | int / `1` | actor 使用的设备数，不能超过 `--gpu_devices` 的个数 |
| `--num_actors` | int / `4` | 每个 actor 设备上的进程数 |
| `--training_device` | str / `'0'` | Learner 所在 GPU 编号；`cpu` 表示用 CPU 训练 |
| `--load_model` | flag / 关 | 从 `{savedir}/{xpid}/model.tar` 恢复训练 |
| `--allow_partial_load` | flag / 关 | 续训时允许部分位置结构不匹配（这些位置从零开始），默认硬报错 |
| `--disable_checkpoint` | flag / 关 | 不保存任何 checkpoint |
| `--savedir` | str / `douzero_checkpoints` | 实验根目录 |

### 6.3 Hyperparameters

| 参数 | 类型 / 默认 | 说明 |
|------|-------------|------|
| `--total_frames` | int / `100000000000` | 总训练帧数（6 个位置合计） |
| `--exp_epsilon` | float / `0.01` | 出牌探索率终值（开衰减时是终点） |
| `--exp_epsilon_start` | float / `0.05` | 出牌探索率初值 |
| `--bid_exp_epsilon` | float / `0.3` | 叫分探索率终值 |
| `--bid_exp_epsilon_start` | float / `0.5` | 叫分探索率初值 |
| `--epsilon_decay_games` | int / `20000` | 每 actor 用多少局把探索率从初值衰减到终值（0=不衰减） |
| `--bid_curriculum_games` | int / `0` | P0 叫 0 分偏置从 1.0 线性衰减到 0 的局数（0=关闭） |
| `--decay` | float / `1.` | MC 回报的逐步折扣系数（1.0=整局共用同一回报） |
| `--batch_size` | int / `30` | Learner 出牌批次大小（以 unroll 段为单位） |
| `--bid_batch_size` | int / `8` | 叫分位置的 Learner 批次大小 |
| `--unroll_length` | int / `100` | 轨迹展开长度 |
| `--max_grad_norm` | float / `40.` | 梯度裁剪阈值 |
| `--gradient_accumulation_steps` | int / `1` | 累积多少个 batch 才做一次优化器更新（1=不累积） |

### 6.4 Oracle 辅助头

| 参数 | 类型 / 默认 | 说明 |
|------|-------------|------|
| `--oracle_decay_ratio` | float / `0.2` | 神谕 λ 衰减覆盖的训练帧比例（如 0.2 = 前 20%） |
| `--oracle_lambda_landlord` | float / `1.0` | `LOR_a`（`N_active`）λ 初值 |
| `--oracle_lambda_peasant` | float / `0.6` | `LOR_p`（`N_passive`）与 `LOR_d`（`ΔN`）λ 初值 |
| `--oracle_lambda_min` | float / `0.03` | λ 下限 |

### 6.5 Model Architecture

| 参数 | 类型 / 默认 | 说明 |
|------|-------------|------|
| `--use_v2` | flag / 关 | 使用 V2 Cross-Attention 出牌模型（否则 V1 双塔） |
| `--d_model` | int / `256` | Transformer 嵌入维度 |
| `--nhead` | int / `8` | 注意力头数 |
| `--num_encoder_layers` | int / `6` | Encoder 层数 |
| `--dim_feedforward` | int / `1024` | Transformer FFN 隐藏维度 |

### 6.6 Logging

| 参数 | 类型 / 默认 | 说明 |
|------|-------------|------|
| `--log_game_steps` | flag / 关 | 每局结束时打印步数与吞吐（多 actor 时日志很吵） |

### 6.7 Optimizer

| 参数 | 类型 / 默认 | 说明 |
|------|-------------|------|
| `--learning_rate` | float / `0.0001` | RAdam 学习率 |
| `--epsilon` | float / `1e-8` | RAdam 的 eps（不是探索率） |

---

## 七、checkpoint 格式与命名

### 7.1 目录与文件

训练输出目录为 `{savedir}/{xpid}/`（默认 `douzero_checkpoints/<xpid>/`），每次保存产生：

| 文件 | 内容 |
|------|------|
| `model.tar` | 续训用：`model_state_dict`（6 个位置各一份）、`optimizer_state_dict`（6 个位置各一份）、`stats`、`flags`（完整参数 dict）、`frames`、`position_frames` |
| `{position}_{frame}.ckpt` | 6 个文件，文件名如 `landlord_611140100.ckpt`、`landlord_down_...`、`landlord_up_...`、`first_...`、`second_...`、`third_...`；内容是该位置的纯 `state_dict`（`torch.save(model.state_dict())`），供评测/导出使用 |
| `out.log` / `logs.csv` / `fields.csv` / `meta.json` | 训练日志与参数元数据（`FileWriter` 建目录时即创建） |

命名规则：`{position}_{frame}.ckpt`，`frame` 是保存时刻的**全局帧数**（6 个位置合计，见 5.2），不是该位置的帧数；同一次保存的 6 个文件 frame 相同。

### 7.2 保存时机与实现

- 主循环每 5 秒检查一次 `timer() - last_checkpoint_time > save_interval * 60`，到点保存；训练正常结束时（含 `total_frames` 跑满或 `KeyboardInterrupt`）再写一次最终 checkpoint。
- 保存前会给 learner 加锁（最多等 30 秒），防止保存与反向传播并发造成显存尖峰；最终 checkpoint 时 learner 已停止，无需加锁。
- 保存时先把所有 `state_dict` 张量递归搬到 CPU 再 `torch.save`，避免 GPU 端再分配约一份模型大小的显存；`torch.save(..., _use_new_zipfile_serialization=False)`。
- `--disable_checkpoint` 时 `checkpoint()` 直接返回（连最终 checkpoint 也不写）。

### 7.3 加载逻辑

`--load_model` 且 `model.tar` 存在时：

1. `torch.load(..., map_location='cpu')` 读入；
2. 逐个位置 `load_state_dict` 模型权重 + 对应优化器状态；
3. 若某位置抛 `RuntimeError` 且消息含 `missing keys`/`unexpected keys`：默认立即报错，报错信息会提示常见原因（`--use_v2` 切换、`d_model/nhead/num_encoder_layers/dim_feedforward` 改动、`DOUENMH_ENABLE_C5` 与训练时不同）；加 `--allow_partial_load` 则该位置跳过恢复、从零训练，其余位置正常恢复。其它 `RuntimeError`（例如维度变化的 `size mismatch`）不在此豁免范围内，会直接抛出；
4. 恢复 `stats`、`frames`、`position_frames` 后继续训练。

---

## 八、常见问题

### 8.1 神谕 C 扩展编译

Windows（cmd.exe）：

```bat
cd python\douzero\env
_compile.bat
```

`_compile.bat` 的定位顺序：设置了 `VCVARS` 环境变量则直接 `call "%VCVARS%"`；否则用 `%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe` 找到带 C++ 工具链的 Visual Studio，再调用其 `VC\Auxiliary\Build\vcvars64.bat`；都找不到时打印安装提示并以非 0 退出。成功后同目录出现 `oracle_core.dll`（以及 `.obj`/`.lib`/`.exp` 中间文件，可删）。

Linux（GCC，C 源码头部注释给出的命令）：

```bash
cd python/douzero/env
gcc -O3 -shared -fPIC -std=c11 oracle_core.c -o oracle_core.dll
```

注意 `oracle.py` 固定按文件名 `oracle_core.dll` 查找并 `ctypes.CDLL` 加载；Linux 上把产物命名为 `oracle_core.dll` 即可（ctypes 不校验扩展名）。

验证是否生效：不设置 `ORACLE_PLACEHOLDER` 跑一局训练，若首帧 DP 明显很快说明 DLL 已加载；或者直接 `python -c "from douzero.env import oracle; print(oracle._oracle_c_lib)"`（非 `None` 为已加载，路径需在 `python/` 下执行）。

### 8.2 numba 不可用

`numba_utils.py` 顶部 `try: from numba import njit except ImportError: _HAS_NUMBA=False`；不可用时只是不 patch，编码函数用纯 Python/NumPy 实现，训练结果不受影响，但 actor 端 CPU 编码开销变大。设 `Numba_DISABLE=1` 可强制关闭 patch（用于排查 numba 引起的问题）；numba 首次导入会编译并写 `__pycache__`，多进程启动时会有一点编译开销。

### 8.3 显存不足 / OOM

按影响从大到小调这些参数：

- `--batch_size`、`--unroll_length`：直接决定单次 forward/backward 的张量规模。
- `--gradient_accumulation_steps`：梯度在原地累加，**不增加**显存占用，只降低优化器更新频率（可用来在小显存下等效增大有效 batch）。
- `--d_model`、`--nhead`、`--num_encoder_layers`、`--dim_feedforward`：模型容量与激活规模。
- `--bid_batch_size`：叫分模型较小，一般不是瓶颈。
- `--actor_device_cpu`：actor 推理移到 CPU，GPU 只留 Learner（GPU 显存大幅下降，但 CPU 需要扛住游戏逻辑与编码）。
- `--num_actors`：actor 进程各自持有独立的 CUDA allocator 与推理激活，增多会抬高整体显存占用。
- 环境变量 `PYTORCH_CUDA_ALLOC_CONF`（脚本已设置 `max_split_size_mb:128,roundup_power2_divisions:16`，Linux 另加 `expandable_segments:True`）用于抑制碎片；主循环在 GPU 分配率 >90% 时才做一次紧急 `empty_cache()`，不要靠频繁清缓存解决问题。

### 8.4 多 actor / 多设备

- 单卡最常见的配置就是脚本里的 `--num_actor_devices 1 --num_actors 8`：1 个 GPU 上跑多个 actor 进程 + 1 个 learner，靠 CUDA 时分复用。
- `--num_actor_devices` 必须是 `--gpu_devices` 列表长度的子集；但当前实现只为**第 0 个 actor 设备上的模型**建优化器并更新权重，其余设备上的模型副本不会被 Learner 更新（actor 用各自设备上的旧权重采样），因此实际训练请保持 `num_actor_devices=1`。
- actor 数过多时，队列积压会让内存上涨、CPU 编码成为瓶颈；可用 `--log_game_steps` 观察每局步数/吞吐，或降低 `num_actors`、增大 `batch_size` 提高 Learner 消费能力。

### 8.5 CUDA 不可用

`train(flags)` 在“非 CPU actor 或非 CPU 训练”时检查 `torch.cuda.is_available()`，失败会抛：

```
CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`.
Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`
```

纯 CPU 训练用：

```bash
cd python
python train.py --actor_device_cpu --training_device cpu --num_actors 1 \
  --batch_size 1 --unroll_length 2 --total_frames 1000
```

（两个训练脚本头部会做 CUDA 断言，纯 CPU 场景请直接调用 `train.py`。）

### 8.6 续训“没有生效”

`--load_model` 找不到 `{savedir}/{xpid}/model.tar` 时静默从头训练（不报错）。先确认：`--xpid` 与保存时一致、`--savedir` 一致（默认相对 `python/` 解析）、目录里有 `model.tar`；只有 per-position `.ckpt` 是不够的。日志里出现 `Resuming preempted job, current stats:` 才表示真的恢复了。

### 8.7 续训报“架构不匹配”

分两类：

- `missing keys` / `unexpected keys`（例如改用/去掉了 `--use_v2`、`d_model/nhead/num_encoder_layers/dim_feedforward` 改动）：默认硬报错，加 `--allow_partial_load` 可让这些位置从零开始。
- `size mismatch`（例如 `DOUENMH_ENABLE_C5` 与保存时不同，X 从 56 变 52）：同样硬报错，但 `--allow_partial_load` **不豁免**；需要改回一致的 `DOUENMH_ENABLE_C5`，或把这些位置的旧权重移走后重新训练。

### 8.8 `evaluate.py` / `auto_test.py` 的模型限制

两者通过 `douzero/evaluation/deep_agent.py` 加载权重：出牌模型按 **V1 结构**（`state_embed`）读 `d_model`、层数与 FFN 维度，`nhead` 硬编码为 8；叫分模型走 `model_dict`（`GeneralModelBid`）。因此：

- V2 出牌 `.ckpt` 会因找不到 `state_embed.weight` 报错；
- 非 8 头的 V1 权重不会报错（`nhead` 不影响参数形状），但按 8 头切分会得到与训练时不同的注意力计算结果，属于静默错误，不要用这两个脚本评测这类权重；
- 评测其他架构请用 `infer.py`（它会从权重推断 V1/V2 与维度，但同样无法推断 `nhead`）。

### 8.9 特征维度与 C5 开关不一致

`DOUENMH_ENABLE_C5` 在**模块导入时**读取一次（`opponent_modeling.ENABLE_C5`），影响 `get_x_dim()` 与模型第一层线性层的输入维度。训练、续训、导出、C++ 推理必须使用同一取值：改开关后旧 checkpoint 会因 X 维度变化而加载失败（`size mismatch`，属于硬报错，`--allow_partial_load` 不豁免，见 8.7）。
