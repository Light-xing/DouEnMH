# 模型导出：PyTorch → TorchScript → LibTorch 1.5.0

本文件说明如何把训练端产出的 6 个 checkpoint 转成 Botzone 上 LibTorch 1.5.0 能加载的 TorchScript 模型，包括 PyTorch 1.5 兼容层、架构自动检测、导出命令与输出布局，以及 FP16 精度验证。

## 一、为什么需要导出

训练端与部署端的 PyTorch 运行时不是同一个版本：

| 环节 | 运行时 | 约束 |
|------|--------|------|
| 训练（`python/`） | PyTorch 2.x | 可以用 `nn.Mish`、`batch_first=True` 等新 API，模型是普通 `nn.Module` |
| 部署（`cpp/`） | LibTorch 1.5.0 C++（Botzone 只提供这一版） | 只能加载与 1.5.0 算子/字节码兼容的 TorchScript |

TorchScript 的向后兼容只保证"旧版本导出的模型能被新版本运行时加载"，不保证"新版本导出的模型能被旧版运行时加载"。PyTorch 2.x 的导出结果会带上 2.x 的算子与序列化信息，交给 LibTorch 1.5.0 加载不能保证成功。因此整条链路是：

```
python/douzero_checkpoints/<XPID>/{position}_{frame}.ckpt
        │  （PyTorch 2.x 训练产物，每个位置一个纯 state_dict）
        ▼
tools/export/  在 Python 3.8 + PyTorch 1.5.0 环境里
        │  1. 用 compat_models_douenmh.py 重新实例化结构等价的模型
        │  2. load_state_dict(checkpoint)
        │  3. torch.jit.trace / torch.jit.script
        ▼
tools/export/export_model/<suffix>/{position}_jit.pt（+ _jit_fp16.pt）
        │  复制 / --copy
        ▼
cpp/build/data/models/*.pt  （Botzone 上为 User Storage 的 data/models/）
```

关键点：**导出必须在 PyTorch 1.5.0 环境里完成**。训练端直接 `torch.jit.trace` 一个 2.x 模型再拿到 C++ 端加载，属于跨版本使用，不在兼容承诺内；此外 2.x 模型用到的 `nn.Mish`、`batch_first=True` 在 1.5.0 里并不存在，连模型都实例化不出来，所以还需要兼容层（见下一节）。

## 二、兼容层 `compat_models_douenmh.py`

`tools/export/compat_models_douenmh.py` 手工重写了训练端 `python/douzero/dmc/models.py` 的 V1 结构，只替换 1.5.0 不支持的写法，模块层级、参数名和形状保持不变。

### 2.1 改动清单

| 问题 | 训练端（PyTorch 2.x） | 兼容版（PyTorch 1.5.0） |
|------|----------------------|------------------------|
| Mish 激活 | `nn.Mish()`（1.9+ 才有） | 函数 `mish(x) = x * tanh(softplus(x))`，在 `BasicBlockM.forward` 中调用 |
| Transformer 的 `batch_first` | `TransformerEncoderLayer(..., batch_first=True)`（1.9+ 才有） | 1.5 只接受 `(S, B, E)`，在编码器前后手动 `transpose(0, 1)` |
| ChannelAttention | `AdaptiveAvgPool1d` + 两层 FC + Sigmoid | 1.5 已支持，无需修改 |
| `state_dict` 键 | — | 模块层级与参数名完全一致，可直接 `load_state_dict`（strict） |
| `forward` 签名 | 带 `return_value` / `flags` 分支，返回 dict | 只保留推理分支，直接返回 Q 张量，便于 trace/script |
| 架构覆盖范围 | `GeneralModelTransformer`（V1）与 `GeneralModelTransformerV2` | 只包含 V1 与叫分模型；V2 不包含 |

`state_dict` 一致性可以离线核对：同一组架构参数下，兼容模型与训练模型的键名逐项相同（默认 6 层出牌模型 157 个键、叫分模型 152 个键）。因此导出时不需要重命名/映射任何权重。

### 2.2 叫分模型 `GeneralModelBid`

固定结构，没有可调架构参数：

```
z (B, 2, 120) ─ layer1: BasicBlockM×3,  2→4,   stride=2 ─ (B, 4, 60)
                layer2: BasicBlockM×3,  4→8,   stride=2 ─ (B, 8, 30)
                layer3: BasicBlockM×3,  8→16,  stride=2 ─ (B, 16, 15)
                flatten(1,2) ─ (B, 240)
                concat x (B, 3) ─ (B, 243)
                Linear 243→256→256→128→3
                3 头 → tanh/加权组合 → Q (B, 1)
```

- 每个 `BasicBlockM` 是 `Conv1d(3×3) → BN → mish → Conv1d(3×3) → BN → SE → 残差 → mish`。
- `x` 固定 3 维（`bid_info`），所以在兼容模型里写死，不需要从 checkpoint 推断。
- 输入 `z` 的 2 行分别是候选叫分动作广播、自己的手牌编码。

### 2.3 出牌模型 `GeneralModelTransformer`（V1）

```
src1 (B, 41, 120) ─┬─ rows 0-5  (1 动作行 + 5 静态行): Conv1d 120→d_model,
                   │             TransformerEncoder ×L, mean pool ─ (B, d_model)
                   └─ rows 6-40 (35 行动作历史): Conv1d 120→d_model,
                                 位置编码, TransformerEncoder ×L, mean pool ─ (B, d_model)
src2 (B, x_dim) ─────────────────────────────────────────────────────┐
                                                                    ▼
                     concat ─ (B, 2·d_model + x_dim) ─ Linear → 1024 → 512 → 128 → 6
                     6 头 → tanh/加权组合 → Q (B, 1)
```

- 两条路径用两个独立的 `TransformerEncoder`，层数、`d_model`、`nhead`、`dim_feedforward` 相同。
- 位置编码 `PositionalEncoding(d_model, max_len=35)` 只加在历史路径；`pe` 是 `register_buffer`，不是 `nn.Parameter`。
- 兼容版的编码器不使用 `batch_first`，而是在送进编码器前从 `(B, S, d_model)` 转成 `(S, B, d_model)`，出来后转回。

### 2.4 state_dict 为什么能直接加载

兼容模型刻意保持与训练端相同的模块树：`state_embed` / `action_embed` / `state_encoder.layers.N...` / `action_encoder...` / `pos_encoder.pe` / `linear1..4` 等键名一致，参数形状由同一套结构决定。因此 `model.load_state_dict(torch.load(ckpt, map_location='cpu'))` 可以严格加载，不需要 `strict=False` 或键名映射。

### 2.5 FP16 与 CPU 推理限制

LibTorch 1.5.0 的 CPU 后端没有 FP16 卷积/矩阵乘内核，直接在 CPU 上跑 half 张量会报类似：

```
RuntimeError: "unfolded2d_copy" not implemented for 'Half'
```

所以 FP16 只用来**减小模型文件体积**（约减半，便于上传 Botzone User Storage），部署时 C++ 端在 `torch::jit::load` 之后检测到 half 参数就整体 `module.to(torch::kFloat32)`，再用 FP32 输入推理（见 [cpp.md](cpp.md) 的 `ModelInference::loadModel`）。加载转换后运行时的内存与 FP32 模型基本相同，收益主要在磁盘与传输体积。

## 三、导出环境

### 3.1 版本要求

| 组件 | 版本 | 说明 |
|------|------|------|
| Python | 3.8.x | 本仓库验证使用 3.8.10 |
| PyTorch | 1.5.0+cpu | 本仓库验证使用 1.5.0+cpu；必须是**导出时**使用的版本 |
| numpy | 任意可用版本 | `verify_fp16.py` 用来求平均误差 |

下文把该解释器记作 `<BRIDGE_PYTHON>`。`export_douenmh.py` 本身不读取任何环境变量；`BRIDGE_PYTHON` 是调用约定，由 `export_douenmh.bat` 解析（也方便文档里统一占位）。

检查解释器是否符合要求：

```bash
<BRIDGE_PYTHON> -c "import sys, torch; print(sys.version); print(torch.__version__)"
# 期望看到 3.8.x 与 1.5.0+cpu
```

### 3.2 为什么不能直接用训练环境导出

- 训练环境是 PyTorch 2.x，导出的 TorchScript 不保证能被 LibTorch 1.5.0 加载；
- 训练端模型代码用了 1.5.0 不存在的 `nn.Mish` 与 `batch_first=True`，在 1.5.0 里无法实例化；
- 目标产物必须只包含 1.5.0 认识的算子，用 1.5.0 导出是最直接的保证。

本机 PATH 上的 `python`（例如 Python 3.12 + PyTorch 2.11）可以运行脚本的 `--help`（参数解析不依赖 1.5 专属逻辑），但**不要用它执行真正的导出**；这样得到的 `.pt` 不在 C++ 端的兼容范围内。导出与验证都应在 `<BRIDGE_PYTHON>` 下执行。

### 3.3 `export_douenmh.bat` 的解释器探测顺序

批量脚本按以下顺序找解释器，找不到会明确报错而不是静默失败：

1. 环境变量 `BRIDGE_PYTHON` 指向的 `python.exe`（存在才采用）；
2. 环境变量 `BRIDGE_PYTHON_DIR` 下的 `python38_full\python.exe`；
3. 仓库根目录上一级的 `environments\bridge_python\python38_full\python.exe`；
4. 都没有则打印错误并退出，提示设置 `BRIDGE_PYTHON` 或 `BRIDGE_PYTHON_DIR`。

找到解释器后还会执行一次 `import torch; print(torch.__version__)`：导入失败直接报错；版本不是 `1.5.*` 时打印警告（说明导出的模型可能无法被 LibTorch 1.5.0 加载）。

## 四、架构自动检测

### 4.1 出牌模型的检测规则

`detect_play_architecture(state_dict, nhead=8)` 从 checkpoint 的 state_dict 推断 V1 出牌模型参数：

| 参数 | 来源 | 说明 |
|------|------|------|
| `d_model` | `state_embed.weight.shape[0]` | `Conv1d(120, d_model, 1)` 的 out_channels |
| `num_encoder_layers` | 数 `state_encoder.layers.{i}.self_attn.in_proj_weight` 的连续层数 | 两个编码器层数相同，按 state_encoder 计数 |
| `dim_feedforward` | `state_encoder.layers.0.linear1.weight.shape[0]` | FFN 第一层输出维度 |
| `x_dim` | `linear1.weight.shape[1] - 2 * d_model` | MLP 头输入 = state + history + x，反推出 52 或 56 |
| `nhead` | **无法从权重推断** | 参见 4.2 |

`x_dim` 的 52 / 56 由训练端的 C5 特征开关决定：`DOUENMH_ENABLE_C5` 默认开启时 X 为 56 维，关闭时为 52 维（见 [architecture.md](architecture.md) 的 C1–C5 表）。检测直接读 MLP 输入宽度，不依赖环境变量。

### 4.2 为什么 `nhead` 必须显式给

`nn.MultiheadAttention` 的权重形状只与 `d_model` 有关：`in_proj_weight` 是 `(3·d_model, d_model)`，与头数无关。因此：

- `nhead` 猜错时 `load_state_dict` **不会报错**，但前向时把通道切成不同数量的头，输出数值会变；
- 导出脚本默认 `--nhead 8`，与训练端 `python/douzero/dmc/arguments.py` 的默认值一致；训练时若改过头数，导出必须传相同的 `--nhead`。

### 4.3 play 模型 V1 / V2 的判定

权重键名可以直接区分两种结构：

| 结构 | 特征键 | 编码器 |
|------|--------|--------|
| V1 双塔 | `state_embed.weight`、`action_embed.weight` | `state_encoder` + `action_encoder` |
| V2 Cross-Attention | `state_proj.weight`、`action_proj.weight`、`history_proj.weight`、`cross_attn.*` | `state_encoder` + `history_encoder` + 2 层 `action_encoder` |

训练端 `python/infer.py` 就是按 `cross_attn.` 或 `state_proj.` 前缀是否存在来判断 V2。**当前导出脚本只实现 V1**：`detect_play_architecture` 第一步读 `state_embed.weight`，对 V2 checkpoint 会抛 `KeyError: 'state_embed.weight'`；`compat_models_douenmh.py` 里也没有 V2 的兼容实现。C++ 端虽然保留了 V2 的拆分输入接口 `predictPlayCrossAttention`（`state_z=(1,5,120)`、`action_z=(B,1,120)`、`history_z=(1,35,120)`、`x=(B,52/56)`，输出 `(B,1)`），但 `main.cpp` 走的是 V1 `predictPlay`。也就是说，线上链路目前是 V1。

### 4.4 叫分模型没有检测

`GeneralModelBid` 结构写死（通道 2→4→8→16、FC 243→256→256→128→3、`x_dim=3`），导出时直接 `GeneralModelBid()` 再加载权重，不做架构推断。

## 五、导出流程与 CLI

### 5.1 前置条件

训练端每个位置单独保存一个纯 state_dict：

```
python/douzero_checkpoints/<XPID>/
├── first_<frame>.ckpt
├── second_<frame>.ckpt
├── third_<frame>.ckpt
├── landlord_<frame>.ckpt
├── landlord_up_<frame>.ckpt
└── landlord_down_<frame>.ckpt
```

文件名格式为 `{position}_{frame}.ckpt`，`frame` 是训练步数（纯整数）。训练端同时还会写一个合并的 `model.tar`（含优化器状态等），**那不是导出脚本要读的文件**；导出只认上面的逐位置 `.ckpt`。权重生成与放置见 [models/README.md](../models/README.md)。

### 5.2 命令示例

在仓库根目录执行 FP32 导出：

```bash
<BRIDGE_PYTHON> tools/export/export_douenmh.py \
    --checkpoint-dir python/douzero_checkpoints/DouEnMH_v1.2 \
    --output-dir tools/export/export_model \
    --suffix 620000000
```

省略 `--suffix` 时自动选 checkpoint 目录里最大的 frame：

```bash
<BRIDGE_PYTHON> tools/export/export_douenmh.py
```

只导出其中两个位置，并跳过导出后自检：

```bash
<BRIDGE_PYTHON> tools/export/export_douenmh.py --positions first landlord --skip-verify
```

FP32 + FP16 一起导出：

```bash
<BRIDGE_PYTHON> tools/export/export_douenmh.py --suffix 620000000 --fp16
```

### 5.3 `export_douenmh.py` 参数表

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint-dir` | 脚本目录的 `../../python/douzero_checkpoints/DouEnMH_v1.2` | checkpoint 目录 |
| `--output-dir` | 脚本目录的 `export_model` | 输出根目录；实际写到 `<output-dir>/<suffix>/` |
| `--suffix` | 自动探测 | checkpoint frame 编号；缺省时取 `*_*.ckpt` 中最大的整数 frame |
| `--positions` | 全部 6 个 | 要导出的位置，取值 `first second third landlord landlord_up landlord_down`，可写多个 |
| `--skip-verify` | 关 | 跳过导出后的原模型/traced 对比与 FP16 跨精度对比 |
| `--fp16` | 关 | 在 FP32 之外额外导出 FP16 |
| `--fp16-only` | 关 | 只导出 FP16 |
| `--nhead` | `8` | 注意力头数；权重里没有这个信息，必须与训练配置一致 |

### 5.4 自动选择 suffix 的规则

`_find_latest_suffix` 扫描 checkpoint 目录下所有 `.ckpt`，把文件名按最后一个 `_` 拆开，取能转成整数的最大 frame。注意：

- 它不检查 position 名，只看后缀；
- 如果 6 个位置的文件 frame 不一致，会选全局最大值，缺这个 frame 的位置会被跳过；
- 一个位置都找不到时打印 `ERROR` 并以非 0 退出。

### 5.5 `export_douenmh.bat` 包装参数

Windows 下可用批处理包装，默认路径相对脚本自身，不依赖盘符：

```bat
tools\export\export_douenmh.bat 620000000 --fp16 --copy
```

| 位置/参数 | 说明 |
|-----------|------|
| 第 1 个非选项参数 | checkpoint frame 编号（可选） |
| `--fp16` | 传给 Python 脚本，额外导出 FP16 |
| `--copy` | 导出后把 6 个 `{position}_jit.pt` 复制到 `cpp/build/data/models/` |
| `--checkpoint <dir>` | 覆盖 checkpoint 目录 |
| `--output <dir>` | 覆盖输出根目录 |

默认值：checkpoint 目录为脚本目录的 `../../python/douzero_checkpoints/DouEnMH_v1.2`，输出目录为脚本目录的 `export_model`。`--copy` 只复制 FP32 的 `_jit.pt`（C++ 默认加载的就是它）。

### 5.6 单个模型的导出步骤

FP32（`torch.jit.trace`）：

1. 按 5.3 的规则确定架构参数；
2. `GeneralModelBid()` 或 `GeneralModelTransformer(**arch)` 实例化并 `eval()`；
3. `torch.load(ckpt, map_location='cpu')` 后 `load_state_dict`；
4. 构造 batch=1 的 dummy 输入（bid `(1,2,120)/(1,3)`、play `(1,41,120)/(1,x_dim)`）；
5. `torch.jit.trace(model, (z, x))`，保存为 `{position}_jit.pt`。

FP16（`torch.jit.script`）：

1. 同样的实例化与加载，然后 `model.half()`（参数和浮点 buffer，包括 `pos_encoder.pe`，一起转 half）；
2. `torch.jit.script(model)`（**不执行**前向，因此不会触发 CPU 不支持 half 卷积的问题）；
3. **先保存** `{position}_jit_fp16.pt`，再做跨精度对比。对比时会把模型 `.float()`，而 `.float()` 是就地转换；如果先验证后保存，保存下来的"FP16"文件实际是 FP32；
4. 跨精度对比读取磁盘上的 FP32 与 FP16 两份文件，各自加载后统一用 FP32 输入推理，误差阈值 `1e-3`。

导出阶段的自检结果（`[OK]` / `[MISMATCH]` / `[LARGE]`）只打印、不改变退出码；要作为验收门槛请用 `verify_fp16.py`（见第八节）。

## 六、6 个模型的输入签名与输出

| 位置 | 模型 | 输入 | 输出 |
|------|------|------|------|
| `first` / `second` / `third` | `GeneralModelBid`（ResNet + SE + MLP） | `z=(B,2,120)`，`x=(B,3)` | 3 个头 → Q `(B,1)` |
| `landlord` / `landlord_up` / `landlord_down` | `GeneralModelTransformer`（V1 双塔） | `src1=(B,41,120)`，`src2=(B,52或56)` | 6 个头 → Q `(B,1)` |

### 6.1 叫分模型

- `z` 两行：row 0 是候选叫分动作值广播到 120 维；row 1 是自己手牌的 8×15 编码（120 维）。
- `x` 三维：`[bid_info[0], bid_info[1], bid_info[2]]`，原始叫分值。
- 3 个头拆成 `win_rate` / `win` / `lose`，`Q = w̃·win + (1−w̃)·lose`，其中 `w̃=(tanh(win_rate)+1)/2`；导出的模型已经把这一步烘焙进 `forward`，直接返回 `(B,1)`。
- C++ 端合法叫分动作最多 4 个（不叫、1、2、3），`predictBid` 的 chunk 上限是 16，实际不会触发多 chunk。

### 6.2 出牌模型 V1

`src1` 的 41 行：

| 行 | 内容 |
|----|------|
| 0 | 当前候选动作的 8×15 编码（120 维） |
| 1 | 自己手牌 |
| 2 | 对手手牌并集 |
| 3 | 自己累计已出 |
| 4 | 上手累计已出 |
| 5 | 下手累计已出 |
| 6–40 | 最近 35 手出牌历史：最近的在前，pass 为全零，不足 35 手用全 `-1` 哨兵补齐 |

`src2` 的 52 / 56 维：

```
bid_info(3) + num_cards_left(3) + spring(1) + bomb_num(15)
+ hand_stats(7) + oracle_features(5) + opponent_features(4) + ctp_features(14)
+ c5_features(4)   # DOUENMH_ENABLE_C5 开启时才有，X 维 56；关闭时 X 维 52
```

各段特征的含义与计算见 [python.md](python.md) 与 [architecture.md](architecture.md)；导出侧只关心总维度，由 `linear1.weight.shape[1] - 2·d_model` 自动得出。

模型内部把行 0–5 送 state 编码器、行 6–40 送历史编码器；6 个头拆成 `LWR / LPW / LPL / LOR_a / LOR_p / LOR_d`，只有前三头参与 Q：

```
Q = w̃ · LPW + (1 − w̃) · LPL
w̃ = (tanh(LWR) + 1) / 2
```

`LOR_a / LOR_p / LOR_d` 是神谕辅助头，只用于训练损失。导出的 TorchScript 直接返回 `(B,1)` 的 Q 向量，C++ 端在候选动作维上做 `argmax`，chunk 上限 50。

### 6.3 出牌模型 V2（当前导出链路不支持）

训练端 `GeneralModelTransformerV2` 的输入是拆分后的四元组：

| 张量 | 形状 | 含义 |
|------|------|------|
| `state_z` | `(1, 5, 120)` | 5 行静态状态，不随候选动作变化 |
| `action_z` | `(N, 1, 120)` | N 个候选动作 |
| `history_z` | `(1, 35, 120)` | 35 手历史 |
| `x` | `(N, 52或56)` | 每个候选动作的辅助特征 |

训练端 `forward` 在推理模式下返回 dict，其中 `values` 为 `(N,1)`；若要导出成 TorchScript，需要一个只返回该张量的简化 `forward`。当前 `compat_models_douenmh.py` 与 `export_douenmh.py` 都只实现 V1（见 4.3），给它们 V2 checkpoint 会在检测阶段失败；C++ 端对应的 `predictPlayCrossAttention` 接口存在但未启用。

## 七、输出文件与目录布局

### 7.1 命名

```
tools/export/export_model/
└── <suffix>/
    ├── first_jit.pt            # 叫分模型 FP32
    ├── second_jit.pt
    ├── third_jit.pt
    ├── landlord_jit.pt         # 出牌模型 FP32
    ├── landlord_up_jit.pt
    ├── landlord_down_jit.pt
    └── （--fp16 时额外）
        first_jit_fp16.pt  …  landlord_down_jit_fp16.pt
```

- 文件名固定为 `{position}_jit.pt` / `{position}_jit_fp16.pt`，**不含 step 后缀**；step 只体现在父目录名上。
- 输出根目录由 `--output-dir` 决定，脚本会在其下创建 `<suffix>/`。
- 大小随架构增长：`d_model=384`、6 层时，单个出牌模型约 78 MiB、单个叫分模型不到 1 MiB；仓库的 `.gitignore` 忽略 `tools/export/export_model/` 与 `*.pt`，权重不入库（原因见 [models/README.md](../models/README.md)）。

### 7.2 复制到 C++ 端

`cpp/src/main.cpp` 从**工作目录**下的 `data/models` 读取模型，本地构建产物的工作目录是 `cpp/build/`，所以放这里：

```bash
cd tools/export
cp export_model/620000000/*_jit.pt ../../cpp/build/data/models/
```

或直接用批处理包装的 `--copy`。C++ 端按 `{dir}/{name}_jit.pt` 拼路径；`modelPath(..., fp16=true)` 才会读 `_jit_fp16.pt`，而 `main.cpp` 目前按默认 FP32 加载。Botzone 提交时把 6 个 `_jit.pt` 上传到 User Storage 的 `data/models/` 即可。

## 八、FP16 导出与 `verify_fp16.py`

### 8.1 为什么 FP16 用 `script` 而不是 `trace`

`torch.jit.trace` 会真实执行一遍前向。在 PyTorch 1.5.0 CPU 上执行 half 卷积会直接报 `"unfolded2d_copy" not implemented for 'Half'`，所以 FP16 改用 `torch.jit.script`：它只编译代码、不执行算子，可以在 CPU 上生成 half 模型。代价是必须保证模型代码可脚本化——这也是兼容版 `forward` 只保留推理分支的原因之一。

### 8.2 导出与验证命令

```bash
# 导出 FP32 + FP16
<BRIDGE_PYTHON> tools/export/export_douenmh.py --suffix 620000000 --fp16

# 对比 FP32 与 FP16 的推理输出
<BRIDGE_PYTHON> tools/export/verify_fp16.py \
    --model-dir tools/export/export_model \
    --suffix 620000000

# --suffix 省略时，自动选 export_model/ 下编号最大的数字子目录
<BRIDGE_PYTHON> tools/export/verify_fp16.py --model-dir tools/export/export_model
```

`verify_fp16.py` 的默认 `--model-dir` 是脚本目录的 `export_model`。

### 8.3 对比方法与指标

1. 在 `<model-dir>/<suffix>/` 下找 `{position}_jit.pt` 与 `{position}_jit_fp16.pt`；缺任一份记 `SKIP`，加载失败记 `ERROR`。
2. FP32 模型直接加载；FP16 模型加载后 `.float()` 转成 FP32 权重（与 C++ 端 `module.to(kFloat32)` 的部署路径一致）。两者都用 **FP32 输入**推理，因此这里衡量的是 FP16 权重舍入经 FP32 推理后的偏差，不是 half 算子本身的误差。
3. 叫分模型在 batch `{1,2,4,8,16,32}` 上、出牌模型在 batch `{1,4,16,64,128}` 上用同一组随机输入各跑一遍。
4. 指标与阈值：

| 指标 | 阈值 | 含义 |
|------|------|------|
| `max_abs_error` | `< 1e-3` | 每个 batch 上逐元素最大绝对误差，再对所有 batch 取平均 |
| `argmax_agreement` | `>= 99%` | 输出 `argmax` 一致率 |

判定条件是两者同时满足才 `PASS`。注意导出模型的输出是 `(B,1)`，最后一维长度为 1，因此 `argmax` 恒为 0、一致率恒为 100%；实际起门槛作用的是 `max_abs_error`。脚本在 `FAIL` 时只打印警告，**不会以非 0 退出**；把它接进 CI 时需要自己检查输出或加退出码判断。

FP16 模型的输入维度由 `_get_x_dim_from_model` 探测：先用 `(1,56)` 试跑 FP32 模型，失败则按 52 维处理。

### 8.4 导出阶段的内置自检

不跑 `verify_fp16.py` 时，导出脚本自身也会对比：

| 阶段 | 比较对象 | batch | 阈值 | 标签 |
|------|----------|-------|------|------|
| FP32 导出后 | 原模型 vs traced | bid `{1,3,10}`、play `{1,3,10,50}` | `1e-5` | `[OK]` / `[MISMATCH]` |
| FP16 导出后 | 磁盘 FP32 traced vs 磁盘 FP16（转 FP32） | 同上 | `1e-3` | `[OK]` / `[LARGE]` |

若输出目录里没有对应的 FP32 文件（例如 `--fp16-only` 且此前没导出过 FP32），FP16 对比会打印 `skip cross-precision verify: FP32 model not found`。

## 九、常见错误与排查

### 9.1 动态 shape、trace 与脚本化

- `torch.jit.trace` 只记录一次前向实际走过的算子；受 Python 控制流影响的分支会被"冻结"。本项目两个模型的 `forward` 没有数据相关的 `if`，卷积/Transformer 也都与 batch 维无关，所以同一个 traced 模型可以用不同 batch（导出脚本已用 batch 1/3/10/50 交叉验证）。
- 历史路径的位置编码里有 `self.pe[:, :x.size(1)]`，trace 时会打印
  `TracerWarning: Converting a tensor to a Python index ...`。这是预期警告：历史长度固定为 35，traced 图按 35 固化；如果以后改历史长度，必须重新导出。
- 模型的 41 行布局、120 维编码长度是硬约束；trace 不会帮你检查这些，形状不对会在运行时报通道/尺寸错误。

### 9.2 `batch_first` 相关

PyTorch 1.5.0 的 `TransformerEncoderLayer` 不接受 `batch_first=True`，必须用 `(S, B, E)`。兼容版在编码器前后手动 transpose（见 2.3）。如果自己改兼容层时漏掉 transpose，典型报错是 `Expecting tensor in shape (seq, batch, feature)` 或 attention 维度不匹配。

### 9.3 FP16 相关

| 现象 | 原因 / 处理 |
|------|-------------|
| `"unfolded2d_copy" not implemented for 'Half'`（导出时） | 对 half 模型用了 `trace`；改用 `script`（脚本已如此） |
| `"unfolded2d_copy" not implemented for 'Half'`（C++ 推理时） | 加载后没有把模型转 FP32；`ModelInference::loadModel` 会检测 half 参数并 `to(torch::kFloat32)` |
| `_jit_fp16.pt` 大小与 FP32 几乎一样 | 之前版本的导出顺序是先验证后保存，而验证里的 `.float()` 是就地转换，导致保存成 FP32；当前脚本先保存再验证，且验证时重新加载一份文件 |
| `skip cross-precision verify: FP32 model not found` | 只导出了 FP16（`--fp16-only`）；跨精度对比需要同目录的 `_jit.pt` |
| `[LARGE]` / `FAIL` | FP16 权重舍入使 Q 值偏差超过 `1e-3`；检查该位置是否对精度特别敏感，或改用 FP32 |

### 9.4 checkpoint 步数或文件名不一致

- `_find_latest_suffix` 取全局最大 frame。若 6 个位置的文件 frame 不同，缺失的位置会打印 `[SKIP] not found`（FP16 阶段则静默跳过）；一个都没找到时脚本报 `ERROR: 未在 ... 中找到任何 .ckpt 文件` 并退出。
- 训练端还可能写一个合并 checkpoint `model.tar`；它不是逐位置 state_dict，`load_state_dict` 会报 `Missing key(s)` / `Unexpected key(s)`。导出只读 `{position}_{frame}.ckpt`。
- 手工指定 `--suffix` 要与实际文件名完全一致（纯数字 frame，不带前导补零差异）。

### 9.5 state_dict 不匹配

| 报错 | 常见原因 |
|------|----------|
| `KeyError: 'state_embed.weight'` | 给 V1 检测函数喂了 V2 checkpoint（V2 用 `state_proj`）；当前导出脚本不支持 V2 |
| `Missing key(s) in state_dict` / `Unexpected key(s)` | checkpoint 不是该位置的逐位置权重（拿错文件、拿了 `model.tar`），或 V1/V2 结构拿反 |
| `size mismatch for ...` | checkpoint 与实例化架构不符：d_model / 层数 / x_dim 不一致；检查是否拿错训练配置的权重 |

### 9.6 `nhead` 静默错误

`nhead` 错时 `load_state_dict` 不报错，但多头注意力的通道切分不同，Q 值会变。导出前确认训练时的 `--nhead`（默认 8），必要时用 `--nhead` 覆盖。若 `d_model` 不能被 `nhead` 整除，实例化阶段会直接报错。

### 9.7 X 维度 52 / 56

`x_dim` 由 MLP 输入宽度反推，不依赖环境变量；但训练时如果 `DOUENMH_ENABLE_C5` 的开关与部署端特征编码不一致，模型本身能加载、输出却对应错误的特征布局。导出前确认训练配置；C++ 端 `predictPlay` 会按输入数据自动推断 52 或 56，不需要额外参数。

### 9.8 LibTorch 加载失败

| 现象 | 检查项 |
|------|--------|
| `torch::jit::load` 抛异常、提示无法识别的格式/算子 | 模型是否由 PyTorch 1.5.0 导出；用 2.x 导出的 `.pt` 不在兼容范围内 |
| 找不到 DLL | 把 LibTorch 的 `torch.dll`、`torch_cpu.dll`、`c10.dll` 等放到 exe 同目录或在 `PATH` 里 |
| 模型加载成功但 Q 值全一样 | 权重没加载对（检查文件名/位置），或 `x_dim` 与导出时不一致 |

## 十、相关文件

| 文件 | 作用 |
|------|------|
| [tools/export/compat_models_douenmh.py](../tools/export/compat_models_douenmh.py) | PyTorch 1.5 兼容的 V1 出牌模型与叫分模型 |
| [tools/export/export_douenmh.py](../tools/export/export_douenmh.py) | 6 个模型的 TorchScript 导出（FP32 trace / FP16 script） |
| [tools/export/export_douenmh.bat](../tools/export/export_douenmh.bat) | Windows 包装：解释器探测、默认路径、`--copy` |
| [tools/export/verify_fp16.py](../tools/export/verify_fp16.py) | FP16 vs FP32 精度对比 |
| [models/README.md](../models/README.md) | 权重放置与导出后复制方式 |
| [cpp.md](cpp.md) | C++ 端模型加载、FP16→FP32、chunk 推理 |
| [architecture.md](architecture.md) | 训练端、导出、推理端的整体衔接 |
