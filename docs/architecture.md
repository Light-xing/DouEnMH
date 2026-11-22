# 架构总览

DouEnMH 是一个斗地主 AI：用 PyTorch 训练 6 个价值网络，导出为 TorchScript，再由 C++ + LibTorch 1.5.0
在 Botzone 平台上实时推理。本文件说明三部分（训练端、模型导出、推理端）如何衔接，各部分的细节见
[python.md](python.md)、[model-export.md](model-export.md)、[cpp.md](cpp.md)。

## 一、全链路

```
        训练（GPU 集群/单机多进程）                     部署（Botzone 或本地）
 ┌──────────────────────────────────┐          ┌──────────────────────────────────┐
 │ python/  DMC actor-learner       │          │ cpp/  bot.exe (LibTorch 1.5.0)   │
 │  ├─ 8×15 编码 + Z/X 特征          │          │  ├─ JSON 解析 / 状态重建          │
 │  ├─ 6 头 Transformer / ResNet     │          │  ├─ 合法动作生成                  │
 │  ├─ 神谕 DP（辅助监督 + 特征）     │          │  ├─ 同构特征编码                  │
 │  └─ 对手建模（辅助特征）           │          │  ├─ TorchScript 推理 → argmax     │
 └───────────────┬──────────────────┘          │  └─ 状态更新闭环                  │
                 │ .ckpt                        └───────────────▲──────────────────┘
                 ▼                                              │ .pt
 ┌──────────────────────────────────┐          ┌───────────────┴──────────────────┐
 │ python/douzero_checkpoints/       │          │ cpp/build/data/models/*.pt        │
 └───────────────┬──────────────────┘          └───────────────▲──────────────────┘
                 │                                              │
                 ▼                                              │
 ┌──────────────────────────────────┐                          │
 │ tools/export/  PyTorch 2.x→1.5.0 │──────────────────────────┘
 │  ├─ export_douenmh.py（6 个模型）  │
 │  └─ verify_fp16.py（精度对比）     │
 └──────────────────────────────────┘

 ┌─────────────────────────────────────────────────────────────────────────────┐
 │ tools/runner/  runner.py（本地对局，不连 Botzone）                            │
 │  内置裁判 judge_botzone_ver.py + 每席位子进程 bot.exe + random/greedy/pass    │
 └─────────────────────────────────────────────────────────────────────────────┘
```

关键约束：**训练端与 C++ 端必须逐位复现同一套特征**。因此编码、Z/X 布局、神谕 DP、对手建模在
两端各有一份实现，改动编码后按 [encoding.md](encoding.md) 第八节逐项核对，并用导出脚本的自检
（trace vs eager）与 `tools/runner/` 的实际对局验证。

## 二、6 个模型

| 位置 | 模型 | 结构 | 输入 | 输出 |
|------|------|------|------|------|
| `first` / `second` / `third` | 叫分 | Conv1d ResNet + SE 注意力 + MLP | `z=(B,2,120)`、`x=(B,3)` | 3 个叫分动作的 Q |
| `landlord` / `landlord_up` / `landlord_down` | 出牌 | Transformer Encoder（V1 双塔 / V2 Cross-Attention） | `z=(B,41,120)`、`x=(B,52/56)` | 6 个头 |

出牌模型的 6 个头与 Q 值：

```
LWR（预测胜率）、LPW（胜时 ADP）、LPL（负时 ADP）  →  Q = w̃·LPW + (1−w̃)·LPL，w̃=(tanh(LWR)+1)/2
LOR_a / LOR_p / LOR_d                            →  神谕辅助头，只参与训练损失，不参与 Q
```

## 三、"奇怪的增强"分别是什么

| 增强 | 维度 | 作用 | 文档 |
|------|------|------|------|
| 8×15 卡牌矩阵编码 | 120 | 把牌型（断点、单顺/连对/飞机起点）显式摊开，替代 DouZero 的 54 维 | [encoding.md](encoding.md) |
| 神谕 DP（oracle） | 5 | 用"最少几步出完"的完全信息 DP 给价值网络提供辅助监督与即时特征 | [oracle.md](oracle.md) |
| 对手建模 | 22 | 手牌强度 OHS、炸弹概率 OBP、牌型偏好 CTP、压制意愿 SWI、节奏 TP | [opponent-modeling.md](opponent-modeling.md) |
| 叫分课程 + ε 衰减 | — | 先教叫分（P0 强制低叫），再放开探索 | [python.md](python.md) |
| 神谕 λ 退火 | — | 训练前期强监督、后期衰减到 0.03，避免辅助头主导 | [python.md](python.md) |
| C5 开关 | 4 | `DOUENMH_ENABLE_C5=0` 可关掉 SWI/TP，X 从 56 降到 52 | [opponent-modeling.md](opponent-modeling.md) |

## 四、一次出牌决策的数据流

```
Botzone request JSON
  → GameState.rebuildFromFullHistory / applyIncrementalUpdate   （首回合 O(N)，之后 O(1)）
  → LegalMoves.generate()                  合法动作（含 PASS；压牌时只生成同型同长）
  → 逐动作 oracle：预热整手牌 DP → 对每个 action 计算 5 维原始值
  → 逐动作特征：8×15 编码动作 + Z 41 行（V1）+ X 52/56（含 oracle / 对手建模 / 手牌统计）
  → ModelInference.predictPlay（分 chunk，≤50）→ 6 头 → Q 值 argmax
  → 状态更新 6 步闭环（删手牌 / 追加历史 / 累计已出 / 炸弹数 / 春天 / lastValidCombo）
  → response JSON（牌列表，PASS 为空）
```

## 五、目录

| 路径 | 内容 |
|------|------|
| `python/` | 训练端：DouZero DMC 框架 + 8×15 环境 + 6 头模型 + 神谕/对手建模 + 训练/评测/推理脚本 |
| `cpp/` | 推理端：Botzone C++ 源码、构建脚本、单文件合成与提交打包 |
| `tools/export/` | checkpoint → TorchScript 导出、FP16 精度对比 |
| `tools/runner/` | 本地对局 runner 与内嵌裁判（不依赖外部 judge 仓库） |
| `docs/` | 本套文档 |
| `models/` | 权重放置说明（权重本身不入库） |
| `.archive/` | 旧设计文档与论文翻译，仅本地保留，不随仓库发布 |

## 六、术语

| 术语 | 含义 |
|------|------|
| DMC | DouZero 的分布式 actor-learner 训练框架 |
| V1 / V2 | 出牌模型两种结构：state+action 拼接双塔 / Cross-Attention 拆分编码；`--use_v2` 切换，线上 C++ 目前用 V1 |
| C1–C5 | X 向量的五段扩展特征：手牌统计、神谕、对手建模、牌型偏好、压制意愿+节奏 |
| ADP | 平均回报目标（本项目的训练目标函数） |
| KEEP_RUNNING | Botzone 长时运行协议：进程常驻、回合间挂起，省去重复加载模型 |
| 神谕 / oracle | 在完全信息下对"最少回合数"做的 DP 结果，训练时作标签、推理时作特征 |
