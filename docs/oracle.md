# 神谕（Oracle）特征与 DP 引擎

本文说明 DouEnMH 的神谕特征体系与 `dp_unified` 动态规划引擎：5 维特征的定义与归一化、组合生成、DP 状态表示与优化项、缓存生命周期、C++ 与 Python 两端的实现对照，以及实测复杂度与边界回退。

神谕回答一个很具体的问题：**这手牌最少要几手才能出完**，以及"有控牌权"和"没有控牌权"两种极端下这个步数差多少。它不依赖对手手牌（推理可用），训练时又能用完全信息算出更准的版本作辅助监督。相关文档：[architecture.md](architecture.md)、[encoding.md](encoding.md)（X 向量布局）、[python.md](python.md)（训练流程）、[cpp.md](cpp.md)（推理管线）。

---

## 一、神谕解决什么问题

斗地主的核心规划量是"清空手牌的步数"：同样的 17 张牌，出成一手顺子还是拆成五手单张，胜率完全不同。Transformer 从 8×15 卡牌矩阵里**可以**自己学出这种结构感，但需要大量样本；神谕把"最少步数"用一次 DP 直接算出来，作为显式的规划信号塞给网络。

五个量分别刻画不同侧面：

| 量 | 直觉 | 为什么需要 |
|----|------|-----------|
| $N_{\text{active}}$ | 始终有控牌权时最少几手出完 | 手牌的"进攻速度"上界 |
| $N_{\text{passive}}$ | 始终没有控牌权时最少几手出完 | 手牌的"跟牌/被动"代价 |
| $\Delta N = N_{\text{passive}}-N_{\text{active}}$ | 控牌敏感度 | 越大越依赖拿到控牌权，越小越能"躺着跟" |
| $N_{\text{preserve}}$ | 带拆弹惩罚的最少步数 | 拆炸弹要付代价，衡量"保留炸弹"的价值 |
| $\text{self\_adv}$ | 自己步数 − 对手估计步数 | 双方速度差，领先还是落后 |

**训练端两种用法**：

1. **X 特征（C2，5 维）**：逐动作计算"打出该动作后剩余手牌"的神谕值，与动作相关；只用己方手牌，因此 DMC 自对弈与推理端的输入分布一致。
2. **辅助监督标签（LOR 头，3 维）**：$N_{\text{active}}$、对手感知 $N_{\text{passive}}$、$\Delta N$ 的归一化值，训练时可用完全信息（能看见对手手牌）把 $N_{\text{passive}}$ 算得更准，作为额外回归目标。LOR 头只参与损失，不进入 Q 值（见 [architecture.md](architecture.md) 第二节）。

**推理端**：C++ 对每个合法动作独立计算 5 维特征，拼进 X 向量偏移 29 的位置。两端都必须逐位复现同一套定义，否则训练-推理特征错位。

---

## 二、5 维特征的定义与公式

### 2.1 记号

| 符号 | 含义 |
|------|------|
| $H$ | 己方手牌，表示为 15 维等级计数向量（等级 0–14：3,4,…,K,A,2,小王,大王） |
| $H_r$ | 手牌中等级 $r$ 的张数 |
| $\mathcal{L}(H)$ | 从 $H$ 出发的所有合法出牌组合（不含 pass） |
| $c$ | 一个组合；$c_r$ 为它使用等级 $r$ 的张数 |
| $\text{type}(c)$, $\text{main}(c)$ | 组合的牌型与主等级（连牌取链条最高等级） |
| $\alpha$ | 拆弹惩罚系数，代码中硬编码为 $1.0$ |

### 2.2 $N_{\text{active}}$（最少主动回合数）

始终握有控牌权时，清空手牌所需的最少出牌步数：

$$
N_{\text{active}}(H)=
\begin{cases}
0, & H=\varnothing,\\[4pt]
\displaystyle\min_{c\in\mathcal{L}(H)}\left(1+N_{\text{active}}(H-c)\right), & \text{否则}.
\end{cases}
$$

pass 不会减少手牌、也不改变控牌权归属，因此不进 $\mathcal{L}(H)$。值域 $[0,20]$（最多 20 张牌），归一化后 $[0,\,20/30)\approx[0,0.67)$。

### 2.3 $N_{\text{preserve}}$（炸弹保持度）

把"拆掉一个四张等级"计为额外一步的代价：

$$
N_{\text{preserve}}(H)=
\begin{cases}
0, & H=\varnothing,\\[4pt]
\displaystyle\min_{c\in\mathcal{L}(H)}\left(1+\pi(H,c)+N_{\text{preserve}}(H-c)\right), & \text{否则},
\end{cases}
$$

$$
\pi(H,c)=\alpha\cdot\bigl|\{\,r \mid H_r=4,\ 0<c_r<4\,\}\bigr|,\qquad \alpha=1.0 .
$$

即：组合 $c$ 使用了某个本来满 4 张的等级、但只用了 1–3 张 → 该等级"被拆"，加一步。若 $c$ 把 4 张全用掉（打炸弹），不算拆。惩罚按**等级个数**计，不按张数计；代码用 15 位 `bomb_mask`（bit $r$ 置位当 $0<c_r<4$）配合 ctz 遍历实现。

### 2.4 $N_{\text{passive}}$（最少被动回合数）

设计文档里给过"对手手牌完全可见时的递推"，但**代码里没有实现该递推**，实际只有两种启发式，用途不同：

**（a）启发式版——X 特征恒用此版**（`dp_passive_steps_heuristic`）

按牌型统计"能领出的不同主等级个数"，取各牌型的最大值，再减去炸弹/火箭条数，下限 1：

$$
N_{\text{passive}}^{\text{heur}}=\max\!\Bigl(1,\ \max_{t}\bigl|\{\text{main}(c)\mid c\in\mathcal{L}(H),\ \text{type}(c)=t,\ t\notin\{\text{BOMB},\text{KING\_BOMB}\}\}\bigr|-\#\text{bombs}\Bigr).
$$

- 排除炸弹与火箭后，按 `type` 分桶，统计每类中不同的 `main_level` 个数（C++ 用 `std::set<Level> levels_by_type[18]`，Python 用 `defaultdict(set)`）。
- `#bombs` = 该手牌生成的炸弹/火箭组合条数（每个四条等级 1 条、火箭 1 条；17 张手牌下最多 4 条）。
- 含义：对手若总能用同型更大的牌压你，你需要按"每种牌型最多能领出几次"里最差的那个来估计回合数。
- 该启发式的入参 `combos` 已是当前手牌的组合集，`hand` 参数未被使用。

**（b）对手感知版——仅训练标签用**（`dp_passive_opponent_aware`）

用对手真实手牌（完美信息）估计"对手总能压住你"时的步数：

$$
N_{\text{passive}}^{\text{opp}}=\max\!\Bigl(1,\ \sum_{t\in\mathcal{T}}\min\bigl(s_t,\,o_t\bigr)-\#\text{bombs}_{\text{self}}\Bigr),\qquad
\mathcal{T}=\{\text{SINGLE},\text{PAIR},\text{TRIPLE},\text{SERIAL\_SINGLE},\text{SERIAL\_PAIR},\text{SERIAL\_TRIPLE}\},
$$

其中 $s_t,o_t$ 分别是己方/对手手牌中牌型 $t$ 的"可领出次数"（`_count_unique_levels_by_type`）：单张/对子/三条统计 ≥1/≥2/≥3 张的等级数，炸弹统计 $H_r=4$ 的等级数，火箭看双王是否齐，顺子/连对/飞机统计滑窗合法起点数（最小长度分别为 5/3/2）。对地主视角，$o_t$ 来自两位农民手牌的**并集**（`all_handcards['landlord_up'] + all_handcards['landlord_down']`）；农民视角则用地主整手牌。

> 标签用 (b)、特征用 (a) 是刻意的：特征必须只靠不完美信息，标签可以用完美信息提高精度。

### 2.5 $\Delta N$（控牌敏感度）

$$
\Delta N=\max\bigl(0,\ N_{\text{passive}}-N_{\text{active}}\bigr).
$$

X 特征里用的是启发式对：$\max(0,\,N_{\text{passive}}^{\text{heur}}-N_{\text{active}})$；LOR 标签里用对手感知对：$\max(0,\,N_{\text{passive}}^{\text{opp}}-N_{\text{active}})$。值域 $[0,20]$。

### 2.6 $\text{self\_adv}$（优势距离）

$$
\text{self\_adv}=N_{\text{active}}(H_{\text{rem}})-\widehat{N}_{\text{opp}},\qquad
\widehat{N}_{\text{opp}}=\frac{c_{\text{opp}}}{2.8},
$$

其中 $H_{\text{rem}}$ 是"打出该动作后"的剩余手牌，$c_{\text{opp}}$ 是作为基准的对手剩余张数，$2.8$ 是每回合平均出牌张数（`AVG_CARDS_PER_STEP = 2.8`，C++/Python 同值）。

**对手基准选择规则**（两端一致）：

| 己方角色 | 基准 |
|----------|------|
| 地主 | 两位农民剩余张数的**较小值** $\min(c_{\text{up}},c_{\text{down}})$（只需跑赢最快清完的那个农民） |
| 农民 | 地主剩余张数 $c_{\text{landlord}}$ |

该基准只取决于局面、不随候选动作变化；代码在逐动作循环外算一次（C++ `main.cpp` 的 `oppRemaining`，Python `env.py` 的 `opponent_for_adv`）。$c_{\text{opp}}=0$ 时 $\widehat{N}_{\text{opp}}=0$。值域约 $[-20/2.8,\,20]$，注释标注 $[-20,20]$，归一化后 $(-1,1)$。

### 2.7 特征值域汇总

| 特征 | 原始值域（代码注释/上限） | 归一化 | 归一化值域 |
|------|---------------------------|--------|-----------|
| `n_active` | $[0,20]$ | $x/(x+10)$ | $[0,0.67)$ |
| `n_passive` | $[0,40]$ | $x/(x+15)$ | $[0,0.73)$ |
| `delta_n` | $[0,20]$ | $x/(x+8)$ | $[0,0.71)$ |
| `n_preserve` | $[0,\sim25]$ | $x/(x+10)$ | $[0,\sim0.71)$ |
| `self_adv` | 约 $[-20,20]$ | $\tanh(x/5)$ | $(-1,1)$ |

---

## 三、归一化

饱和归一化统一为 $f(x)=x/(x+K)$，只对原始值取非负后使用：

```cpp
// C++: oracle.h  normalize_oracle()
n_active   = a / (a + 10.0f);
n_passive  = p / (p + 15.0f);
delta_n    = d / (d + 8.0f);
n_preserve = pr/ (pr + 10.0f);
self_adv   = std::tanh(sa / 5.0f);
```

```python
# Python: oracle.py  saturate(x, k) = max(0.0, x) / (max(0.0, x) + k)
K_N_ACTIVE, K_N_PASSIVE, K_DELTA_N, K_N_PRESERVE, K_SELF_ADV = 10.0, 15.0, 8.0, 10.0, 5.0
```

| 特征 | 常数 | C++ | Python | 位置 |
|------|------|-----|--------|------|
| `n_active` | $K=10$ | 字面量 `10.0f` | `K_N_ACTIVE = 10.0` | `oracle.h` / `oracle.py` |
| `n_passive` | $K=15$ | `15.0f` | `K_N_PASSIVE = 15.0` | 同上 |
| `delta_n` | $K=8$ | `8.0f` | `K_DELTA_N = 8.0` | 同上 |
| `n_preserve` | $K=10$ | `10.0f` | `K_N_PRESERVE = 10.0` | 同上 |
| `self_adv` | $K=5$ | `5.0f` | `K_SELF_ADV = 5.0` | 同上 |

LOR 标签复用同样的 $K$：`saturate(n_active, 10)`、`saturate(n_passive, 15)`、`saturate(delta_n, 8)`（`compute_oracle_labels_from_raw`），因此标签与 X 特征落在同一尺度。之所以用饱和函数而不是线性缩放：$x/(x+K)$ 在 $x$ 小时接近线性、在大 $x$ 时压住尾部，手牌从 20 张降到 0 张的过程中不会出现量纲突变。

---

## 四、组合生成 `generate_all_combos`

### 4.1 覆盖的牌型

C++ 端在计数向量空间直接生成 **14 种标准牌型**（`OracleComboType` 枚举，0–15，其中 `PASS=0`、`INVALID=15` 不参与生成）：

| id | 枚举 | 中文 | 生成条件（C++） |
|---:|------|------|----------------|
| 1 | `SINGLE` | 单张 | 每个 $H_r\ge1$ |
| 2 | `PAIR` | 对子 | 每个 $H_r\ge2$，$r\le$ `MAX_REGULAR_LEVEL=12`（2，不含王） |
| 3 | `TRIPLE` | 三条 | 每个 $H_r\ge3$，$r<$ 小王 |
| 4 | `BOMB` | 炸弹 | $H_r=4$ 的普通等级 |
| 5 | `KING_BOMB` | 火箭 | 双王齐 |
| 6 | `TRIPLE_1` | 三带一 | 三条 + 任意不同等级 1 张（可用王） |
| 7 | `TRIPLE_2` | 三带二 | 三条 + 任意不同等级对子 |
| 8 | `SERIAL_SINGLE` | 顺子 | 连续段长 ≥5，每级 ≥1，上限 A（等级 11） |
| 9 | `SERIAL_PAIR` | 连对 | 连续段长 ≥3，每级 ≥2 |
| 10 | `SERIAL_TRIPLE` | 飞机（不带翼） | 连续段长 ≥2，每级 ≥3 |
| 11 | `SERIAL_3_1` | 飞机带单 | 飞机 + 与链条等量的单张（不同等级） |
| 12 | `SERIAL_3_2` | 飞机带对 | 飞机 + 与链条等量的对子（不同等级） |
| 13 | `QUAD_2_SINGLE` | 四带两单 | 四条 + 两个不同等级各 1 张 |
| 14 | `QUAD_2_PAIR` | 四带两对 | 四条 + 两个不同等级各 2 张 |

**不含航天飞机**（`SSHUTTLE` 系列）：C++ 神谕 DP 的搜索空间限定为标准 14 种牌型，`move_generator.h` 的航天飞机类型不进入神谕组合集。连续牌型上限由 `MAX_STRAIGHT_LEVEL=11`（A）控制，2 与王不参与链条。

实现分解为三个 helper：`add_simple_combos`（单/对/三/炸/火箭）、`add_serial_combos`（顺子/连对/飞机，用 `find_chain_segments` 找连续段）、`add_kicker_combos`（三带、四带、飞机带翼，用 `enumerate_kicker_combos` 做栈式组合枚举）。排序在 DP 内部完成（按 `size` 降序），生成阶段不排序。

### 4.2 为什么用 `CardCount` 计数向量

```cpp
using CardCount = std::array<short, NUM_LEVELS>;   // 15 维，每维 0–4
```

- **状态即哈希**：每个等级 3 bit（$0\ldots4$ 共 5 种取值），15 个等级恰好拼成 45 位完美哈希 `Σ (c_r & 7) << 3r`，无碰撞、无排序、无 `Counter`。
- **减法 O(1)**：`sub[i] = hand[i] - combo.counts[i]`，就是 15 次整数减法；换成"牌列表"则要排序/查找。
- **位掩码预过滤**：由计数向量直接得到 15 位 `bitmask`（该组合涉及哪些等级），与手牌掩码做一次与运算即可淘汰大量不可用组合。
- **天然去重**：花色不影响步数，按等级计数把同等级不同花色的等价组合合并，DP 状态空间只由等级构成决定。

### 4.3 与 Python `MovesGener` 的关系

训练端不自己写组合生成，而是调用 `douzero.env.move_generator.MovesGener`（按"牌"而非"等级"生成，`gen_type_1`…`gen_type_14`），再在 `_combo_to_metadata` 里转成 `(size, count_vec, bitmask, bomb_mask)`。两侧关系：

| 方面 | Python（`MovesGener` + `_combo_to_metadata`） | C++（`generate_all_combos`） |
|------|-----------------------------------------------|------------------------------|
| 生成单位 | 具体牌（含花色），同一等级多重集可能有多个花色变体 | 等级计数向量，天然唯一 |
| 转换 | 逐张牌累加等级计数 → 计数向量 | 直接构造计数向量 |
| 重复项 | 花色变体产生重复计数向量，DP 内被 memo 合并（只多花遍历时间） | 无重复 |
| DP 输入 | 同一套 `(count_vec, bitmask, bomb_mask)` 元数据 | 同一套，`ComboMeta` 字段对齐 |
| 带牌规则 | 四带两单 / 飞机带单的"带牌"可以是同等级两张（即拆一对当两张单） | `enumerate_kicker_combos` 要求带牌等级**互不相同** |

上表最后一行是两端组合集合的唯一实质差异（实测，同一手牌按等级多重集比对）：

| 手牌 | C++ 组合数 | Python 唯一计数向量数 | 差集 |
|------|-----------:|---------------------:|------|
| 17 张、无炸弹 | 61 | 61 | 0 |
| 20 张、1 炸弹 | 147 | 150 | 3（均为四带两单的"带一对"） |
| 17 张、2 炸弹 | 187 | 189 | 2（四带两单/飞机带单各 1） |
| 17 张、3 三条 | 213 | 215 | 2（飞机带单的"带一对"） |

即 C++ 的搜索空间是 Python 的子集，搜出的 $N_{\text{active}}$ 可能略大（少一种拆法）。实测这些手牌上未改变最终最优值，但误差方向是单向的（C++ 只能偏大）。

> `oracle.cpp` 头部注释写"组合总数上界约 1000–3000"，与实测不符：17–20 张典型手牌的组合数在 **几十到约 250** 量级（见上表；炸弹/三条越多越大）。Python 端 `_MAX_COMBOS = 25000` 的缓冲区上限是保守预留。

---

## 五、DP 核心 `dp_unified`

### 5.1 入口与签名

```cpp
// C++ (oracle.h): 一次 DP 得到两个值，按 bomb_penalty 取用
short dp_min_steps(const CardCount& hand,
                   const std::vector<OracleCombo>& combos,
                   float bomb_penalty = 0.0f);   // 0 → N_active，1 → N_preserve
std::vector<OracleCombo> generate_all_combos(const CardCount& hand);
void  dp_clear_cache();                          // 每局开始清空 memo
short dp_passive_steps_heuristic(const CardCount&, const std::vector<OracleCombo>&);
OracleRaw compute_oracle_raw(const CardCount& hand_counts);
```

```c
/* C 扩展 (oracle_core.c)，供 Python ctypes 调用 */
int dp_unified_c(const int *hand_vec, const int *combo_counts, const int *combo_masks,
                 const int *combo_sizes, const uint16_t *combo_bomb_masks,
                 int n_combos, const int *first_ci_for_rem, float *result /*[2]*/);
void dp_clear_cache_c(void);
int  dp_cache_size_c(void);
```

```python
# Python 调度层 (oracle.py)
def dp_unified(hand_vec, combo_data, first_ci_for_rem):  # 先查 Python 全局缓存，再走 C 扩展/纯 Python
def compute_oracle_batch(hand_cards, legal_actions, hand_opp=None, opponent_remaining=0):
def normalize_oracle_features(oracle_raw):               # → (5,)
def compute_oracle_labels_from_raw(oracle_raw):          # → (3,)
```

### 5.2 状态表示与 key

- 状态 = 剩余手牌的 15 维计数向量（`CardCount` / `int[15]`）。
- `key = Σ_r (c_r & 7) << 3r`，45 位有效信息；空槽用 `key == 0` 作哨兵（空手牌本身返回 $(0,0)$，不入 memo 也正确）。
- `hmask = Σ_r 1<<r`（`c_r > 0`），与组合 bitmask 配合做 O(1) 淘汰。
- 哈希表槽位：`key_to_slot(k)` 用 SplitMix64 终混器（乘 `0xbf58476d1ce4e5b9`、`0x94d049bb133111eb`，两次异或移位）把 45 位键摊到槽位，再取低 `MEMO_BITS` 位。

```text
dp(hand) -> (best_a, best_p):
    key, hmask = hash_and_mask(hand)          # 一次遍历 15 个等级
    if memo 命中: return memo[key]
    if remaining == 0: return (0, 0)
    lb = ceil(remaining / max_combo_size)
    best_a, best_p = init_a, init_p           # 根节点用贪心上界，子节点用 +inf
    for ci in [first_ci_for_rem[remaining], n_combos):
        if lb >= best_a and lb >= best_p: break          # 下界剪枝
        if (combo.bitmask & hmask) != combo.bitmask: continue
        if combo.counts > hand: continue
        if combo.size == remaining:                       # 单 combo 快速路径
            best_a = min(best_a, 1); best_p = min(best_p, 1 + penalty(hand, combo))
            continue
        sub = hand - combo.counts
        sub_a, sub_p = dp(sub)                            # 传 +inf，不传父上界
        best_a = min(best_a, 1 + sub_a)
        best_p = min(best_p, 1 + sub_p + penalty(hand, combo))
    memo[key] = (best_a, best_p)
    return (best_a, best_p)
```

### 5.3 一次递归双输出

$N_{\text{active}}$ 与 $N_{\text{preserve}}$ 共享同一棵递归树：每个子状态同时返回 `(n_active, n_preserve)`，两条值只在"是否累加拆弹惩罚"上分叉。相比跑两遍 DP，组合匹配、位掩码、memo 命中的开销都只付一次；拆弹惩罚只在当前步累加（`bomb_penalty=0` 时 $N_{\text{active}}$，$=1$ 时 $N_{\text{preserve}}$）。

### 5.4 贪心初始上界

`greedy_bound` 每次贪心选"当前可用组合中牌数最大"的一个，直到出完，得到一个**完整可行解**的步数（同时算出带惩罚版本）。这个值是**上界**：任意完整解的步数都不会比 DP 最优更小，所以拿它作为根节点的初始 `best` 是安全的，从第一次递归起就有紧界可剪枝。

注意贪心只用于**根节点**：递归子调用一律传 `INF_SHORT = 32767`（C++）/ `FLOAT_INF = 1e30`（C）作为初始上界。

### 5.5 下界剪枝

每步最多消耗 `max_combo_size` 张牌（组合按 size 降序排，`metas[0].size` 即最大组合牌数），因此

$$
\text{lb}=\left\lceil \frac{\text{remaining}}{max\_combo\_size} \right\rceil
$$

是剩余步数的下界；当 `lb >= best_a && lb >= best_p` 时，之后任何组合都不可能把任一目标改小，直接 `break`（组合已按 size 降序，后面的组合只会更小、lb 不会变大）。当贪心上界恰等于下界时，根节点可能一步递归都不做就返回（实测见 5.11 的 trips 手牌）。

### 5.6 memo 哈希表

| 项 | C++ (`oracle.cpp`) | C 扩展 (`oracle_core.c`) | Python 纯回退 (`_dp_unified_py`) |
|----|--------------------|--------------------------|----------------------------------|
| `MEMO_BITS` | 20 | 20 | 无（`dict`） |
| 槽位数 | $2^{20}=1{,}048{,}576$ | 同左 | — |
| 单条大小 | `MemoEntry` 16 B（`uint64 key` + 2×`short`，对齐到 16） | `MemoSlot` 16 B（`uint64` + 2×`float`） | — |
| 内存 | 16 MiB | 16 MiB（`calloc`，首次调用分配） | 动态 |
| 冲突 | 线性探测，`MAX_PROBES = 64` | 同左 | 无 |
| 空槽哨兵 | `key == 0` | `key == 0` | — |
| 表满行为 | 探测 64 槽仍无空位 → **丢弃该条**（只影响性能，不影响正确性） | 同左 | 不适用 |
| 清空 | `std::fill` 全表 | `memset` 全表 | 清 `_dp_global_cache` |

1,048,576 槽 × 16 B = 16 MiB，在 Botzone 256 MB 限制下可接受；实测典型 17 张手牌首帧写入 36,862 条（负载 ~3.5%），20 张含炸弹手牌 138,228 条（~13%），探测溢出风险低。

### 5.7 `first_ci_for_rem` 跳表

`first_ci_for_rem[r]`（长度 21，$r=0..20$）表示"第一个 `size <= r` 的组合下标"。组合按 size 降序排列后，比剩余牌数还大的组合全部集中在前缀，用一次表查就能跳过它们。

构建方式与 Python 端 `_precompute_combo_data` 一致：$r$ **从 20 递减到 0**，`ci` 单调递增。

```cpp
size_t ci = 0;
for (short r = 20; r >= 0; r--) {
    while (ci < metas.size() && metas[ci].size > r) ci++;
    first_ci_for_rem[r] = (short)ci;
}
```

DP 内取 `start_ci = first_ci_for_rem[remaining]` 后从该下标开始遍历，即保证 `metas[ci].size <= remaining`。方向不能反：组合按 size 降序，若 $r$ 从 0 开始升序推进，$r=0$ 那一轮就会把 `ci` 推到 `n_combos`，之后所有表项都等于 `n_combos`，combo 循环将一次都不执行（历史版本曾如此，见 7.2① 的修复记录）。

### 5.8 单 combo 快速路径

若某个组合恰好用光剩余手牌（`combo.size == remaining`），该子问题无需递归：$N_{\text{active}} = 1$，$N_{\text{preserve}} = 1+\pi(H,c)$。这既省一次递归，也避免为全清子状态建 memo 条目。

### 5.9 组合按 size 降序

组合元数据先按 `size` 降序排序再进 DP：大牌型优先遍历，能让 `best` 更快接近最优、增强下界剪枝；同时让 `first_ci_for_rem` 的"前缀都是 oversized"性质成立。

### 5.10 不传播父节点上界

递归调用**不**把父节点的 `best` 当子节点初始上界，一律传 `+inf`。原因：子问题的真实最优可能 ≥ 父节点当前 best，若把父 best 传下去，子节点会在 `lb >= best` 时提前 `break`，返回一个非最优的"上界值"，父节点的 min 就会算错。根节点用贪心上界是安全的——贪心构造的是一个**完整可行解**，不是局部的猜测。

### 5.11 优化项清单

| # | 优化 | C++ | C | Python 回退 |
|--:|------|-----|---|-------------|
| 1 | 15 维计数向量、45 位完美哈希 | ✓ | ✓ | `tuple` 作 `dict` 键（无位哈希） |
| 2 | 位掩码预过滤 `(cmask & hmask) == cmask` | ✓ | ✓ | ✓ |
| 3 | 组合按 size 降序 | ✓ | 由 Python 预排序后传入 | ✓ |
| 4 | 下界剪枝 | ✓ | ✓ | ✓（贪心界缺省为 ∞，见下） |
| 5 | 单次递归双输出 | ✓ | ✓ | ✓ |
| 6 | 全局哈希表 memo | ✓ | ✓ | `dict` + 合并进 `_dp_global_cache` |
| 7 | 贪心初始上界 | ✓ | ✓ | **无**（初值 `inf`，更慢但结果同为最优） |
| 8 | `bomb_mask` + ctz 遍历拆弹等级 | ✓ | ✓ | `_CTZ_TABLE` 查表 |
| 9 | `first_ci_for_rem[21]`（$r$ 降序构建） | ✓ | ✓ | ✓ |
| 10 | 合并 hash + hmask 单次遍历 | ✓ | ✓ | 分两步 |
| 11 | 单 combo 快速路径 | ✓ | ✓ | ✓ |

实测（同一手牌 17 张、61 组合）：纯 Python 回退 742.6 ms、36,862 个 memo 状态；C 扩展 10–20 ms（耗时对比见第八节）。纯 Python 慢的主因是没有贪心上界 + 字典状态开销。

---

## 六、缓存生命周期

### 6.1 两层缓存（Python 训练端）

| 层 | 内容 | 清理时机 | 作用 |
|----|------|----------|------|
| `_dp_global_cache` | `hand_vec → (n_active, n_preserve)` 顶层结果 | `Env.reset()` → `oracle._dp_global_cache_clear()` | 同局内重复查询（pass 后再次轮到自己）O(1) 命中，免跨 FFI |
| C `g_memo` | DP 递归内部所有子状态 | 同上，调用 `dp_clear_cache_c()` | 之后各帧手牌是首帧手牌的子集，子状态大概率已算过 |
| `_gen_moves_cache` | `sorted(hand) → combos` | 同上 | 组合生成结果复用 |

清理入口 `_dp_global_cache_clear()` 一并清空三层。由于同一局内手牌只减不增，首帧 DP 已把大量子手牌算过，之后各帧的首次查询大量命中 C memo。

### 6.2 预热策略（两端一致）

```text
预热: compute_oracle_raw(完整手牌)        # 填满 C memo（根 + 全部子状态）
逐动作: for action in legal_actions:
            hand_rem = hand − action        # 计数相减（多重集减法）
            raw = compute_oracle_raw(hand_rem)   # 子状态 memo 命中
```

- **训练端**：`env.py` 的 `_get_obs_resnet` → `oracle.compute_oracle_batch()`，先 `compute_oracle_raw(hand_cards)` 预热，再对每个候选动作算 `hand − action` 的神谕值，返回 `(B,5)` 特征与 `(B,3)` 标签。
- **推理端**：`main.cpp` 先 `compute_oracle_raw(handCounts)`（结果丢弃、只为填 memo），再对每个合法动作做计数相减后逐动作查询。DP 系数是共享的，所以逐动作开销只剩"重新生成该子手牌的组合集 + memo 查根"。
- 计数相减必须是**多重集减法**：手里 3 张 5、出一对 5 后应剩 1 张，不能按集合成员删掉全部同等级牌（训练端注释里记录过这个历史 bug）。

### 6.3 每局清理（C++）

`main.cpp` 用 `g_firstPlayOfEpisode` 标志：一局第一次出牌前调用 `dp_clear_cache()`（`std::fill` 16 MiB，实测 0.43 ms），之后整局复用。KEEP_RUNNING 模式下进程常驻、GameState 跨回合保留，只有 flag 为真时才清。

---

## 七、C++ 与 Python（C 扩展）对照

### 7.1 相同项

| 检查项 | C++ (`oracle.cpp`) | Python 侧 | 结论 |
|--------|--------------------|-----------|------|
| DP 目标 | `N_active` / `N_preserve` 双输出 | C 扩展 `dp_unified_c` 双输出 | 一致 |
| 拆弹惩罚 | $\alpha=1.0$，按等级计数 | `BOMB_PENALTY 1.0f` | 一致 |
| 递推形式 | $1+\text{sub}$，惩罚在当步累加 | 同 | 一致 |
| 位掩码/下界剪枝/快速路径 | 11 项优化 | C 扩展 12 项（第 12 项是 MemoSlot 16 B 紧凑打包） | 一致 |
| 贪心上界 | `greedy_bound` 构造完整解 | 同（`greedy_bound`） | 一致 |
| memo 规模 | `MEMO_BITS=20`，1,048,576 槽，16 MiB | 同 | 一致 |
| 探测上限 | `MAX_PROBES=64` | 同 | 一致 |
| 槽位哈希 | SplitMix64 终混器 | 同 | 一致 |
| 清缓存 | `dp_clear_cache()` 每局 | `dp_clear_cache_c()` 每局 | 一致 |
| 被动启发式 | `max(1, max_chain − bomb_count)` | 同 | 一致 |
| self_adv 基准 | 地主 min(up,down)，农民地主 | 同 | 一致 |
| 归一化 | 5 个 $K$ 与 tanh | 同 | 一致 |

### 7.2 已知差异与历史问题

**① `first_ci_for_rem` 构建方向（历史 bug，已修复）**

历史版本的 `oracle.cpp` 把该表的循环写成 $r$ **升序**（0→20）推进 `ci`；组合本身是 size 降序，于是 $r=0$ 那一轮 `ci` 已到 `metas.size()`，21 项全部等于组合数。`dp_unified_internal` 取 `start_ci = first_ci_for_rem[remaining]` 后 combo 循环一次也不执行，函数直接返回初始值，也就是**贪心上界**。

- 修复前用插桩副本实测：`dp_min_steps` 的递归调用次数为 0（只在根节点算一次贪心）；MSVC /O2 下独立插桩同样得到 `dp_calls=1`（见 [cpp.md](cpp.md) §4.1）。
- 修复：`dp_unified_impl` 的循环改为 `for (short r = 20; r >= 0; r--)`，与 Python `_precompute_combo_data` 保持一致；修复后 `first_ci_for_rem[r]` 恰为"第一个 `size <= r` 的下标"，DP 正常递归。
- 修复后 C++ 的 `N_active`/`N_preserve` 与 Python/C 扩展逐值一致（下表右两列相同）。

对照实测（同一手牌，修复前 vs 修复后 vs Python/C 扩展）：

| 手牌 | 组合数 | 修复前（=贪心上界） | 修复后（=真最优） | Python/C 扩展 |
|------|-------:|--------------------:|------------------:|--------------:|
| 17 张、无炸弹 | 61 | $N_a=5,\ N_p=5$ | $N_a=4,\ N_p=4$ | $N_a=4,\ N_p=4$ |
| 20 张、1 炸弹 | 147 | $N_a=5,\ N_p=6$ | $N_a=4,\ N_p=4$ | $N_a=4,\ N_p=4$ |
| 17 张、2 炸弹 | 187 | $N_a=3,\ N_p=5$ | $N_a=2,\ N_p=3$ | $N_a=2,\ N_p=3$ |
| 17 张、3 三条 | 213 | $N_a=6,\ N_p=6$ | $N_a=2,\ N_p=2$ | $N_a=2,\ N_p=2$ |

修复前的偏差方向是单向的：贪心是完整可行解，所以 $N_a$、$N_p$ 只会偏大，`delta_n = max(0, N_p−N_a)` 随之偏小；`n_preserve` 也偏大。也就是说，修复前训练-推理特征在第 2/3/4 维上存在系统性错位，修复后两端同表同值。

另一组与 [cpp.md](cpp.md) §4.1 同源的手牌（同一插桩副本、MSVC /O2；"修复前"指历史版本）：

| 手牌 | 修复前（贪心） | 修复后（真最优） | Python DP |
|------|----------------|------------------|-----------|
| 17 张（4 个炸弹 + 单张，Botzone ID 0–16） | `dp_calls=1`，$N_a=2,\ N_p=6$ | `dp_calls=24815`，$N_a=2,\ N_p=5$ | $N_a=2,\ N_p=5$ |
| 20 张（上述 + 底牌 ID 48,49,50） | $N_a=3,\ N_p=7$ | $N_a=3,\ N_p=4$ | $N_a=3,\ N_p=4$ |
| 小牌局（三张 3 + 单张 ×2） | `dp_calls=1`，$N_a=2,\ N_p=2$ | 同左 | $N_a=2,\ N_p=2$ |

这组对照说明修复只影响"拆牌/炸弹保持"类手牌：小牌局两种实现都直接到达贪心最优（差异为 0），而含 4 个炸弹时 $N_{\text{preserve}}$ 从 6 降到 5，20 张时从 7 降到 4。

**② 带牌同等级组合缺失**

四带两单（type 13）与飞机带单（type 11）的带牌，`MovesGener` 允许"同等级两张"（把一对拆成两张单牌用），C++ `enumerate_kicker_combos` 强制带牌等级两两不同。C++ 组合集是 Python 的子集（4.3 节实测），只会让 $N_{\text{active}}$ 偏大或相等。

**③ 返回类型与数值精度**

C++ 用 `short`/`float`，C 扩展用 `float`；DP 值都是 ≤ 20 的整数，双精度/单精度均无舍入差异。Python 纯回退用 `float` + `dict`。

**④ memo 检查顺序**

C 扩展先查 memo 再判空手牌；C++ 先判空手牌再查 memo。空手牌在两侧都返回 `(0,0)`，顺序不影响结果。

**⑤ 注释与旧文档的过期数值（不影响运行）**

| 位置 | 过期写法 | 实际 |
|------|----------|------|
| `oracle_core.c` `key_to_slot` 注释 | "low 17 bits" | `MEMO_BITS=20`，1,048,576 槽 |
| `oracle.cpp` 注释 | "MAX_STRAIGHT_LEVEL 已在 oracle.h 中定义" | 定义在 `card_defs.h` |
| 旧审查文档 | "MEMO_BITS 18/19 与 Python 20 不一致" | 两端现在都是 20 |
| 旧审查文档 | "C++ self_adv 用两农民剩余之和" | 现在两端都是地主取 min |
| 旧设计文档 | 声称实现了"对手手牌可见的完整被动 DP" | 代码只有两种启发式（2.4 节） |

---

## 八、复杂度与耗时

**状态数**：DP 的状态空间是"从整手牌出发、按合法组合逐步相减"能到达的计数向量集合，受手牌结构（顺子/连对/炸弹）强烈影响。实测 memo 条目数：

| 手牌 | 唯一组合数（C++ / Python） | memo 状态数 |
|------|--------------------------:|------------:|
| 17 张、无炸弹 | 61 / 61 | 36,862 |
| 20 张、1 炸弹 | 147 / 150 | 138,228 |
| 17 张、2 炸弹 | 187 / 189 | 12,704 |
| 17 张、3 三条 | 213 / 215 | 1（下界剪枝在根节点直接结束） |

**复杂度量级**：每个状态遍历组合（受 `first_ci_for_rem` 与位掩码剪枝），最坏 $O(S\cdot C)$，其中 $S$ 为可达状态数、$C$ 为组合数；$S$ 的上界是指数级，但实际手牌下降到 $10^4$–$10^5$。实测 17 张 / 36,862 状态 / 61 组合 ≈ $2.2\times10^6$ 次组合检查。

**耗时**（单线程本地实测，C++ 行为 g++ -O2 构建；括号内为条件）：

| 路径 | 17 张 | 20 张 | 说明 |
|------|------:|------:|------|
| C 扩展首帧冷启动 | 10–20 ms | 96.7 ms | 实测，含 DP 递归与 memo 填充；波动来自 CPU 状态（另测 19 张手牌 47–62 ms） |
| C++ 修复后首帧 | 11.8–13.3 ms | 95.0–95.4 ms | g++ -O2 实测（本表手牌）；另 MSVC /O2 不同手牌的实测为 1.3 ms（17 张）/ 6.6 ms（20 张），见 [cpp.md](cpp.md) §4.1；同一实现下耗时由状态数决定（双炸 17 张 8.6 ms、三条 17 张 0.02 ms） |
| C++ 修复前版本 | ~0.01 ms | ~0.01 ms | 实测，仅贪心界（7.2①），保留作对照，不代表 DP 成本 |
| Python 纯回退 | 742.6 ms | —（估算 ~2–3 s） | 实测 17 张；20 张按状态数比例 138,228/36,862 ≈ 3.7× 估算 |
| 组合生成 | 1–5 μs | 3–10 μs | 实测 |
| 被动步数启发式 | 2–9 μs | 同量级 | 实测 |
| memo 清空 | 0.43 ms | 同 | 实测，16 MiB `fill` |
| memo 热态根查询 | 1–12 μs | 同 | 实测：修复后 DP 结果已入 memo，重复查询只做一次哈希与探测 |

绝对耗时对手牌结构极敏感：修复后的同一实现从 0.02 ms（17 张三条手牌，根节点即被下界剪枝）到 95 ms（20 张炸弹手牌）都出现过，状态数相差 4 个数量级；MSVC /O2 下不同手牌的实测（1.3 ms / 6.6 ms）见 [cpp.md](cpp.md) §4.1，量级差异来自手牌结构而非实现差异。

整机层面（本地 runner，含模型加载/切换）：修复前后首个出牌决策的中位耗时在噪声内（地主 20 张首手 1.437 s vs 1.463 s，农民 17 张 0.888 s vs 0.950 s；之后 4–5 张手牌决策 0.029–0.035 s），说明该修复只改变 C2 数值、不改变推理路径与预算（见 [cpp.md](cpp.md) §4.1）。

**逐帧摊销**（Python 训练端，预热后）：根候选手牌的重复查询命中 `_dp_global_cache` 约 1–12 μs；逐动作子手牌查询命中 C memo 平均 2 μs、最大 5 μs（实测 40 个动作）。也就是说，一帧的 DP 成本基本等于首帧一次冷启动，之后每个动作只付组合生成 + 查表。

**内存**：

| 项 | 大小 | 依据 |
|----|------|------|
| DP memo | 16 MiB | $2^{20}\times16$ B（C++ 静态 vector / C `calloc`） |
| Python ctypes 缓冲区 | $\approx$ 1.75 MiB | `_MAX_COMBOS=25000` ×（15×4 B + 4 + 4 + 2）|
| Python 纯回退 memo | 动态，一般 ≪ 16 MiB | `dict` 条目数同状态数 |

---

## 九、边界与回退

| 情形 | 行为 |
|------|------|
| 空手牌 | `compute_oracle_raw` 返回全零（`total == 0` 提前返回）；DP 基例 `(0,0)` |
| 无可用组合（`combos.empty()`） | `dp_unified_impl` 返回 `(0,0)`，不进入递归 |
| memo 表满 | 线性探测 64 槽后丢弃该条：结果仍正确（该状态下次重算），只损失缓存命中 |
| C 扩展 DLL 缺失 / 加载失败 / 平台不支持 | `_load_oracle_c()` 静默失败，`dp_unified` 回退到 `_dp_unified_py`（慢约 40–70×，结果同为最优） |
| Oracle 不可用时的调试开关 | 环境变量 `ORACLE_PLACEHOLDER=1` 时 `compute_oracle_raw` 返回固定占位值（常数为 5/7/2/6），跳过全部 DP |
| 组合数超过 ctypes 缓冲区 | `_MAX_COMBOS = 25000`，超出抛 `RuntimeError`（实际手牌远达不到） |
| 训练时拿不到对手手牌（`hand_opp is None`） | LOR 标签退化为启发式 $N_{\text{passive}}$（与特征同值） |
| 基准对手剩余为 0 | `estimate_opponent_n_active` 直接返回 0，`self_adv = N_active` |
| 合法动作不是当前手牌子集（数据异常） | 训练端 `hand_rem.remove(c)` 抛 `ValueError` 时 `break`，退化为少扣一张，不中断训练 |
| 每局缓存残留 | 训练端 `Env.reset()`、推理端 `g_firstPlayOfEpisode` 都会清缓存；跨局不清会读到上一局同型手牌的旧值 |
| DP 超时/时间预算 | 代码没有超时中断或降级路径：DP 一旦开始就跑完整个状态空间，靠下界剪枝与 memo 控制耗时；实测修复后冷启动为 1–100 ms 量级（取决于手牌状态数），预热度后逐动作查询为微秒级，见第八节 |

---

## 十、相关文件

| 文件 | 内容 |
|------|------|
| `cpp/src/oracle.h` / `cpp/src/oracle.cpp` | C++ 实现：常量、`OracleCombo`、组合生成、`dp_unified`、启发式、归一化 |
| `python/douzero/env/oracle.py` | Python 调度层：缓存、C 扩展封装、纯 Python 回退、特征/标签接口 |
| `python/douzero/env/oracle_core.c` | C 扩展 DP 实现（ctypes 调用） |
| `python/douzero/env/env.py` | `_get_obs_resnet` 中调用 `compute_oracle_batch`，把 5 维特征放进 X |
| `cpp/src/main.cpp` | `decidePlayModel` 中预热 + 逐动作查询，拼进 X 偏移 29 |
| `docs/encoding.md` | X 向量完整布局与 C2 段位置 |
