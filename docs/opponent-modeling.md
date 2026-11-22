# 对手建模（22 维特征）

本文说明 DouEnMH 的对手建模模块：未知牌池的构建、C3 对手信念（OHS/OBP）、C4 牌型偏好（CTP）、C5 行为风格（SWI/TP）的全部公式与常数，以及训练端增量更新与 C++ 端每步全量回放的等价性与性能差异。

这 22 维只使用不完美观测（己方手牌、三家已出牌、底牌、剩余张数、出牌历史），训练与推理都能算，因此能作为 DMC 的输入特征而不是只当训练标签。X 向量里 C3 占偏移 34、C4 占 38、C5 占 52，完整布局见 [encoding.md](encoding.md) 第四节；本文只给结论。

---

## 一、22 维总览

| 层 | 特征 | 缩写 | 维数 | 计算方式 | 依赖信息 |
|----|------|------|-----:|----------|----------|
| C3 | 对手手牌强度 | OHS | 2（up/down 各 1） | 未知牌池加权期望，O(15) | 牌池 + 对手剩余张数 |
| C3 | 对手炸弹概率 | OBP | 2（up/down 各 1） | 超几何概率查表，O(15) | 牌池 + 对手剩余张数 |
| C4 | 牌型偏好 | CTP | 14（7 类 × 2 对手） | 领出历史 EMA，γ=0.9 | 出牌历史 |
| C5 | 压制意愿 | SWI | 2（up/down 各 1） | p_can 加权 EMA | 出牌历史 + 牌池 + OBP |
| C5 | 节奏偏好 | TP | 2（up/down 各 1） | 领出张数 EMA，γ=0.9 | 出牌历史 |
| | **合计** | | **22** | | |

**输出顺序**（两端一致）：

```
C3:  [ohs_up, ohs_down, obp_up, obp_down]
C4:  [ctp_up[0..6], ctp_down[0..6]]
C5:  [swi_up, swi_down, tp_up, tp_down]
```

`up`/`down` 是**相对位置**：出牌顺序为 landlord → landlord_down → landlord_up → landlord。训练端由 `env.py` 的 `relative_order` 映射（如自己是 `landlord_down` 时 up=landlord、down=landlord_up）；C++ 端用 `upPos=(myPos+2)%3`、`downPos=(myPos+1)%3`，与 `opponent_modeling.h` 内部按 `landlordPosition` 换算座位号的方式一致。

---

## 二、未知牌池 `buildUnknownPool`

### 2.1 定义与约束

设完整牌库 $\mathcal{D}$（54 张 = 13 个普通等级各 4 张 + 大小王各 1 张），时刻 $t$ 的观测划分：

$$
\mathcal{D}=H_{\text{self}}\ \cup\ P_{\text{self}}\ \cup\ P_{\text{up}}\ \cup\ P_{\text{down}}\ \cup\ H_{\text{up}}\ \cup\ H_{\text{down}}\ \cup\ D_{\text{bottom}},
$$

未知牌池（对两位对手手牌之并的估计）：

$$
\mathcal{U}=\mathcal{D}\setminus\bigl(H_{\text{self}}\cup P_{\text{all}}\cup D_{\text{unplayed}}\bigr),\qquad
P_{\text{all}}=P_{\text{self}}\cup P_{\text{up}}\cup P_{\text{down}},
$$

$$
n_r=\#\{r\in\mathcal{U}\},\qquad N=|\mathcal{U}|=\sum_{r=0}^{14} n_r,\qquad N\le 54-17=37 .
$$

约束：$c_{\text{up}}+c_{\text{down}}=N$（两位对手当前手牌之和恰为未知牌池），$c_o\le 20$（地主最多 20 张）。

### 2.2 底牌的两条规则（最容易写错的地方）

三张底牌在规则上属于地主手牌，因此**不能无条件减掉**：

| 视角 | 是否减底牌 | 原因 |
|------|-----------|------|
| 地主 | **不减** | `hand_self` 已包含底牌，再减一次会让牌池少 1–3 张（且偏差随局面变化，不是固定偏移） |
| 农民 | 减，但**只减尚未被打出的底牌** | 底牌已被地主打出时，那张牌已在"所有已出牌"里扣过一次，再按完整 3 张扣会重复 |

两端实现方式不同但等价：

- **Python**：`_build_unknown_pool(..., is_landlord=False)` 时把 `three_landlord_cards` 整体计入扣减；而 `game.py` 在地主打出底牌后会从 `self.three_landlord_cards` 中移除该牌，所以送进来的列表天然只剩"未打出的底牌"。
- **C++**：`buildUnknownPool` 只对非地主减底牌，并逐张检查它是否出现在 `state.landlordPlayedCards` 中，已打出则跳过。

### 2.3 构建实现（C++ / Python 对照）

| 项 | C++ `buildUnknownPool` | Python `_build_unknown_pool` |
|----|------------------------|------------------------------|
| 签名 | `void buildUnknownPool(const GameState&, std::array<int,15>& poolVec, int& poolTotal)` | `_build_unknown_pool(hand_self, all_played_cards, three_landlord_cards, is_landlord=False)` |
| 初始牌库 | `poolVec[0..12]=4`、`[13]=[14]=1` | `np.array([4]*13 + [1,1])` |
| 减己方手牌 | `state.myHandCards` 逐张 `botzoneCardToRank` | `_cards_to_rank_counts(hand_self)` |
| 减已出牌 | 依次减 `landlordPlayedCards`、`landlordUpPlayedCards`、`landlordDownPlayedCards` | 遍历 `all_played_cards.values()` |
| 减底牌 | 非地主视角；跳过着地主已打出的牌 | `not is_landlord` 时整表计入 |
| 负值防御 | 逐等级 `if (poolVec[i] < 0) poolVec[i] = 0` | `np.clip(pool_vec, 0, None)` |
| 返回 | 计数向量 + `poolTotal` | `(pool_vec, pool_total)` |

### 2.4 重建频率与复用

- **每次观测重建一次**：牌池随出牌单调收缩，不能跨帧缓存。
- **C3 单次建池复用 4 个子计算**：`compute_opponent_features`（C++ `computeOpponentFeatures`）建池后同时算 `ohs_up/ohs_down/obp_up/obp_down`，避免 4 次重复减法；C++ 还通过可选出参 `outPoolVec/outPoolTotal` 把牌池交给 C5 复用。
- **C5 回放**不复用"当前池快照"，而是自行从出牌阶段开始时重建并逐手推进（见 5.4），因为历史动作的 $p_{\text{can}}$ 必须用当刻牌池。
- 牌池为空（`poolTotal == 0`）时 OHS、OBP 都返回 0。

---

## 三、C3 对手信念（4 维）

### 3.1 OHS（对手手牌强度）

**符号**：

| 符号 | 含义 |
|------|------|
| $N$ | 未知牌池总张数 |
| $n_r$ | 牌池中等级 $r$ 的张数 |
| $c_o$ | 对手 $o$ 的剩余张数 |
| $w(r)$ | 等级价值权重 `RANK_WEIGHTS[r]` |

对手持有等级 $r$ 的期望张数按超几何抽样为 $c_o\cdot n_r/N$，于是

$$
\text{OHS}_o=\sum_{r=0}^{14} w(r)\cdot \frac{c_o\,n_r}{N}
=\frac{c_o}{N}\sum_{r=0}^{14} w(r)\,n_r .
$$

代码先算 $\text{weightedSum}=\sum_r w(r)n_r$，再乘 $c_o/N$（`computeOHSFromPool` / `_compute_ohs_from_pool`）；$c_o\le0$ 或 $N\le0$ 时返回 0。

**权重表**（`RANK_WEIGHTS[15]`，两端逐元素相同）：

| 等级 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | J | Q | K | A | 2 | 小王 | 大王 |
|------|---|---|---|---|---|---|---|---|---|---|---|----|---|---|---|
| 索引 | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
| $w$ | 0 | 0 | 1 | 1 | 2 | 2 | 3 | 3 | 4 | 4 | 5 | 6 | 8 | 10 | 12 |

**归一化**：`OHS_MAX = 100.0`，特征取 $\min(\text{OHS}_o/100,\ 1)$。100 是安全上界——极端分布（未知牌池全是高牌、对手拿满 20 张）下原始值可能超过 100，用 `min` 截断到 1 而不是线性溢出。

复杂度 $O(15)$，实测 Python 侧整个 C3（建池 + 4 个 OHS/OBP）约 21 μs/次（本地 2000 次循环均值）；C++ 为纯数组累加，量级更小。

### 3.2 OBP（对手炸弹概率）

**普通炸弹**：某等级 $r$ 在牌池中恰有 $n_r=4$ 张时，对手恰好抓走这 4 张（即形成炸弹）的朴素超几何概率

$$
P(\text{bomb}_r)=\frac{\binom{N-4}{\,c_o-4\,}}{\binom{N}{c_o}}\quad(n_r=4,\ c_o\ge4),
$$

**火箭**：牌池中双王都在时，对手同时抓走两张的概率

$$
P(\text{rocket})=\frac{\binom{N-2}{\,c_o-2\,}}{\binom{N}{c_o}}\quad(c_o\ge2).
$$

两点合并不做联合精确计算，而是**按独立近似**相乘：

$$
\text{OBP}_o=1-\bigl(1-P(\text{rocket})\bigr)\prod_{r:\ n_r=4}\bigl(1-P(\text{bomb}_r)\bigr),
$$

其中乘积只遍历 $r\in[0,12]$（13 个普通等级）且只对牌池中仍有 4 张的等级计入（不足 4 张不可能成炸）。值域 $[0,1]$，不再归一化。

**预计算查表**（模块加载时一次性算好，推理纯查表）：

| 项 | C++ `HypergeomTables` | Python 模块级表 |
|----|----------------------|-----------------|
| 维度 | `[OM_MAX_POOL+1][OM_MAX_REM+1]` = `[38][21]` | `[_MAX_POOL+1][_MAX_REM+1]` = `[38][21]` |
| 参数范围 | `OM_MAX_POOL=37`（$54-17$）、`OM_MAX_REM=20` | `_MAX_POOL=37`、`_MAX_REM=20` |
| `p_bomb_single[N][c]` | `C(N-4,c-4)/C(N,c)`，$N\in[4,37]$、$c\in[4,\min(N,20)]$ | 相同 |
| `p_rocket[N][c]` | `C(N-2,c-2)/C(N,c)`，$N\in[2,37]$、$c\in[2,\min(N,20)]$ | 相同 |
| 存储 | `float` 二维数组，2 × 38 × 21 × 4 B ≈ **6.2 KiB** | 两层 Python list（对象开销远大于 6 KiB） |

**边界**：$c_o\le3$ 或 $N<c_o$ 直接返回 0（炸弹至少要 4 张）；$N>37$ 或 $c_o>20$ 时查表项按 0 处理（实际局面不会出现）。

### 3.3 C3 两端对照

| 项 | C++ | Python |
|----|-----|--------|
| 入口 | `computeOpponentFeatures(state, upPos, downPos, outPoolVec, outPoolTotal)` | `compute_opponent_features(hand_self, cards_remaining_up, cards_remaining_down, all_played_cards, three_landlord_cards, is_landlord)` |
| OHS 归一化 | `std::min(ohs / OHS_MAX, 1.0f)` | `min(ohs_up / OHS_MAX, 1.0)` |
| 建池次数 | 1 次 | 1 次 |
| 输出顺序 | `{ohs_up, ohs_down, obp_up, obp_down}` | `[ohs_up, ohs_down, obp_up, obp_down]` |
| 辅助接口 | — | `compute_hand_strength(cards)`（绝对强度 $\sum w n$，不除以牌池、不截断）、`compute_ohs(...)`、`compute_obp(...)` 单值封装 |

---

## 四、C4 牌型偏好 CTP（14 维）

### 4.1 七类牌型

| 类别 | 含义 | 包含的牌型（C++ `ComboType` / Python `TYPE_*`） |
|-----:|------|-----------------------------------------------|
| 0 | 单张 | `SINGLE` = `TYPE_1_SINGLE` |
| 1 | 对子 | `PAIR` = `TYPE_2_PAIR` |
| 2 | 三条系列 | `TRIPLE`=3、`TRIPLE_1`=6（三带一）、`TRIPLE_2`=7（三带二） |
| 3 | 顺子系列 | `SERIAL_SINGLE`=8、`SERIAL_PAIR`=9、`SERIAL_TRIPLE`=10、`SERIAL_3_1`=11、`SERIAL_3_2`=12 |
| 4 | 四带二 | `QUAD_2_SINGLE`=13（四带两单）、`QUAD_2_PAIR`=14（四带两对） |
| 5 | 炸弹 | `BOMB`=4 |
| 6 | 火箭 | `KING_BOMB`=5 |

映射实现：C++ `classifyCtpCategory(ComboType)` 的 `switch`，Python `_MOVE_TYPE_TO_CTP` 字典（`TYPE_0_PASS → -1`，其他未列出类型 `.get(..., -1)`）。航天飞机（C++ `SSHUTTLE`/`SSHUTTLE_2`/`SSHUTTLE_4`，训练端不存在这些牌型）返回 -1，不参与统计。

### 4.2 "主动领出"的判定

只有**主动领出**（自由出牌）才计入 CTP，跟牌/压制不计入：

```python
# Python is_lead_action(action, card_play_action_seq)
如果 action 为空:  False
如果历史为空:      True                     # 第一手必为领出
rival = 历史最后一手；若为空(pass)，则取倒数第二手（只有一手时视为空）
返回 rival 为空
```

C++ `isLeadAction(action, history)` 的逻辑逐行对齐（末尾连续 pass 只回看一手，因为一轮最多两次 pass）。两端都把"当前动作"排除在历史之外：Python 在 `game.step()` 里用 `self.card_play_action_seq`（尚未追加当前动作），C++ 在回放循环里用 `pastHistory`（尚未 push 当前项）。

### 4.3 EMA 更新规则

对位置 $p$（绝对位置，0=landlord，1=landlord_down，2=landlord_up）与类别 $t$：

$$
\text{CTP}_p(t)\leftarrow
\begin{cases}
\gamma\cdot\text{CTP}_p(t)+(1-\gamma), & \text{该位置本次领出的类别}=t,\\[2pt]
\gamma\cdot\text{CTP}_p(t), & \text{该位置本次领出了其他类别},
\end{cases}
\qquad \gamma=\texttt{CTP\_GAMMA}=0.9 .
$$

- **初值**：全零（`init_ctp_state()` 返回 $3\times7$ 的零矩阵；C++ 局部 `float ctpState[3][7] = {}`）。
- **非领出动作完全不更新**：既不加分子，也不触发衰减——衰减只在"该位置发生一次领出"时发生。
- **只更新一个类别**：本次领出类别的值加 $(1-\gamma)$，其余 6 类在该位置整体乘 $\gamma$（Python `update_ctp` 用布尔掩码乘 0.9；C++ 循环 `if (t != cat) ctpState[pos][t] *= CTP_GAMMA`）。
- **值域** $[0,1)$：连续同类别领出 $n$ 次后趋近 $1-0.9^n$。

### 4.4 14 维特征的组成

每位对手 7 维，按相对位置拼接：

```
[CTP_up[0..6], CTP_down[0..6]]      # 共 14 维
```

自己那一行也维护（回放时三个位置都算），但不进特征。C++ `computeCtpFeatures(state, upPos, downPos)` 与 Python `get_ctp_features(ctp_state, self_pos, up_pos, down_pos)` 输出顺序一致。

### 4.5 训练端增量 vs C++ 全量回放

| 项 | Python（训练） | C++（推理） |
|----|----------------|-------------|
| 状态载体 | `Game.ctp_state`（$3\times7$），每局新建 | `computeCtpStateFromHistory` 的局部 `ctpState[3][7]`，每次决策重建 |
| 更新时机 | `game.step()` 每出一手，若 `is_lead_action` 为真则 `update_ctp` 原地更新（O(1)） | 每次推理回放整段 `state.cardPlayActionSeq`（O(H)） |
| 快照 | `get_infoset()` 里 `self.ctp_state.copy()` 给当前行动者，**不含**当前动作（无泄漏） | 回放到"当前动作之前"为止 |
| 结果 | 同一串历史、同一 γ、同一类别映射 → 状态逐位相同 | 同左 |
| 性能 | 每步 O(1) | H ≤ 约 100 手，每手最多 7 次浮点更新，估算 < 0.05 ms |

两端的唯一差别是"增量 vs 回放"的计算组织方式；历史是同一份 `(pos, cards)` 序列，状态转移确定，所以结果等价。C++ 端在每次决策重放一遍，换来的是**无跨回合状态**（KEEP_RUNNING 挂起/唤醒、状态回滚都不会留下脏数据）。

---

## 五、C5 对手行为风格（4 维）

C5 维护三个统计量：`swi_num[3]`、`swi_den[3]`、`tp_val[3]`，按绝对位置索引；初值与常数两端一致。

### 5.1 轮次状态机

回放历史时维护四个量（C++ `C5ReplayState` 的 `roundActive/prevPassPos/curLeadType/curLeadLevel`；Python 存在 `c5_state` 字典里）：

```text
roundActive = false, prevPassPos = -1, curLead = (PASS, -1)
对每个动作 (pos, cards):
    if not roundActive:
        if cards 非空:                       # 新一轮，该玩家获得控牌权
            roundActive = true; prevPassPos = -1
            curLead = (getComboTypeLevel(cards))
            TP 更新（见 5.3）
    else:
        SWI 更新（见 5.2）
        if cards 非空:
            curLead = getComboTypeLevel(cards); prevPassPos = -1
        else:
            if prevPassPos >= 0 且 prevPassPos != pos:   # 连续两次 pass（不同人）
                roundActive = false; curLead = (PASS, -1); prevPassPos = -1
            else:
                prevPassPos = pos
```

注意：SWI 对**所有位置**的跟牌机会都更新（包括自己），特征只取 up/down 两行。

### 5.2 SWI（压制意愿指数）

**符号**：

| 符号 | 含义 |
|------|------|
| $t,\ell$ | 当前领出的牌型与等级（等级 0–14；三带取三条等级、四带取四条等级、连牌取链条最高等级） |
| $N,\ n_r$ | 未知牌池总张数与等级 $r$ 的张数（**该动作发生时刻**的值） |
| $c_o$ | 该对手在该时刻的剩余张数 |
| $\text{OBP}_o$ | 该对手在该时刻的炸弹概率估计 |
| $\text{need}(t)$ | 牌型 $t$ 需要的同等级张数 |
| $d_t$ | 观测：该对手本手是否出牌（压制=1，pass=0） |

**能压制概率的估计**：

$$
p_{\text{same}}(t,\ell)=\min\!\left(1,\ \frac{1}{\text{need}(t)}\sum_{\substack{r>\ell\\ n_r\ge \text{need}(t)}}\frac{c_o\,n_r}{N}\right),
$$

$$
p_{\text{can}}(t,\ell)=1-\bigl(1-p_{\text{same}}(t,\ell)\bigr)\bigl(1-\text{OBP}_o\bigr).
$$

**特判**（两端一致，顺序也一致）：

| 情形 | $p_{\text{can}}$ | 理由 |
|------|-----------------|------|
| 牌型不可识别（pass/invalid/None） | 1.0 | 保守：认为对手总能压制 |
| 领出是火箭 | 0.0 | 火箭无法被压制 |
| 牌型可识别但没有等级（航天飞机，`kNoLeadLevel = -1` / `None`） | 1.0 | 训练端不给该牌型 rank，无法算 $p_{\text{same}}$ |
| 领出是炸弹 | $\text{OBP}_o\times 0.05$ | 只有火箭能压炸弹，OBP 中约 5% 是火箭 |
| $c_o\le0$ 或 $N\le0$ | 0.0 | 无牌可压 |

**need 表**（C++ `needForType` / Python `_NEED_FOR_TYPE`）：

| 牌型 | need | 牌型 | need |
|------|-----:|------|-----:|
| 单张 / 顺子 | 1 | 三条 / 三带一 / 三带二 / 飞机 / 飞机带单 / 飞机带对 | 3 |
| 对子 / 连对 | 2 | 炸弹 / 火箭 / 四带两单 / 四带两对 | 4 |

**EMA（只在 pass 时计算 $p_{\text{can}}$，压制时按 1.0 计权）**：

$$
\text{num}_o\leftarrow\gamma\,\text{num}_o+(1-\gamma)d_t,\qquad
\text{den}_o\leftarrow\gamma\,\text{den}_o+(1-\gamma)p_{\text{can}}^{*},\qquad
\text{SWI}_o=\frac{\text{num}_o}{\max(\text{den}_o,\ 0.01)},
$$

其中 $p_{\text{can}}^{*}=\max(p_{\text{can}},0.01)$（防除零），压制的动作直接取 $p_{\text{can}}^{*}=1.0$；$\gamma=\texttt{SWI\_GAMMA}=0.9$。初值 $\text{num}=1.0,\ \text{den}=2.0$（$\texttt{SWI\_INIT\_NUM}/\texttt{SWI\_INIT\_DEN}$），冷启动 $\text{SWI}=0.5$。分母下限 0.01、分子上限 1，取值落在 $[0,1]$ 附近，并随样本累积向真实压制率收敛。

### 5.3 TP（节奏偏好）

对手每次**获得控牌权后首手**（轮次开始的那一手，即 `!roundActive` 且动作非空）的出牌张数：

$$
\text{TP}_o\leftarrow\gamma\,\text{TP}_o+(1-\gamma)\cdot\min\!\left(\frac{n_{\text{cards}}}{6},\ 1\right),\qquad
\gamma=\texttt{TP\_GAMMA}=0.9,\quad \texttt{TP\_MAX\_CARDS}=6.0 .
$$

- 截断式归一化 $\min(n/6,1)$：出 6 张及以上都算 1。
- 初值 0.5（`tp_val[3] = {0.5, 0.5, 0.5}` / `np.full(3, 0.5)`），中性冷启动。
- Python 侧 `tp_cnt` 计数器同步累加，但只用于观测，不进特征。

### 5.4 回放/增量的时间推进（关键点）

$p_{\text{can}}$ 必须用**该动作发生时刻**的牌池与剩余张数，否则会把"动作之后才发生的信息"回填到历史动作上。两端做法：

**C++ `computeC5StateFromHistory`（全量回放）**

1. 重建"出牌阶段开始时"的牌池：完整牌库 − 当前手牌 − 我已打出的牌 −（非地主才减）底牌。此时三家都还没出过牌。
2. 各位置剩余张数初值 = 当前剩余 + 该位置已出牌张数。
3. 逐手推进：先扣该动作的牌（**别人**打出的牌从牌池扣，但跳过已在底牌列表里的牌，避免与步骤 1 的底牌扣减重复；**自己**打出的牌不扣，步骤 1 已整手扣过），再更新该位置剩余张数，然后按当刻快照算 $p_{\text{can}}$ 并更新 EMA。
4. 走完整段历史时，牌池恰等于 `buildUnknownPool(当前局面)` 的结果，与 C3 用的池自洽。

**Python `update_c5_from_history`（增量）**

1. 调用方传入**当前观测时刻**的牌池快照 `pool_vec/pool_total`、各位置剩余张数 `cards_remaining`、各对手 OBP `obp_cache`。
2. 只处理 `c5_state['last_processed']` 之后的新动作；轮次状态（`round_active/prev_pass_pos/current_lead_*`）持久化在字典里，所以不需要重放历史（避免 O(n²)）。
3. 自对弈/评估时每次观测恰好新增一个动作，因此"当前快照"就是"该动作发生时刻"的快照——与 C++ 逐步推进的结果一致。若一次性传入多个新动作，Python 会用同一个（较晚的）快照给所有新动作算 $p_{\text{can}}$，与 C++ 的逐动作快照不同；现有调用链（`env.py`、`game_eval.py`）都是逐动作观测，不会触发该差异。

### 5.5 C5 开关与 X 维度 52/56

| 项 | Python | C++ |
|----|--------|-----|
| 开关 | 环境变量 `DOUENMH_ENABLE_C5`（默认 `'1'`，模块导入时读取一次） | 无环境变量开关；由调用方是否传入非空 `c5Feat` 决定 |
| 维度 | `get_x_dim()` = `_X_DIM_BASE(52) + (C5_DIM(4) if ENABLE_C5 else 0)` | `hasC5 = (c5Feat.size() >= 4)` → `x_dim = hasC5 ? 56 : 52` |
| 关闭时 | 不计算 SWI/TP，`x_batch` 为 52 维；`reset_c5_state`/特征计算/拼接三处都会被跳过 | `feature_encoder` 的 C5 段不追加；`model_inference` 按 `x_batch.size()/batch_size` 反推维度 |
| 当前默认 | 开 → **56** | `main.cpp` 始终计算并传入 C5 → **56** |

52 = `bid_info(3) + cards_left(3) + spring(1) + bomb_num(15) + hand_stats(7) + oracle(5) + C3(4) + CTP(14)`；再加 C5 的 4 维即 56。逐段偏移见 [encoding.md](encoding.md) 第四节。

---

## 六、两端公式与常数对照总表

| 项 | C++（`cpp/src/opponent_modeling.h`） | Python（`python/douzero/env/opponent_modeling.py`） |
|----|--------------------------------------|---------------------------------------------------|
| 等级映射 | `botzoneCardToRank`：王→13/14，否则 `card/4` | `_card_to_rank`：env 码 `{3..14,17,20,30} → 0..14`；`_ENV_CARD_TO_LEVEL` 供历史 rank 换算 |
| 完整牌库 | 13×4 + 1 + 1 | `np.array([4]*13+[1,1])` |
| `RANK_WEIGHTS` | `{0,0,1,1,2,2,3,3,4,4,5,6,8,10,12}` | `[0,0,1,1,2,2,3,3,4,4,5,6,8,10,12]` |
| `OHS_MAX` | `100.0f` | `100.0` |
| OHS 公式 | `(float)oppRem / (float)poolTotal * weightedSum` | `(opponent_remaining / pool_total) * weighted_sum` |
| OHS 截断 | `std::min(raw / OHS_MAX, 1.0f)` | `min(raw / OHS_MAX, 1.0)` |
| 超几何表维度 | `[38][21]`（`OM_MAX_POOL=37`、`OM_MAX_REM=20`） | `[38][21]`（`_MAX_POOL=37`、`_MAX_REM=20`） |
| `p_bomb_single` | `C(N-4,c-4)/C(N,c)` | 同 |
| `p_rocket` | `C(N-2,c-2)/C(N,c)` | 同 |
| OBP 公式 | `1 - (1-p_rocket) * ∏(1-p_bomb_single)` | 同 |
| OBP 早退 | `c_o <= 3` 或 `poolTotal < c_o` → 0 | 同 |
| CTP 类别数 | `NUM_CTP_TYPES = 7` | `NUM_CTP_TYPES = 7` |
| `CTP_GAMMA` | `0.9f` | `0.9` |
| CTP 更新 | `γ·old + (1-γ)`；其余 6 类 ×γ | 同（布尔掩码实现） |
| CTP 初值 | 全零 | `np.zeros((3,7))` |
| CTP 输出 | `up[0..6] ++ down[0..6]`（14 维） | 同 |
| `SWI_GAMMA` / `TP_GAMMA` | `0.9f` / `0.9f` | `0.9` / `0.9` |
| SWI 初值 | `SWI_INIT_NUM=1.0f`、`SWI_INIT_DEN=2.0f` | `1.0`、`2.0` |
| TP 初值 | `tp_val[3] = {0.5f,0.5f,0.5f}` | `np.full(3, 0.5)` |
| `TP_MAX_CARDS` | `6.0f` | `6.0` |
| SWI 取值 | `num / max(den, 0.01f)` | `num / max(den, 0.01)` |
| $p_{\text{can}}$ 下限 | `max(p_can, 0.01f)` | `max(p_can, 0.01)` |
| 炸弹特判 | `opponentObp * 0.05f` | `opponent_obp * 0.05` |
| 轮次结束 | 连续两次 pass 且 `prevPassPos != pos` | 同 |
| 领出等级哨兵 | `kNoLeadLevel = -1` | `None`（`_get_combo_type_level` 返回 `(None,None)` 表示无法识别） |
| C5 开关 | 由调用方是否传 `c5Feat` 决定 | `ENABLE_C5`（`DOUENMH_ENABLE_C5`） |
| 维度常量 | `X_DIM=52`、`X_DIM_C5=56`、`X_OPPONENT=4`、`X_CTP=14`、`X_C5=4` | `_X_DIM_BASE=52`、`C5_DIM=4`、`get_x_dim()` |

> 编译期常量全部 `constexpr`，Python 侧同名常量在模块顶层定义中可直接查。

---

## 七、X 向量中的位置（结论）

| 段 | 偏移 | 维数 | 顺序 |
|----|------|-----:|------|
| C3 对手信念 | **34–37** | 4 | `OHS_up, OHS_down, OBP_up, OBP_down` |
| C4 牌型偏好 | **38–51** | 14 | `CTP_up[0..6], CTP_down[0..6]` |
| C5 行为风格 | **52–55** | 4 | `SWI_up, SWI_down, TP_up, TP_down`（仅 C5 开启时存在） |

X 总长 52（C1–C4）或 56（C1–C5），完整布局、C1/C2 段与开关同步要求见 [encoding.md](encoding.md) 第四节；`feature_encoder.h` 中 `ORACLE_OFFSET = 29` 是逐动作覆盖 C2 的偏移，其余段所有候选动作共享。

---

## 八、生命周期与状态

| 状态 | Python（训练/评估） | C++（推理） |
|------|---------------------|-------------|
| CTP | `Game.__init__`/`reset` 新建 `ctp_state`；每步 `step()` 增量更新；`get_infoset()` 把副本挂到当前行动者 | 每次决策由历史全量回放，无跨回合状态 |
| C5 | 全局 `_c5_state`，`Env.reset()` 调 `reset_c5_state()` 置空；首次使用 `get_c5_state()` 懒初始化 | 每次决策由历史全量回放，无跨回合状态 |
| C3 牌池 | 每次观测重建 | 每次决策重建 |
| 状态回滚 | 新开 `Game` 即重置 | KEEP_RUNNING 下 `rebuildFromFullHistory` 首回合重建、之后增量，回放输入始终是完整历史，因此无脏状态 |

代价（估算）：C3 建池 + 4 次计算 $O(15)$；CTP 回放 $O(H)$；C5 回放 $O(H)$ 且每个 pass 动作多一次 OBP 查表 + $O(15)$ 求和；H 为本局历史手数（≤ 约 100），整体在 0.1 ms 量级，相对神经网络推理可忽略。

---

## 九、边界与回退

| 情形 | 行为 |
|------|------|
| 牌池计数出现负值（异常输入） | 逐等级截断到 0（C++ `if (poolVec[i] < 0) poolVec[i] = 0`；Python `np.clip`） |
| 牌池为空 / 对手剩余 0 | OHS 返回 0；OBP 在 $c_o\le3$ 时返回 0；$p_{\text{can}}$ 在 $c_o\le0$ 或 $N\le0$ 时返回 0 |
| 对手剩余 < 4 张 | 不可能有炸弹，OBP 恒 0 |
| 无任何领出历史 | CTP 全零；SWI/TP 保持初值（0.5 / 0.5） |
| 历史里有航天飞机（Botzone 对手可能出） | CTP 类别 -1，不统计；C5 领出等级缺失 → $p_{\text{can}}=1.0$（保守） |
| 连续两次 pass | 轮次结束，下次非空动作按"新领出"处理（TP 更新、curLead 重置） |
| C5 关闭 | Python 不计算、不拼接这 4 维（X=52）；`get_c5_features(None,...)` 返回全零 4 维 |
| C5 状态未初始化 | `get_c5_features` 收到 `None` 时返回零向量 |
| 模型与 C5 不一致 | X 维度由数据推断（C++ `x_batch.size()/batchSize`）或由 `get_x_dim()` 决定（Python）；训练/导出/推理必须使用同一开关值 |
| 地主/农民视角 | 地主不减底牌；农民只减未打出的底牌（2.2 节） |

---

## 十、相关文件

| 文件 | 内容 |
|------|------|
| `cpp/src/opponent_modeling.h` | C++ 全部实现：牌池、超几何表、OHS/OBP、CTP 回放、C5 回放 |
| `python/douzero/env/opponent_modeling.py` | Python 实现：C3、CTP 增量更新、C5 增量更新与特征提取 |
| `python/douzero/env/env.py` | `_get_obs_resnet` 中组装 C3/C4/C5 并拼进 X |
| `python/douzero/env/game.py` | `step()` 中 CTP 增量更新、`get_infoset()` 的 CTP 快照、底牌随出牌移除 |
| `python/game_eval.py` | 评估端与 `game.py` 相同的 CTP/C5 维护逻辑 |
| `cpp/src/feature_encoder.h` | X 向量组装：C3/C4/C5 段与偏移 34/38/52 |
| `docs/encoding.md` | X 向量完整布局（本文只给 C3–C5 的结论与偏移） |
