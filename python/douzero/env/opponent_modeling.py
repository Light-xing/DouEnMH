"""
对手与队友统计建模 — C3: 对手手牌强度 (OHS) + 对手炸弹概率 (OBP)
                      C4: 对手牌型偏好 (CTP)
                      C5: 对手行为风格补充 (SWI + TP)

实现 opponent_modeling_design.md 的完整设计。
所有计算仅依赖不完美观测，训练与推理均可获取，兼容 DMC 框架。

接口:
  compute_hand_strength(cards) → float
      输入已知手牌，输出绝对手牌强度 (0~100)。

  compute_opponent_features(hand_self, cards_remaining_up, cards_remaining_down,
                            all_played_cards, three_landlord_cards) → np.ndarray(4,)
      输入不完美观测状态，输出 C3 特征 [ohs_up, ohs_down, obp_up, obp_down]。

C4 牌型偏好 (CTP):
  classify_ctp_category(action) → int
      将出牌动作映射到 7 种 CTP 类别 (0-6)。

  is_lead_action(action, card_play_action_seq) → bool
      判断该动作是否为"主动领出"。

  init_ctp_state() → np.ndarray
      初始化 CTP 状态矩阵 (3 绝对位置 × 7 类别)。

  update_ctp(ctp_state, pos_idx, category) → None
      以 EMA (gamma=0.9) 更新 CTP 状态。

  get_ctp_features(ctp_state, self_pos, up_pos, down_pos) → np.ndarray(14,)
      从 CTP 状态提取相对位置特征 [ctp_up_0..6, ctp_down_0..6]。

C5 对手行为风格 (SWI + TP):
  全局开关:
    ENABLE_C5 — 通过环境变量 DOUENMH_ENABLE_C5 控制 (默认开启)

  状态管理:
    reset_c5_state() — 每局开始时清空 C5 状态 (由 env.reset 调用)
    get_c5_state() → dict — 获取全局 C5 状态

  特征提取:
    update_c5_from_history(card_play_action_seq, c5_state,
                           pool_vec, pool_total,
                           cards_remaining, obp_cache) → None
        增量解析出牌历史，使用 p_can 加权更新 SWI/TP EMA 统计。
        调用方需传入未知牌池快照和对手 OBP 以供 p_can 计算。

    get_c5_features(c5_state, self_pos, up_pos, down_pos) → np.ndarray(4,)
        从 C5 状态提取特征 [swi_up, swi_down, tp_up, tp_down]。
"""

import math
import os
import numpy as np
from douzero.env.utils import (
    TYPE_0_PASS, TYPE_1_SINGLE, TYPE_2_PAIR, TYPE_3_TRIPLE,
    TYPE_4_BOMB, TYPE_5_KING_BOMB, TYPE_6_3_1, TYPE_7_3_2,
    TYPE_8_SERIAL_SINGLE, TYPE_9_SERIAL_PAIR, TYPE_10_SERIAL_TRIPLE,
    TYPE_11_SERIAL_3_1, TYPE_12_SERIAL_3_2, TYPE_13_4_2, TYPE_14_4_22,
)

# ── 完整牌池 (54 张, env 编码) ──────────────────────────────────────────
_FULL_DECK = []
for i in range(3, 15):
    _FULL_DECK.extend([i] * 4)
_FULL_DECK.extend([17] * 4)
_FULL_DECK.extend([20, 30])

# ── Rank 权重表 ──────────────────────────────────────────────────────────
# w[r] = 该等级牌在控牌和压制中的相对价值
# 等级:  3  4  5  6  7  8  9 10  J  Q  K  A  2  小王 大王
# rank:  0  1  2  3  4  5  6  7  8  9 10 11 12  13  14
RANK_WEIGHTS = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 8, 10, 12], dtype=np.float32)

# OHS 归一化上界: 略大于理论最大期望值 (~79)，为极端分布留余量
OHS_MAX = 100.0

# ── 预计算超几何概率表 ──────────────────────────────────────────────────
# P_bomb_single[N][c_o] = 对手恰好持有一个特定 rank 的 4 张牌 (即炸弹) 的概率
#   = C(N-4, c_o-4) / C(N, c_o),  其中 N ∈ [4, 37], c_o ∈ [4, 20]
# P_rocket[N][c_o] = 对手同时持有大小王 (火箭) 的概率
#   = (c_o/N) * ((c_o-1)/(N-1)),  当 pool 中两种 joker 各 1 张
#
# 参数维度极小 (38×21), 模块加载时一次性预计算, 推理时纯查表 O(1).

_MAX_POOL = 37   # 未知牌池最多 54-17=37 张 (己方 17 张后)
_MAX_REM  = 20   # 对手最多 20 张 (地主)

_P_BOMB_SINGLE = [[0.0] * (_MAX_REM + 1) for _ in range(_MAX_POOL + 1)]
_P_ROCKET = [[0.0] * (_MAX_REM + 1) for _ in range(_MAX_POOL + 1)]

for _N in range(4, _MAX_POOL + 1):
    for _c in range(4, min(_N, _MAX_REM) + 1):
        _P_BOMB_SINGLE[_N][_c] = float(
            math.comb(_N - 4, _c - 4) / math.comb(_N, _c)
        )
# P_ROCKET 从 N=2 开始 (最少 2 张: 大小王各 1)
for _N in range(2, _MAX_POOL + 1):
    for _c in range(2, min(_N, _MAX_REM) + 1):
        _P_ROCKET[_N][_c] = float(
            math.comb(_N - 2, _c - 2) / math.comb(_N, _c)
        )


def _comb(n: int, k: int) -> float:
    """计算组合数 C(n, k)."""
    if k < 0 or k > n:
        return 0.0
    return float(math.comb(n, k))


# ── 卡牌 → Rank 映射 ────────────────────────────────────────────────────

def _card_to_rank(card: int) -> int:
    """env 卡牌编码 → 等级编号 0–14 (3→0, 4→1, …, 2→12, 20→13, 30→14)."""
    if card < 20:
        return {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
                10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 17: 12}[card]
    elif card == 20:
        return 13
    else:
        return 14


# env 卡牌码 → 等级 0..14（3→0, 4→1, …, 14→11, 17→12, 20→13, 30→14）
# move_detector.get_move_type() 返回的 'rank' 是 **env 卡牌码**，
# 而牌池计数向量 / 设计文档 §2.3 的 ℓ 都是**等级**，两者必须显式换算。
_ENV_CARD_TO_LEVEL = {3 + i: i for i in range(12)}
_ENV_CARD_TO_LEVEL.update({17: 12, 20: 13, 30: 14})


def _cards_to_rank_counts(cards) -> np.ndarray:
    """牌列表 → 15 维 rank 计数向量."""
    cnt = np.zeros(15, dtype=np.int32)
    for c in cards:
        cnt[_card_to_rank(c)] += 1
    return cnt


# ── 未知牌池构建 ────────────────────────────────────────────────────────

def _build_unknown_pool(hand_self, all_played_cards, three_landlord_cards,
                        is_landlord: bool = False):
    """构建未知牌池的 rank 计数向量.

    未知牌池 = 完整牌库 − 己方手牌 − 所有已出牌 − 三张底牌

    Args:
        is_landlord: 观察者是否是地主。**三张底牌是地主手牌的一部分**
            （opponent_modeling_design.md §1.2 注："D: 三张底牌（完全可见，
            地主手牌的一部分）"），所以地主视角下 hand_self 已经包含底牌，
            不能再把底牌扣一次（否则牌池少 1~3 张，且由于 np.clip 截断
            表现为"偏差随局面不均匀"）。农民视角底牌不属于自己，必须扣除。
            该参数默认 False 以保持旧调用点行为。

    使用 rank 计数向量做减法 (env 编码中相同数字代表同 rank 不同花色的牌,
    如 hand_self = [3, 3, 3] 表示 3 张 rank 3 的牌).

    Returns:
        pool_vec: np.ndarray(15,) — 每个 rank 在未知牌池中的张数
        pool_total: int — 未知牌池总张数
    """
    # 完整牌库的 rank 计数: 普通 rank (0-12) 各 4 张, 大小王各 1 张
    full_counts = np.array([4]*13 + [1, 1], dtype=np.int32)

    known_counts = _cards_to_rank_counts(hand_self)
    for cards in all_played_cards.values():
        known_counts += _cards_to_rank_counts(cards)
    if not is_landlord:
        # 地主视角: 底牌已在 hand_self 内, 再减一次是重复扣减 (旧行为的 bug)
        known_counts += _cards_to_rank_counts(three_landlord_cards)

    pool_vec = full_counts - known_counts
    # 防御: 正常局面下计数不会为负 (同一张牌不会同时出现在己方手牌与已出牌里),
    # 但异常输入 (如底牌被地主打出而被重复扣减) 下可能出现负值。
    # 与 C++ 端 buildUnknownPool 的防御截断保持一致。
    np.clip(pool_vec, 0, None, out=pool_vec)
    pool_total = int(pool_vec.sum())
    return pool_vec, pool_total


# ── 核心特征计算 ────────────────────────────────────────────────────────

def compute_hand_strength(cards) -> float:
    """计算已知手牌的绝对强度 (0~100)."""
    cnt = _cards_to_rank_counts(cards)
    return float(np.dot(cnt, RANK_WEIGHTS))


def _compute_ohs_from_pool(pool_vec, pool_total, opponent_remaining):
    """从已构建的未知牌池计算 OHS (内部函数, 避免重复建池)."""
    if opponent_remaining <= 0 or pool_total <= 0:
        return 0.0
    weighted_sum = float(np.dot(pool_vec, RANK_WEIGHTS))
    return (opponent_remaining / pool_total) * weighted_sum


def _compute_obp_from_pool(pool_vec, pool_total, opponent_remaining):
    """从已构建的未知牌池计算 OBP — 纯查表, 无 math.comb 调用."""
    c_o = opponent_remaining
    if c_o <= 3 or pool_total < c_o:
        return 0.0

    N = pool_total
    # 火箭: 需未知池中同时存在大小王
    p_rocket = 0.0
    if pool_vec[13] >= 1 and pool_vec[14] >= 1 and N <= _MAX_POOL and c_o <= _MAX_REM:
        p_rocket = _P_ROCKET[N][c_o]

    # 普通炸弹: 遍历所有 n_r = 4 的 rank, 乘独立近似
    p_no_bomb = 1.0
    p_bomb = _P_BOMB_SINGLE[N][c_o] if N <= _MAX_POOL and c_o <= _MAX_REM else 0.0
    for r in range(13):
        if pool_vec[r] == 4:
            p_no_bomb *= (1.0 - p_bomb)

    return float(1.0 - (1.0 - p_rocket) * p_no_bomb)


def compute_ohs(hand_self, opponent_remaining, all_played_cards,
                three_landlord_cards, is_landlord: bool = False):
    """估计对手手牌强度 OHS — 公开接口."""
    if opponent_remaining <= 0:
        return 0.0
    pool_vec, pool_total = _build_unknown_pool(hand_self, all_played_cards,
                                              three_landlord_cards,
                                              is_landlord=is_landlord)
    return _compute_ohs_from_pool(pool_vec, pool_total, opponent_remaining)


def compute_obp(hand_self, opponent_remaining, all_played_cards,
                three_landlord_cards, is_landlord: bool = False):
    """估计对手至少持有一个炸弹的概率 OBP — 公开接口."""
    if opponent_remaining <= 3:
        return 0.0
    pool_vec, pool_total = _build_unknown_pool(hand_self, all_played_cards,
                                              three_landlord_cards,
                                              is_landlord=is_landlord)
    return _compute_obp_from_pool(pool_vec, pool_total, opponent_remaining)


def compute_opponent_features(hand_self,
                               cards_remaining_up: int,
                               cards_remaining_down: int,
                               all_played_cards: dict,
                               three_landlord_cards,
                               is_landlord: bool = False):
    """计算 C3 对手特征 (4 维) — 单次建池, 复用给 4 个子计算.

    Args:
        hand_self: 己方手牌列表
        cards_remaining_up: 上手剩余牌数
        cards_remaining_down: 下手剩余牌数
        all_played_cards: dict {pos: [cards]} 各玩家已出牌
        three_landlord_cards: 三张底牌列表
        is_landlord: 观察者是否地主（地主视角不再重复扣底牌，见 _build_unknown_pool）

    Returns:
        (features, pool_vec, pool_total):
          features: np.ndarray(4,), dtype=float32
          pool_vec: np.ndarray(15,) int32
          pool_total: int
    """
    # 单次构建未知牌池, 避免 4 次重复计算
    pool_vec, pool_total = _build_unknown_pool(hand_self, all_played_cards,
                                              three_landlord_cards,
                                              is_landlord=is_landlord)

    ohs_up = _compute_ohs_from_pool(pool_vec, pool_total, cards_remaining_up)
    ohs_down = _compute_ohs_from_pool(pool_vec, pool_total, cards_remaining_down)
    obp_up = _compute_obp_from_pool(pool_vec, pool_total, cards_remaining_up)
    obp_down = _compute_obp_from_pool(pool_vec, pool_total, cards_remaining_down)

    features = np.array([
        min(ohs_up / OHS_MAX, 1.0),
        min(ohs_down / OHS_MAX, 1.0),
        obp_up,
        obp_down,
    ], dtype=np.float32)
    return features, pool_vec, pool_total


# ═══════════════════════════════════════════════════════════════════════════════
# C5: 对手行为风格补充 — SWI (压制意愿指数) + TP (节奏偏好)
# 设计文档: opponent_modeling_design.md §2.3-2.4
#
# SWI (Suppression Willingness Index):
#   追踪对手在 respond 位置面对领出时的压制倾向.
#   SWI = EMA_num / max(EMA_den, 0.01)
#     EMA_num ← γ·num + (1-γ)·d_t                  d_t ∈ {0,1} 实际是否压制
#     EMA_den ← γ·den + (1-γ)·p_can(t,ℓ)           p_can ∈ [0,1] 能压制的估计概率
#   初始: num=1, den=2 (Laplace 平滑, 从 0.5 冷启动)
#
#   p_can(t,ℓ) = 1 − (1 − p_same(t,ℓ)) · (1 − OBP_o)
#     p_same(t,ℓ) = min(1.0, (1/need(t)) · Σ_{r > ℓ, n_r >= need(t)} c_o·n_r/N)
#     其中 t=领出牌型, ℓ=领出等级, c_o=对手剩余张数, N=未知牌池总张数,
#         n_r=未知牌池中 rank r 的张数, OBP_o=对手炸弹概率
#
# TP (Tempo Preference):
#   追踪对手获得控牌权后领出的平均牌张数.
#   TP ← γ·TP + (1-γ)·min(n_cards/6, 1)  (EMA, 归一化 ÷6)
#   初始: 0.5 (中性)
#
#   注: 设计文档定义 TP 为简单算术平均; 此处使用 EMA 以自适应对手风格漂移.
# ═══════════════════════════════════════════════════════════════════════════════

ENABLE_C5 = os.environ.get('DOUENMH_ENABLE_C5', '1') == '1'
C5_DIM = 4  # swi_up, swi_down, tp_up, tp_down

SWI_GAMMA = 0.9
TP_GAMMA = 0.9
SWI_INIT_NUM = 1.0
SWI_INIT_DEN = 2.0
TP_MAX_CARDS = 6.0

_X_DIM_BASE = 52


def get_x_dim():
    """返回当前配置下的 X 向量维度."""
    return _X_DIM_BASE + (C5_DIM if ENABLE_C5 else 0)


# ── p_can 计算所需常量 ────────────────────────────────────────────────────

# need(t): 牌型 t 所需的最少同等级张数 (用于 p_same 分母归一化)
_NEED_FOR_TYPE = {
    TYPE_1_SINGLE: 1,
    TYPE_2_PAIR: 2,
    TYPE_3_TRIPLE: 3,
    TYPE_6_3_1: 3,
    TYPE_7_3_2: 3,
    TYPE_4_BOMB: 4,
    TYPE_5_KING_BOMB: 4,
    TYPE_8_SERIAL_SINGLE: 1,
    TYPE_9_SERIAL_PAIR: 2,
    TYPE_10_SERIAL_TRIPLE: 3,
    TYPE_11_SERIAL_3_1: 3,
    TYPE_12_SERIAL_3_2: 3,
    TYPE_13_4_2: 4,
    TYPE_14_4_22: 4,
}

# 绝对位置 → 索引映射
_ABS_POS_IDX = {'landlord': 0, 'landlord_down': 1, 'landlord_up': 2}

# 全局 C5 状态 (每局 reset 时清空)
_c5_state = None


def _get_combo_type_level(cards):
    """从出牌列表提取 (combo_type, main_level).

    用于 p_can 计算中识别当前领出的牌型和等级。
    返回 (type_int, **等级 0..14**) 或 (None, None) 表示 pass/无法识别。

    ⚠ 修正 2026-10-18: `get_move_type()` 返回的 `rank` 是 **env 卡牌码**
    （3..14 / 17 / 20 / 30），而 `_estimate_p_can` 的 ℓ 与未知牌池 n_r 都是
    **等级 0..14** 索引。旧实现直接把 env 码当等级传给 p_can，导致 `Σ_{r>ℓ}`
    扫错区间：例如单张 A 的 env 码是 14，循环 `range(15, 15)` 直接为空
    → p_same 恒 0（认为对手永远压不住）；单张 3 的 env 码是 3，则会扫
    等级 4..14（"7"~大王），漏掉"4/5/6"。这里统一换算成等级，
    与设计文档 §2.3（ℓ = 领出等级、n_r = 牌池等级计数）保持一致。
    """
    if not cards:
        return None, None
    try:
        from douzero.env.move_detector import get_move_type
        mt = get_move_type(cards)
        t = mt['type']
        if t in (TYPE_0_PASS,):
            return None, None
        rank_env = mt.get('rank')
        level = _ENV_CARD_TO_LEVEL.get(rank_env) if rank_env is not None else None
        # 连续牌型(顺子/连对/飞机/飞机带翼): move_detector 的 rank 是链条**起始(最低)**牌,
        # 而 p_can 的 ℓ 需要"这张组合的比较等级"= 链条**最高**牌
        # （与 C++ Combo::main_level 口径一致, 否则两端的 Σ_{r>ℓ} 会错开 len-1 个等级）.
        if level is not None and t in (TYPE_8_SERIAL_SINGLE, TYPE_9_SERIAL_PAIR,
                                       TYPE_10_SERIAL_TRIPLE, TYPE_11_SERIAL_3_1,
                                       TYPE_12_SERIAL_3_2):
            _len = int(mt.get('len') or 1)
            level = min(level + _len - 1, 14)
        return t, level
    except Exception:
        return None, None


def _estimate_p_can(lead_type, lead_level, pool_vec, pool_total,
                    opponent_remaining, opponent_obp):
    """估计对手能压制当前领出的概率 p_can(t, ℓ).

    p_can(t,ℓ) = 1 − (1 − p_same(t,ℓ)) · (1 − OBP_o)

    Args:
        lead_type: 领出牌型 (TYPE_* 常量)
        lead_level: 领出等级 (0-14, 可为 None)
        pool_vec: 未知牌池 rank 计数 (15,)
        pool_total: 未知牌池总张数
        opponent_remaining: 对手剩余张数
        opponent_obp: 对手至少持有一个炸弹的概率 OBP_o ∈ [0,1]

    Returns:
        float: p_can ∈ [0, 1]
    """
    if lead_type is None:
        return 1.0  # 无法识别牌型 → 保守估计总能压制

    # 火箭不可被压制。
    # ⚠ 修正 2026-10-18: 必须在"等级缺失"判断**之前**。
    # get_move_type() 对火箭不返回 'rank'（实测 {'type': 5}），
    # 旧实现先判 `lead_level is None` → 直接返回 1.0，下面这句 0.0 永远走不到，
    # 与设计文档 §2.3「火箭不可被压制 → p_can = 0」矛盾。
    if lead_type == TYPE_5_KING_BOMB:
        return 0.0

    if lead_level is None:
        # 牌型可识别但取不到等级（如航天飞机, move_detector 不给 rank）
        # → 保守估计总能压制（设计文档未规定该牌型的 p_can）
        return 1.0

    # 炸弹: 仅火箭可压制 (OBP 已覆盖炸弹概率, 但火箭概率需单独考虑)
    if lead_type == TYPE_4_BOMB:
        return opponent_obp * 0.05  # OBP 中约 5% 是火箭

    if opponent_remaining <= 0 or pool_total <= 0:
        return 0.0

    need = _NEED_FOR_TYPE.get(lead_type, 1)

    # p_same(t,ℓ) = min(1.0, (1/need) · Σ_{r > ℓ, n_r >= need} c_o·n_r/N)
    p_same_sum = 0.0
    for r in range(lead_level + 1, 15):
        n_r = pool_vec[r]
        if n_r >= need:
            p_same_sum += (opponent_remaining * n_r) / pool_total

    p_same = min(1.0, p_same_sum / need)

    # p_can = 1 − (1 − p_same) · (1 − OBP)
    return 1.0 - (1.0 - p_same) * (1.0 - opponent_obp)


def init_c5_state():
    """初始化 C5 状态字典.

    round_active / prev_pass_pos / current_lead_* 存储当前轮次状态,
    避免每次观测都从历史头部重放 (消除 O(n²) 累计开销)。
    """
    return {
        'swi_num': np.full(3, SWI_INIT_NUM, dtype=np.float32),
        'swi_den': np.full(3, SWI_INIT_DEN, dtype=np.float32),
        'tp_val': np.full(3, 0.5, dtype=np.float32),
        'tp_cnt': np.zeros(3, dtype=np.float32),
        'last_processed': 0,
        # 持久化轮次状态 — 消除 O(n²) 重放
        'round_active': False,
        'prev_pass_pos': None,
        'current_lead_type': None,
        'current_lead_level': None,
    }


def reset_c5_state():
    """每局开始时清空 C5 状态 (由 env.reset 调用)."""
    global _c5_state
    _c5_state = None


def get_c5_state():
    """获取或延迟初始化全局 C5 状态."""
    global _c5_state
    if _c5_state is None:
        _c5_state = init_c5_state()
    return _c5_state


def update_c5_from_history(card_play_action_seq, c5_state,
                           pool_vec, pool_total,
                           cards_remaining, obp_cache):
    """增量解析出牌历史，使用 p_can 加权更新 SWI/TP EMA 统计.

    仅处理 c5_state['last_processed'] 之后的新动作。
    轮次状态 (round_active / prev_pass_pos / current_lead_*) 持久化于
    c5_state 中，消除每步 O(n) 历史重放。

    Args:
        card_play_action_seq: [(pos, cards), ...] 完整出牌历史
        c5_state: C5 状态字典
        pool_vec: 未知牌池 rank 计数 (15,) — 当前观测时刻的快照
        pool_total: 未知牌池总张数
        cards_remaining: dict {pos: remaining_count} — 各玩家剩余张数
        obp_cache: dict {pos: obp_value} — 各玩家当前 OBP 估计值 (已计算好的)
    """
    if c5_state is None or not card_play_action_seq:
        return

    start = c5_state['last_processed']
    if start >= len(card_play_action_seq):
        return

    seq = card_play_action_seq

    # 从持久化状态恢复轮次上下文 (无需重放历史)
    round_active = bool(c5_state.get('round_active', False))
    prev_pass_pos = c5_state.get('prev_pass_pos', None)
    cur_lead_type = c5_state.get('current_lead_type', None)
    cur_lead_level = c5_state.get('current_lead_level', None)

    for i in range(start, len(seq)):
        pos, cards = seq[i]
        idx = _ABS_POS_IDX[pos]

        if not round_active:
            if cards:
                # 新一轮: 该玩家获得控牌权 → TP 更新
                round_active = True
                prev_pass_pos = None
                cur_lead_type, cur_lead_level = _get_combo_type_level(cards)
                n_cards = len(cards)
                old_val = float(c5_state['tp_val'][idx])
                norm_cards = min(n_cards / TP_MAX_CARDS, 1.0)
                c5_state['tp_val'][idx] = TP_GAMMA * old_val + (1.0 - TP_GAMMA) * norm_cards
                c5_state['tp_cnt'][idx] += 1.0
        else:
            # 轮次激活中: respond 机会 → SWI 更新 (p_can 加权分母)
            did_suppress = 1.0 if cards else 0.0

            # 计算 p_can: 对手能压制当前领出的估计概率
            p_can = 1.0
            if not cards:
                # 仅 pass 时需要 p_can 加权; 压制时权重不影响分子/分母比例
                opp_remaining = cards_remaining.get(pos, 0)
                opp_obp = obp_cache.get(pos, 0.0)
                p_can = _estimate_p_can(
                    cur_lead_type, cur_lead_level,
                    pool_vec, pool_total,
                    opp_remaining, opp_obp
                )
                # 夹紧到合理范围, 防止 p_can=0 导致除零
                p_can = max(p_can, 0.01)

            c5_state['swi_num'][idx] = (
                SWI_GAMMA * float(c5_state['swi_num'][idx])
                + (1.0 - SWI_GAMMA) * did_suppress
            )
            c5_state['swi_den'][idx] = (
                SWI_GAMMA * float(c5_state['swi_den'][idx])
                + (1.0 - SWI_GAMMA) * p_can
            )

            if cards:
                # 压制 — 更新当前领出为压制者的牌 (新的压制目标)
                cur_lead_type, cur_lead_level = _get_combo_type_level(cards)
                prev_pass_pos = None
            else:
                # pass — 检查是否轮次结束 (连续两次 pass 且非同一人)
                if prev_pass_pos is not None and prev_pass_pos != pos:
                    round_active = False
                    cur_lead_type = None
                    cur_lead_level = None
                    prev_pass_pos = None
                else:
                    prev_pass_pos = pos

    # 持久化轮次状态
    c5_state['round_active'] = round_active
    c5_state['prev_pass_pos'] = prev_pass_pos
    c5_state['current_lead_type'] = cur_lead_type
    c5_state['current_lead_level'] = cur_lead_level
    c5_state['last_processed'] = len(seq)


def get_c5_features(c5_state, self_pos, up_pos, down_pos):
    """从 C5 状态提取 4 维特征向量 [swi_up, swi_down, tp_up, tp_down].

    Args:
        c5_state: C5 状态字典 (init_c5_state 的输出)
        self_pos: 当前玩家绝对位置
        up_pos: 上手绝对位置
        down_pos: 下手绝对位置

    Returns:
        np.ndarray(4,) dtype=float32
    """
    if c5_state is None:
        return np.zeros(C5_DIM, dtype=np.float32)

    up_idx = _ABS_POS_IDX[up_pos]
    down_idx = _ABS_POS_IDX[down_pos]

    swi_up = float(c5_state['swi_num'][up_idx]) / max(float(c5_state['swi_den'][up_idx]), 0.01)
    swi_down = float(c5_state['swi_num'][down_idx]) / max(float(c5_state['swi_den'][down_idx]), 0.01)
    tp_up = float(c5_state['tp_val'][up_idx])
    tp_down = float(c5_state['tp_val'][down_idx])

    return np.array([swi_up, swi_down, tp_up, tp_down], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════════
# C4: 对手牌型偏好 CTP (Combo-Type Preference)
# 设计文档: opponent_modeling_design.md §2.2
# 追踪每位对手主动领出时的牌型选择频率分布 (EMA, γ=0.9).
# ═══════════════════════════════════════════════════════════════════════════════

CTP_GAMMA = 0.9
NUM_CTP_TYPES = 7

# 15 种 move_type → 7 类 CTP 映射表
_MOVE_TYPE_TO_CTP = {
    TYPE_0_PASS: -1,                  # pass 不参与统计
    TYPE_1_SINGLE: 0,                 # 单张
    TYPE_2_PAIR: 1,                   # 对子
    TYPE_3_TRIPLE: 2,                 # 三条系列
    TYPE_6_3_1: 2,
    TYPE_7_3_2: 2,
    TYPE_8_SERIAL_SINGLE: 3,          # 顺子系列
    TYPE_9_SERIAL_PAIR: 3,
    TYPE_10_SERIAL_TRIPLE: 3,
    TYPE_11_SERIAL_3_1: 3,
    TYPE_12_SERIAL_3_2: 3,
    TYPE_13_4_2: 4,                   # 四带二
    TYPE_14_4_22: 4,
    TYPE_4_BOMB: 5,                   # 炸弹
    TYPE_5_KING_BOMB: 6,              # 火箭
}


def classify_ctp_category(action):
    """将出牌动作映射到 7 种 CTP 牌型类别 (0-6).

    类别:
        0: 单张 (Solo)
        1: 对子 (Pair)
        2: 三条系列 (Trio / 三带一 / 三带二)
        3: 顺子系列 (单顺 / 双顺 / 飞机 / 飞机带)
        4: 四带二 (四带两单 / 四带两对)
        5: 炸弹 (Bomb)
        6: 火箭 (Rocket)

    Returns:
        int: 0-6 类别索引, -1 表示 pass 或无法分类.
    """
    from douzero.env.move_detector import get_move_type
    move_type = get_move_type(action)['type']
    return _MOVE_TYPE_TO_CTP.get(move_type, -1)


def is_lead_action(action, card_play_action_seq):
    """判断该动作是否为"主动领出" (自由出牌, gen_moves).

    与 game.py `get_legal_card_play_actions()` 中的 rival_move 判定逻辑一致:
      - rival_move = 最后一个非 pass 动作 (跳过末尾连续的 pass)
      - 若 rival_move 为空 (= 无历史 或 连续两个 pass 结束回合) → 领出

    仅检查 card_play_action_seq[-1][1] 为 pass 会误将单次 pass 后的
    跟牌 (响应) 判为领出, 因为一轮中第一个 pass 后其他玩家仍需跟牌.

    Args:
        action: 当前动作牌列表 ([] = pass)
        card_play_action_seq: 出牌历史 [(pos, action), ...] — 不含当前动作

    Returns:
        bool
    """
    if len(action) == 0:
        return False
    if len(card_play_action_seq) == 0:
        return True            # 游戏第一手: 必为领出
    # 确定 rival_move: 若上一手是 pass, 再往前看一手
    if len(card_play_action_seq[-1][1]) == 0:
        if len(card_play_action_seq) >= 2:
            rival_move = card_play_action_seq[-2][1]
        else:
            rival_move = []    # 仅有一手历史且为 pass (不应出现, 防御)
    else:
        rival_move = card_play_action_seq[-1][1]
    # rival_move 为空 → 自由出牌 → 领出
    return len(rival_move) == 0


def init_ctp_state():
    """初始化 CTP 状态矩阵 (3 绝对位置 × 7 类别), 全零启动.

    Returns:
        np.ndarray(3, 7) dtype=float32
    """
    return np.zeros((3, NUM_CTP_TYPES), dtype=np.float32)


def update_ctp(ctp_state, pos_idx, category):
    """以 EMA 更新 CTP 状态.

    CTP_o(t) = γ · CTP_o(t) + (1-γ) · 1[lead_type = t]

    Args:
        ctp_state: np.ndarray(3, 7) — 原地更新
        pos_idx: 绝对位置索引 (0=landlord, 1=landlord_down, 2=landlord_up)
        category: 0-6 CTP 类别
    """
    if category < 0 or category >= NUM_CTP_TYPES:
        return
    # 仅更新当前领出对应的类别, 其余类别隐式衰减
    old_val = ctp_state[pos_idx, category]
    ctp_state[pos_idx, category] = CTP_GAMMA * old_val + (1.0 - CTP_GAMMA) * 1.0
    # 其余 6 个类别做纯衰减
    mask = np.ones(NUM_CTP_TYPES, dtype=bool)
    mask[category] = False
    ctp_state[pos_idx, mask] *= CTP_GAMMA


def get_ctp_features(ctp_state, self_pos, up_pos, down_pos):
    """从 CTP 状态提取相对位置特征.

    Args:
        ctp_state: np.ndarray(3, 7) — 绝对位置 CTP
        self_pos: 当前玩家绝对位置 (landlord/landlord_down/landlord_up)
        up_pos: 上手绝对位置
        down_pos: 下手绝对位置

    Returns:
        np.ndarray(14,) — [ctp_up_0..6, ctp_down_0..6], dtype=float32
    """
    up_idx = _ABS_POS_IDX[up_pos]
    down_idx = _ABS_POS_IDX[down_pos]
    return np.concatenate([ctp_state[up_idx], ctp_state[down_idx]]).astype(np.float32)
