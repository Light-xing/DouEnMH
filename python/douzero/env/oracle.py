"""
DouEnMH 递推神谕函数 — 双模神谕 / 炸弹保持度 / 优势距离

实现 oracle_functions_design.md 中定义的神谕特征计算。所有 DP 仅依赖自手牌，
兼容 DMC 训练框架（训练与推理输入一致）。

神谕特征 (5 维, X 向量 C2):
  n_active   — 主动步数 N_a   ∈ [0, 20],  归一化 x/(x+10)
  n_passive  — 被动步数 N_p   ∈ [0, 40],  归一化 x/(x+15)
  delta_n    — 控牌敏感度 ΔN  ∈ [0, 20],  归一化 x/(x+8)
  n_preserve — 炸弹保持步数   ∈ [0, ~25], 归一化 x/(x+10)
  self_adv   — 优势距离       ∈ [-20, 20], 归一化 tanh(x/5)

监督标签 (LOR 头, 3 维):
  LOR_a — N_active / (N_active + 10)   (值域 [0, 0.67))
  LOR_p — N_passive / (N_passive + 15) (值域 [0, 0.73))
  LOR_d — ΔN / (ΔN + 8)              (值域 [0, 0.71))

关键设计:
  - X 输入特征始终使用启发式 N_passive（仅依赖自手牌，DMC 兼容）
  - LOR 训练标签可使用对手感知的 N_passive（利用完美信息提高标签精度）
  - compute_oracle_raw() 只运行一次 DP；特征和标签均由此派生
"""

import ctypes
import math
import os
import random
from collections import Counter, defaultdict

import numpy as np

from douzero.env.move_generator import MovesGener
from douzero.env.move_detector import get_move_type
from douzero.env.utils import (
    TYPE_0_PASS,
    TYPE_1_SINGLE,
    TYPE_2_PAIR,
    TYPE_3_TRIPLE,
    TYPE_4_BOMB,
    TYPE_5_KING_BOMB,
    TYPE_8_SERIAL_SINGLE,
    TYPE_9_SERIAL_PAIR,
    TYPE_10_SERIAL_TRIPLE,
    MIN_SINGLE_CARDS,
    MIN_PAIRS,
    MIN_TRIPLES,
)

# ── 饱和归一化参数 (与训练标签使用相同函数) ────────────────────────────
K_N_ACTIVE = 10.0    # N_a / (N_a + 10)
K_N_PASSIVE = 15.0   # N_p / (N_p + 15)
K_DELTA_N = 8.0      # ΔN / (ΔN + 8)
K_N_PRESERVE = 10.0  # N_pres / (N_pres + 10)
K_SELF_ADV = 5.0     # tanh(self_adv / 5)

# MC SelfAdv 采样参数
MC_SELF_ADV_SAMPLES = 50

# ── 占位符模式 (DEBUG) ─────────────────────────────────────────────────────
_ORACLE_PLACEHOLDER = os.environ.get('ORACLE_PLACEHOLDER', '0') == '1'
if _ORACLE_PLACEHOLDER:
    import warnings
    warnings.warn('[ORACLE] PLACEHOLDER MODE ENABLED — all DP computations skipped')

# ── C 扩展加载 ──────────────────────────────────────────────────────────
_oracle_c_lib = None


def _load_oracle_c():
    """加载 oracle_core.dll。若 DLL 不可用则回退到纯 Python DP。"""
    global _oracle_c_lib
    if _ORACLE_PLACEHOLDER:
        return
    _dir = os.path.dirname(os.path.abspath(__file__))
    _dll_path = os.path.join(_dir, 'oracle_core.dll')
    if not os.path.exists(_dll_path):
        return  # DLL 不存在，静默回退到 Python DP
    try:
        _oracle_c_lib = ctypes.CDLL(_dll_path)
        _oracle_c_lib.dp_unified_c.argtypes = [
            ctypes.POINTER(ctypes.c_int),      # hand_vec [15]
            ctypes.POINTER(ctypes.c_int),      # combo_counts [n*15]
            ctypes.POINTER(ctypes.c_int),      # combo_masks [n]
            ctypes.POINTER(ctypes.c_int),      # combo_sizes [n]
            ctypes.POINTER(ctypes.c_uint16),   # combo_bomb_masks [n]
            ctypes.c_int,                       # n_combos
            ctypes.POINTER(ctypes.c_int),      # first_ci_for_rem [21]
            ctypes.POINTER(ctypes.c_float),    # result [2]
        ]
        _oracle_c_lib.dp_unified_c.restype = ctypes.c_int
        _oracle_c_lib.dp_clear_cache_c.argtypes = []
        _oracle_c_lib.dp_clear_cache_c.restype = None
    except OSError:
        _oracle_c_lib = None  # 加载失败，回退到 Python DP


_load_oracle_c()

# ── ctypes 缓冲区预分配 ─────────────────────────────────────────────────

_MAX_COMBOS = 25000  # 20 张手牌的 combo 数上限 (~20K, 留余量)

_counts_buf = (ctypes.c_int * (_MAX_COMBOS * 15))()
_masks_buf = (ctypes.c_int * _MAX_COMBOS)()
_sizes_buf = (ctypes.c_int * _MAX_COMBOS)()
_bomb_masks_buf = (ctypes.c_uint16 * _MAX_COMBOS)()
_first_ci_buf = (ctypes.c_int * 21)()
_result_buf = (ctypes.c_float * 2)()


def saturate(x: float, k: float) -> float:
    """饱和归一化 f(x) = x / (x + k)，值域 [0, 1)。"""
    x = max(0.0, float(x))
    return x / (x + k)


# ── 卡牌工具函数 ────────────────────────────────────────────────────────

def _get_rank(card: int) -> int:
    """env 卡牌编码 → 等级编号 (0–14)。"""
    if card < 20:
        return {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6,
                10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 17: 12}[card]
    elif card == 20:
        return 13
    else:
        return 14


def _combo_main_level(combo):
    """获取 combo 的主等级 (用于去重)。"""
    move_type = get_move_type(combo)
    t = move_type['type']
    if t in (TYPE_4_BOMB, TYPE_5_KING_BOMB, TYPE_0_PASS):
        return None
    return move_type.get('rank')


def _combo_type(combo):
    """获取 combo 的牌型类型。"""
    move_type = get_move_type(combo)
    return move_type['type']


def _is_bomb_or_rocket(combo):
    t = _combo_type(combo)
    return t in (TYPE_4_BOMB, TYPE_5_KING_BOMB)


# ── CTZ 查找表 ──────────────────────────────────────────────────────────

_CTZ_TABLE = {1 << i: i for i in range(15)}


def _iter_bomb_ranks(bomb_mask: int):
    """迭代 bomb_mask 中置位的 rank 编号。"""
    bm = bomb_mask
    while bm:
        lsb = bm & -bm
        yield _CTZ_TABLE[lsb]
        bm &= bm - 1


# ── 计数向量工具函数 ─────────────────────────────────────────────────

def _hand_to_count_vec(hand_cards):
    """手牌 → 15 维计数向量 (rank 0–14), 每维 = 该 rank 的张数。O(|H|)"""
    vec = [0] * 15
    for card in hand_cards:
        vec[_get_rank(card)] += 1
    return tuple(vec)


def _combo_to_metadata(combo):
    """combo → (size, count_vec, bitmask, bomb_mask) 元数据。"""
    cv = [0] * 15
    for card in combo:
        cv[_get_rank(card)] += 1
    size = sum(cv)
    cv = tuple(cv)
    mask = 0
    bomb_mask = 0
    for i, c in enumerate(cv):
        if c > 0:
            mask |= (1 << i)
            if c < 4:
                bomb_mask |= (1 << i)
    return size, cv, mask, bomb_mask


def _precompute_combo_data(combos):
    """预计算所有 combo 的元数据 + first_ci_for_rem 查找表。"""
    data = [_combo_to_metadata(c) for c in combos]
    data.sort(key=lambda x: x[0], reverse=True)

    n = len(data)
    first_ci_for_rem = [n] * 21  # 0..20
    ci = 0
    for r in range(20, -1, -1):
        while ci < n and data[ci][0] > r:
            ci += 1
        first_ci_for_rem[r] = ci

    return data, first_ci_for_rem


# ── 跨调用全局 DP 缓存 ─────────────────────────────────────────────────

_dp_global_cache = {}  # count_vec → (n_active, n_preserve)
_gen_moves_cache = {}  # sorted_hand_tuple → combos list


def _dp_global_cache_clear():
    """清空全局 DP 缓存和 gen_moves 缓存 (每个新 episode 调用一次)。"""
    _dp_global_cache.clear()
    _gen_moves_cache.clear()
    if _oracle_c_lib is not None:
        _oracle_c_lib.dp_clear_cache_c()


def cached_gen_moves(hand_cards):
    """带缓存的 combo 生成。"""
    key = tuple(sorted(hand_cards))
    if key in _gen_moves_cache:
        return _gen_moves_cache[key]
    mg = MovesGener(hand_cards)
    combos = mg.gen_moves()
    _gen_moves_cache[key] = combos
    return combos


# ── C 扩展 DP 包装器 ──────────────────────────────────────────────────

def _dp_unified_c(hand_vec, combo_data, first_ci_for_rem):
    """通过 ctypes 调用 C 实现的 dp_unified_c()。"""
    n = len(combo_data)
    if n > _MAX_COMBOS:
        raise RuntimeError(f"combo 数量 {n} 超出预分配缓冲区 {_MAX_COMBOS}")

    hv_arr = (ctypes.c_int * 15)(*hand_vec)

    for ci, (size, cv, mask, bm) in enumerate(combo_data):
        _sizes_buf[ci] = size
        _masks_buf[ci] = mask
        _bomb_masks_buf[ci] = bm
        off = ci * 15
        for ri in range(15):
            _counts_buf[off + ri] = cv[ri]

    for i in range(21):
        _first_ci_buf[i] = first_ci_for_rem[i]

    _oracle_c_lib.dp_unified_c(
        hv_arr, _counts_buf, _masks_buf, _sizes_buf,
        _bomb_masks_buf, n, _first_ci_buf, _result_buf)

    return (float(_result_buf[0]), float(_result_buf[1]))


# ── 纯 Python DP 实现 (回退) ──────────────────────────────────────────

def _dp_unified_py(hand_vec, combo_data, first_ci_for_rem):
    """纯 Python DP — C 扩展不可用时的回退实现。"""
    if not combo_data:
        return (0.0, 0.0)

    max_combo_cards = combo_data[0][0]
    memo = {}

    def dp(hv):
        if not any(hv):
            return (0.0, 0.0)
        if hv in memo:
            return memo[hv]

        remaining = sum(hv)
        lb = (remaining + max_combo_cards - 1) // max_combo_cards

        hmask = 0
        for i, c in enumerate(hv):
            if c:
                hmask |= (1 << i)

        best_a = float('inf')
        best_p = float('inf')

        start_ci = first_ci_for_rem[remaining]
        for ci in range(start_ci, len(combo_data)):
            if lb >= best_a and lb >= best_p:
                break

            _size, cv, cmask, bomb_mask = combo_data[ci]

            if (cmask & hmask) != cmask:
                continue

            ok = True
            rem = []
            for i in range(15):
                if cv[i] > hv[i]:
                    ok = False
                    break
                rem.append(hv[i] - cv[i])
            if not ok:
                continue

            rem = tuple(rem)

            if _size == remaining:
                val_a = 1.0
                if val_a < best_a:
                    best_a = val_a

                penalty = 0.0
                for r in _iter_bomb_ranks(bomb_mask):
                    if hv[r] == 4:
                        penalty += 1.0
                val_p = 1.0 + penalty
                if val_p < best_p:
                    best_p = val_p
                continue

            sub_a, sub_p = dp(rem)

            val_a = 1.0 + sub_a
            if val_a < best_a:
                best_a = val_a

            penalty = 0.0
            for r in _iter_bomb_ranks(bomb_mask):
                if hv[r] == 4:
                    penalty += 1.0
            val_p = 1.0 + sub_p + penalty
            if val_p < best_p:
                best_p = val_p

        memo[hv] = (best_a, best_p)
        return memo[hv]

    result = dp(hand_vec)
    _dp_global_cache.update(memo)
    return result


# ── 统一 DP 调度器 ─────────────────────────────────────────────────────

def dp_unified(hand_vec, combo_data, first_ci_for_rem):
    """单次 DP 同时计算 N_active 和 N_preserve。

    调度优先级:
      1. Python 全局缓存 — O(1) 命中
      2. C 扩展 (_dp_unified_c) — ~10ms/首帧, 65x 加速
      3. 纯 Python DP (_dp_unified_py) — 回退, ~700ms/首帧
    """
    if hand_vec in _dp_global_cache:
        return _dp_global_cache[hand_vec]

    if _oracle_c_lib is not None:
        result = _dp_unified_c(hand_vec, combo_data, first_ci_for_rem)
    else:
        result = _dp_unified_py(hand_vec, combo_data, first_ci_for_rem)

    _dp_global_cache[hand_vec] = result
    return result


def dp_min_steps(hand_cards, combos, bomb_penalty=0.0):
    """主动步数 (bomb_penalty=0) 或炸弹保持度 (bomb_penalty=1.0) DP。便捷接口。"""
    hand_vec = _hand_to_count_vec(hand_cards)
    combo_data, first_ci = _precompute_combo_data(combos)
    n_active, n_preserve = dp_unified(hand_vec, combo_data, first_ci)
    if bomb_penalty == 0.0:
        return int(n_active)
    else:
        return int(n_preserve)


# ── 被动步数 (启发式 — 推理用) ──────────────────────────────────────────

def _count_unique_levels_by_type(hand_cards):
    """分析手牌组成，统计每种牌型可领出的唯一等级数。"""
    counter = Counter(hand_cards)

    rank_cnt = [0] * 15
    for card, cnt in counter.items():
        r = _get_rank(card)
        if r < 15:
            rank_cnt[r] += cnt

    levels = {}

    levels[TYPE_1_SINGLE] = sum(1 for c in rank_cnt if c >= 1)
    levels[TYPE_2_PAIR] = sum(1 for c in rank_cnt if c >= 2)
    levels[TYPE_3_TRIPLE] = sum(1 for c in rank_cnt if c >= 3)
    levels[TYPE_4_BOMB] = sum(1 for c in rank_cnt[:13] if c == 4)
    levels[TYPE_5_KING_BOMB] = 1 if (rank_cnt[13] >= 1 and rank_cnt[14] >= 1) else 0

    n_serial_single = 0
    for i in range(15 - MIN_SINGLE_CARDS + 1):
        if all(rank_cnt[i + k] >= 1 for k in range(MIN_SINGLE_CARDS)):
            n_serial_single += 1
    levels[TYPE_8_SERIAL_SINGLE] = n_serial_single

    n_serial_pair = 0
    for i in range(15 - MIN_PAIRS + 1):
        if all(rank_cnt[i + k] >= 2 for k in range(MIN_PAIRS)):
            n_serial_pair += 1
    levels[TYPE_9_SERIAL_PAIR] = n_serial_pair

    n_serial_triple = 0
    for i in range(15 - MIN_TRIPLES + 1):
        if all(rank_cnt[i + k] >= 3 for k in range(MIN_TRIPLES)):
            n_serial_triple += 1
    levels[TYPE_10_SERIAL_TRIPLE] = n_serial_triple

    return levels


def dp_passive_steps_heuristic(hand_cards, combos):
    """被动步数启发式估计：按 combo 类型统计唯一等级数，取最大值减去炸弹数。"""
    by_type_level = defaultdict(set)
    for c in combos:
        t = _combo_type(c)
        if t in (TYPE_4_BOMB, TYPE_5_KING_BOMB, TYPE_0_PASS):
            continue
        level = _combo_main_level(c)
        if level is not None:
            by_type_level[t].add(level)

    total_bombs_and_rockets = sum(1 for c in combos if _is_bomb_or_rocket(c))

    max_chain = 0
    for t, levels in by_type_level.items():
        max_chain = max(max_chain, len(levels))

    return max(1, max_chain - total_bombs_and_rockets)


def dp_passive_opponent_aware(hand_self_cards, hand_opp_cards):
    """对手感知被动步数 (训练标签专用)。"""
    self_levels = _count_unique_levels_by_type(hand_self_cards)
    opp_levels = _count_unique_levels_by_type(hand_opp_cards)

    chain_types = [
        TYPE_1_SINGLE, TYPE_2_PAIR, TYPE_3_TRIPLE,
        TYPE_8_SERIAL_SINGLE, TYPE_9_SERIAL_PAIR, TYPE_10_SERIAL_TRIPLE,
    ]

    total_steps = 0
    for t in chain_types:
        s = self_levels.get(t, 0)
        o = opp_levels.get(t, 0)
        total_steps += min(s, o)

    bombs = self_levels.get(TYPE_4_BOMB, 0) + self_levels.get(TYPE_5_KING_BOMB, 0)

    return max(1, total_steps - bombs)


# ── 优势距离 ────────────────────────────────────────────────────────────

def estimate_self_adv_heuristic(n_active_self, opponent_remaining):
    """启发式 SelfAdv (推理用，DMC 兼容)。"""
    avg_cards_per_step = 2.8
    n_opp_est = opponent_remaining / avg_cards_per_step
    return float(n_active_self) - n_opp_est


def estimate_self_adv_mc(n_active_self, unplayed_pool, opponent_remaining,
                         num_samples=MC_SELF_ADV_SAMPLES):
    """MC 采样 SelfAdv (训练时可选，更高精度)。"""
    if not unplayed_pool or opponent_remaining <= 0:
        return float(n_active_self)

    if len(unplayed_pool) < opponent_remaining:
        opponent_remaining = len(unplayed_pool)

    min_opp_active = float('inf')
    for _ in range(num_samples):
        sampled = list(random.sample(unplayed_pool, opponent_remaining))
        combos = cached_gen_moves(sampled)
        hv = _hand_to_count_vec(sampled)
        cd, first_ci = _precompute_combo_data(combos)
        n_opp, _ = dp_unified(hv, cd, first_ci)
        if n_opp < min_opp_active:
            min_opp_active = n_opp

    return float(n_active_self) - float(min_opp_active)


# ── 完整神谕特征计算 (核心入口) ────────────────────────────────────────

def compute_oracle_raw(hand_cards, hand_opp=None):
    """从手牌计算完整神谕原始值。仅运行一次 DP。

    X 输入特征始终使用启发式 N_passive (n_passive_heuristic, delta_n_heuristic)
    以保证 DMC 训练/推理兼容。
    LOR 训练标签可使用对手感知 N_passive (n_passive, delta_n) 以提高精度。
    """
    if _ORACLE_PLACEHOLDER:
        return {
            'n_active': 5.0,
            'n_passive': 7.0,
            'n_passive_heuristic': 7.0,
            'delta_n': 2.0,
            'delta_n_heuristic': 2.0,
            'n_preserve': 6.0,
            'self_adv': 0.0,
        }

    if not hand_cards:
        return {
            'n_active': 0.0,
            'n_passive': 0.0,
            'n_passive_heuristic': 0.0,
            'delta_n': 0.0,
            'delta_n_heuristic': 0.0,
            'n_preserve': 0.0,
            'self_adv': 0.0,
        }

    combos = cached_gen_moves(hand_cards)

    hand_vec = _hand_to_count_vec(hand_cards)
    combo_data, first_ci = _precompute_combo_data(combos)

    n_active, n_preserve = dp_unified(hand_vec, combo_data, first_ci)
    n_active = float(n_active)
    n_preserve = float(n_preserve)
    n_passive_heuristic = float(dp_passive_steps_heuristic(hand_cards, combos))

    if hand_opp is not None:
        n_passive = float(dp_passive_opponent_aware(hand_cards, hand_opp))
    else:
        n_passive = n_passive_heuristic

    delta_n = max(0.0, n_passive - n_active)
    delta_n_heuristic = max(0.0, n_passive_heuristic - n_active)

    return {
        'n_active': n_active,
        'n_passive': n_passive,
        'n_passive_heuristic': n_passive_heuristic,
        'delta_n': delta_n,
        'delta_n_heuristic': delta_n_heuristic,
        'n_preserve': n_preserve,
        'self_adv': 0.0,  # 由调用方根据角色补充
    }


def normalize_oracle_features(oracle_raw):
    """将原始神谕值转为归一化特征向量 (5 维) — 放入 X 向量 C2。"""
    feat = np.zeros(5, dtype=np.float32)
    feat[0] = saturate(oracle_raw['n_active'], K_N_ACTIVE)
    feat[1] = saturate(oracle_raw['n_passive_heuristic'], K_N_PASSIVE)
    feat[2] = saturate(oracle_raw['delta_n_heuristic'], K_DELTA_N)
    feat[3] = saturate(oracle_raw['n_preserve'], K_N_PRESERVE)
    feat[4] = math.tanh(oracle_raw['self_adv'] / K_SELF_ADV)
    return feat


def compute_oracle_labels_from_raw(oracle_raw):
    """从原始神谕值计算 LOR 监督标签 (3 维)。

    使用对手感知 N_passive (n_passive, delta_n) 以提高训练标签精度。
    """
    labels = np.zeros(3, dtype=np.float32)
    labels[0] = saturate(oracle_raw['n_active'], K_N_ACTIVE)
    labels[1] = saturate(oracle_raw['n_passive'], K_N_PASSIVE)
    labels[2] = saturate(oracle_raw['delta_n'], K_DELTA_N)
    return labels


def compute_oracle_labels(hand_cards, hand_opp=None):
    """便捷接口: 从手牌直接计算 LOR 监督标签 (3 维)。"""
    oracle_raw = compute_oracle_raw(hand_cards, hand_opp)
    return compute_oracle_labels_from_raw(oracle_raw)


def compute_oracle_features(hand_cards):
    """便捷接口: 从手牌直接计算归一化神谕特征向量 (5 维)。"""
    oracle_raw = compute_oracle_raw(hand_cards)
    return normalize_oracle_features(oracle_raw)


def compute_oracle_batch(hand_cards, legal_actions, hand_opp=None, opponent_remaining=0):
    """逐动作批量计算神谕特征与标签（出牌后剩余手牌）。

    用于替代原有的每状态单次 oracle 计算。先对完整手牌做一次 DP 预热缓存，
    后续每个动作的子状态查询直接命中 memo，单次 O(1) 开销。

    返回:
      oracle_feat_list:  [(5,) ndarray] × B — 每个动作的归一化神谕特征 (X 向量 C2)
      oracle_labels:     (B, 3) ndarray      — 每个动作的 LOR 监督标签
    """
    # 预热: 完整手牌的 DP 填充全局缓存 (Python + C 双层)
    compute_oracle_raw(hand_cards, hand_opp)

    feat_list = []
    labels_list = []
    for action in legal_actions:
        # 出牌后剩余手牌 = hand − action（**多重集减法**）。
        #
        # ⚠ 修正 2026-10-18: 旧实现是
        #       hand_rem = [c for c in hand_cards if c not in action]
        #   `c not in action` 是"集合成员"判断，会把该等级的**所有副本**一起删掉：
        #   手里 3 张 5、出一对 5 时，旧实现把 3 张全删（正确应剩 1 张），
        #   于是逐动作 C2 神谕特征与 3 维 LOR 标签都系统性**偏乐观**
        #   （剩余手牌看起来更少 / n_active 更小 / self_adv 更大）。
        #   设计文档要求的是计数相减（input_design.md C2：
        #   hand_remaining = hand − action；推理部署设计文档：handCounts − actionCounts）。
        if action:
            hand_rem = list(hand_cards)
            for c in action:
                try:
                    hand_rem.remove(c)
                except ValueError:
                    # 理论上不会发生（合法动作必然是当前手牌的子集）；
                    # 万一数据异常也不让训练崩，退化为不扣这张牌。
                    break
        else:
            hand_rem = list(hand_cards)

        raw = compute_oracle_raw(hand_rem, hand_opp)
        raw['self_adv'] = estimate_self_adv_heuristic(
            n_active_self=raw['n_active'],
            opponent_remaining=opponent_remaining)

        feat_list.append(normalize_oracle_features(raw))
        labels_list.append(compute_oracle_labels_from_raw(raw))

    oracle_labels = np.stack(labels_list)  # (B, 3)
    return feat_list, oracle_labels
