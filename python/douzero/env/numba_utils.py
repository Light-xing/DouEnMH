"""
Numba JIT 加速的游戏模拟热点函数。

设计原则:
  - 保留 env.py / opponent_modeling.py 中的原始函数不变 (旧版本)
  - 本模块通过 patch_hot_functions() 在运行时替换热点函数 (新版本)
  - Numba 不可用时自动跳过, 无任何副作用
  - 模块导入时自动 patch; 设置 Numba_DISABLE=1 环境变量禁用

热点函数:
  cards2array_v2       — 8×15 卡牌编码, 每步 15-35 次调用, 最大加速目标
  _cards_to_rank_counts — rank 计数, 每步 1 次 (经过去重优化后)
  _cards2tensor_v2     — Actor 端卡牌编码 (dmc/utils.py)
"""

import os
import numpy as np

try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False

CARD_MATRIX_DIM = 120

# 是否禁用 Numba 加速
_DISABLED = os.environ.get('Numba_DISABLE', '0') == '1'


# ═══════════════════════════════════════════════════════════════════════════
# Numba JIT 核心实现
# ═══════════════════════════════════════════════════════════════════════════

if _HAS_NUMBA:

    @njit(cache=True)
    def _cards2array_v2_core(cards, n, out):
        """单手牌 8×15 编码 → 120 维 Fortran 列优先向量.

        Args:
            cards: int32[n]
            n: 牌张数
            out: int8[120] 输出缓冲区 (已清零)
        """
        if n == 0:
            return

        # 统计每列牌张数 (含大小王)
        cc = np.zeros(15, dtype=np.int32)
        for i in range(n):
            c = cards[i]
            if c < 20:
                if c <= 14:
                    col = c - 3       # 3→0, 4→1, …, 14→11
                else:
                    col = 12          # 17→12 (2)
            elif c == 20:
                col = 13              # 小王
            else:
                col = 14              # 大王
            cc[col] += 1

        # 保存大小王计数, 暂时清零 (原始断点检测排除大小王: 仅检查 c<20 的牌)
        j13, j14 = cc[13], cc[14]
        cc[13] = 0
        cc[14] = 0

        # Rows 0-3: 计数递增编码 (Fortran 列优先: out[col*8 + row])
        # 大小王列使用保存的值
        for col in range(13):
            cnt = cc[col]
            if cnt >= 4:
                out[col * 8 + 3] = 1
                out[col * 8 + 2] = 1
                out[col * 8 + 1] = 1
                out[col * 8 + 0] = 1
            elif cnt == 3:
                out[col * 8 + 2] = 1
                out[col * 8 + 1] = 1
                out[col * 8 + 0] = 1
            elif cnt == 2:
                out[col * 8 + 1] = 1
                out[col * 8 + 0] = 1
            elif cnt == 1:
                out[col * 8 + 0] = 1
        # Column 13: 小王
        if j13 >= 1:
            out[13 * 8 + 0] = 1
        # Column 14: 大王
        if j14 >= 1:
            out[14 * 8 + 0] = 1

        # Row 4: 断点标记 (仅检查普通牌列 0-12, 与原始 ranks_present 一致)
        for i in range(13):
            if cc[i] > 0 and cc[i + 1] == 0:
                out[i * 8 + 4] = 1

        # Row 5: 单顺起点 (≥5 连, ranks 0..7)
        for i in range(8):
            if (cc[i] >= 1 and cc[i+1] >= 1 and cc[i+2] >= 1 and
                cc[i+3] >= 1 and cc[i+4] >= 1):
                out[i * 8 + 5] = 1

        # Row 6: 连对起点 (≥3 连对, ranks 0..9)
        for i in range(10):
            if cc[i] >= 2 and cc[i+1] >= 2 and cc[i+2] >= 2:
                out[i * 8 + 6] = 1

        # Row 7: 飞机起点 (≥2 连三条, ranks 0..10)
        for i in range(11):
            if cc[i] >= 3 and cc[i+1] >= 3:
                out[i * 8 + 7] = 1

    @njit(cache=True)
    def _cards2array_v2_batch_core(cards_flat, offsets, out):
        """批量编码 M 组手牌 → (M, 120) 矩阵.

        Args:
            cards_flat: int32[total] 全部牌拼接
            offsets: int32[M+1] 每组牌的起止索引
            out: int8[M, 120] 输出缓冲区 (已清零)
        """
        M = len(offsets) - 1
        for m in range(M):
            start = offsets[m]
            end = offsets[m + 1]
            n = end - start
            if n == 0:
                continue

            cc = np.zeros(15, dtype=np.int32)
            for i in range(start, end):
                c = cards_flat[i]
                if c < 20:
                    if c <= 14:
                        col = c - 3
                    else:
                        col = 12
                elif c == 20:
                    col = 13
                else:
                    col = 14
                cc[col] += 1

            # 保存大小王, 清零用于断点/链检测
            j13, j14 = cc[13], cc[14]
            cc[13] = 0
            cc[14] = 0

            # Rows 0-3
            for col in range(13):
                cnt = cc[col]
                if cnt >= 4:
                    out[m, col * 8 + 3] = 1
                    out[m, col * 8 + 2] = 1
                    out[m, col * 8 + 1] = 1
                    out[m, col * 8 + 0] = 1
                elif cnt == 3:
                    out[m, col * 8 + 2] = 1
                    out[m, col * 8 + 1] = 1
                    out[m, col * 8 + 0] = 1
                elif cnt == 2:
                    out[m, col * 8 + 1] = 1
                    out[m, col * 8 + 0] = 1
                elif cnt == 1:
                    out[m, col * 8 + 0] = 1
            if j13 >= 1:
                out[m, 13 * 8 + 0] = 1
            if j14 >= 1:
                out[m, 14 * 8 + 0] = 1

            # Row 4: 断点 (仅普通牌列 0-12)
            for i in range(13):
                if cc[i] > 0 and cc[i + 1] == 0:
                    out[m, i * 8 + 4] = 1

            # Row 5: 单顺起点
            for i in range(8):
                if (cc[i] >= 1 and cc[i+1] >= 1 and cc[i+2] >= 1 and
                    cc[i+3] >= 1 and cc[i+4] >= 1):
                    out[m, i * 8 + 5] = 1

            # Row 6: 连对起点
            for i in range(10):
                if cc[i] >= 2 and cc[i+1] >= 2 and cc[i+2] >= 2:
                    out[m, i * 8 + 6] = 1

            # Row 7: 飞机起点
            for i in range(11):
                if cc[i] >= 3 and cc[i+1] >= 3:
                    out[m, i * 8 + 7] = 1

    @njit(cache=True)
    def _rank_counts_core(cards, n):
        """牌列表 → 15 维 rank 计数 (int32). 包含大小王 (rank 13, 14)."""
        cnt = np.zeros(15, dtype=np.int32)
        for i in range(n):
            c = cards[i]
            if c < 20:
                if c <= 14:
                    r = c - 3
                else:
                    r = 12
            elif c == 20:
                r = 13
            else:
                r = 14
            cnt[r] += 1
        return cnt

    # ── 预热编译 (导入时完成, 训练无编译开销) ──────────────────────────

    _dummy = np.array([3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 17, 20, 30],
                       dtype=np.int32)
    _dummy_out = np.zeros(120, dtype=np.int8)
    _cards2array_v2_core(_dummy, len(_dummy), _dummy_out)

    _dummy_offsets = np.array([0, 3, 7], dtype=np.int32)
    _dummy_batch = np.zeros((2, 120), dtype=np.int8)
    _cards2array_v2_batch_core(_dummy, _dummy_offsets, _dummy_batch)

    _dummy_rc = _rank_counts_core(_dummy, len(_dummy))

    del _dummy, _dummy_out, _dummy_offsets, _dummy_batch, _dummy_rc


# ═══════════════════════════════════════════════════════════════════════════
# 公开接口 (Numba 加速, 与原始函数签名兼容)
# ═══════════════════════════════════════════════════════════════════════════

def cards2array_v2_numba(list_cards):
    """Numba 加速版 cards2array_v2. 输入 Python list, 返回 (120,) int8."""
    n = len(list_cards)
    out = np.zeros(CARD_MATRIX_DIM, dtype=np.int8)
    if n > 0:
        cards_arr = np.array(list_cards, dtype=np.int32)
        _cards2array_v2_core(cards_arr, n, out)
    return out


def cards2array_v2_batch(list_of_card_lists):
    """批量 Numba 编码. 输入 list of list, 返回 (M, 120) int8."""
    M = len(list_of_card_lists)
    if M == 0:
        return np.zeros((0, CARD_MATRIX_DIM), dtype=np.int8)

    total = sum(len(cards) for cards in list_of_card_lists)
    cards_flat = np.empty(total, dtype=np.int32)
    offsets = np.empty(M + 1, dtype=np.int32)
    offsets[0] = 0
    idx = 0
    for j, cards in enumerate(list_of_card_lists):
        n = len(cards)
        if n > 0:
            cards_flat[idx:idx + n] = cards
        idx += n
        offsets[j + 1] = idx

    out = np.zeros((M, CARD_MATRIX_DIM), dtype=np.int8)
    _cards2array_v2_batch_core(cards_flat, offsets, out)
    return out


def rank_counts_numba(list_cards):
    """Numba 加速 rank 计数. 输入 Python list, 返回 (15,) int32."""
    n = len(list_cards)
    if n == 0:
        return np.zeros(15, dtype=np.int32)
    cards_arr = np.array(list_cards, dtype=np.int32)
    return _rank_counts_core(cards_arr, n)


def _cards2tensor_v2_numba(list_cards):
    """Numba 加速版 _cards2tensor_v2 (Actor 端). 返回 torch int8 tensor."""
    import torch
    n = len(list_cards)
    out = np.zeros(CARD_MATRIX_DIM, dtype=np.int8)
    if n > 0:
        cards_arr = np.array(list_cards, dtype=np.int32)
        _cards2array_v2_core(cards_arr, n, out)
    return torch.from_numpy(out)


# ═══════════════════════════════════════════════════════════════════════════
# 运行时 patch: 自动替换 env.py 中的热点函数
# ═══════════════════════════════════════════════════════════════════════════

def _patch_hot_functions():
    """用 Numba 版本替换 env.py 和 opponent_modeling.py 中的热点函数.

    原始函数保留在模块源码中不变 (旧版本), 仅在运行时替换引用 (新版本).
    设置环境变量 Numba_DISABLE=1 可禁用此行为.
    """
    if not _HAS_NUMBA or _DISABLED:
        return

    # ── patch env.cards2array_v2 ──
    import douzero.env.env as _env
    _env.cards2array_v2 = cards2array_v2_numba

    # ── patch opponent_modeling._cards_to_rank_counts ──
    import douzero.env.opponent_modeling as _om
    _om._cards_to_rank_counts = rank_counts_numba

    # ── patch dmc utils._cards2tensor_v2 (Actor 端) ──
    try:
        import douzero.dmc.utils as _utils
        _utils._cards2tensor_v2 = _cards2tensor_v2_numba
    except ImportError:
        pass


# 导入时自动 patch
_patch_hot_functions()
