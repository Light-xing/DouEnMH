// =============================================================================
// oracle.cpp — 神谕函数实现 (Botzone C++ 部署)
//
// 实现 oracle.h 中声明的全部神谕计算函数:
//   - generate_all_combos()     组合生成 (CardCount 计数向量空间)
//   - dp_unified()              统一 DP (N_active + N_preserve 同时计算)
//   - dp_passive_steps_heuristic() 被动步数启发式
//   - compute_oracle_raw()      主入口: 手牌→神谕原始值
//   - build_hand_counts()       Botzone Card ID → CardCount
//   - estimate_opponent_n_active() SelfAdv 启发式
//
// DP 优化 (11 项):
//   1. 15-dim int 计数向量, O(1) 哈希/减法
//   2. 位掩码预过滤 (cmask & hmask) == cmask
//   3. OracleCombo按 size 降序排列, 大牌型优先
//   4. 下界剪枝 ceil(remaining/max_combo_size) >= best → break
//   5. 单次递归双输出 (N_active, N_preserve)
//   6. 全局哈希表缓存, 线性探测
//   7. 贪心初始上界
//   8. bomb_mask (uint16) + ctz 遍历拆弹 rank
//   9. first_ci_for_rem[21] 索引表, 跳过 oversized combos
//  10. 合并 hash+hmask 计算, 单次遍历
//  11. 单 combo 快速路径: combo_sz == remaining → 直接返回
//
// MEMO_BITS=20 → 1048576 槽位 (~16MB), 与 Python C 扩展一致, 满足 Botzone 256MB 约束
// =============================================================================

#include "oracle.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <set>
#include <unordered_set>
#include <vector>

// ── CTZ (Count Trailing Zeros) ──────────────────────────────────────────────
// 用于快速遍历 bomb_mask 中置位的 bit
#if defined(__GNUC__) || defined(__clang__)
inline int ctz16(uint16_t x) { return __builtin_ctz(x); }
#elif defined(_MSC_VER)
#include <intrin.h>
inline int ctz16(uint16_t x) {
    unsigned long idx;
    _BitScanForward(&idx, (unsigned long)x);
    return (int)idx;
}
#else
inline int ctz16(uint16_t x) {
    int n = 0;
    while (!(x & 1)) { x >>= 1; n++; }
    return n;
}
#endif

// 连续牌型上限 (顺子/飞机最多到 A=11, 不含 2 和王)
// MAX_STRAIGHT_LEVEL 已在 oracle.h 中定义

// ── 内部类型 ────────────────────────────────────────────────────────────────

// DP 专用的组合元数据 (从 OracleCombo 预计算, 加速递归内层匹配)
// 与 oracle_core.c 中 ComboMeta 严格对齐
struct ComboMeta {
    short    size;        // 组合总牌数 (1~20, 如顺子5连=5张)
    CardCount counts;     // 各等级使用张数 (15-tuple, 与手牌直接按等级相减)
    uint16_t bitmask;     // 15-bit: bit i = 1 若 counts[i] > 0 (用于位掩码预过滤)
    uint16_t bomb_mask;   // 15-bit: bit i = 1 若 0 < counts[i] < 4 (可能拆弹的等级)
};

// DP 记忆化条目: 每个子问题状态 16 bytes
// 使用完美哈希 (hash_hand, 每等级 3 bits = 45 bits) + 线性探测
struct MemoEntry {
    uint64_t key;         // 手牌完美哈希值 (45 bits 有效信息, 高位为0)
    short    n_active;    // N_active 最优值 (该子状态的最少主动步数)
    short    n_preserve;  // N_preserve 最优值 (该子状态的炸弹保持最少步数)
};

// ── 全局常量 ────────────────────────────────────────────────────────────────

constexpr int MEMO_BITS    = 20;           // 1,048,576 slots, ~16MB (与 Python C 扩展一致, 彻底避免表满溢出)
constexpr int MEMO_SIZE    = 1 << MEMO_BITS;
constexpr uint64_t MEMO_MASK = MEMO_SIZE - 1;
constexpr int MAX_PROBES   = 64;           // 线性探测上限

static std::vector<MemoEntry> g_memo(MEMO_SIZE);

// ── OracleCombo→ ComboMeta 转换 ─────────────────────────────────────────────────

// OracleCombo → ComboMeta 转换 (预计算 size/bitmask/bomb_mask 以加速 DP 遍历)
static ComboMeta combo_to_meta(const OracleCombo& c) {
    ComboMeta m;
    short sz = 0;
    uint16_t mask = 0, bmask = 0;
    for (short i = 0; i < NUM_LEVELS; i++) {
        m.counts[i] = c.counts[i];
        sz += c.counts[i];
        if (c.counts[i] > 0) {
            mask |= (uint16_t)(1u << i);
            if (c.counts[i] < 4)
                bmask |= (uint16_t)(1u << i);
        }
    }
    m.size = sz;
    m.bitmask = mask;
    m.bomb_mask = bmask;
    return m;
}

// ── 组合生成助手函数 ──────────────────────────────────────────────────────

// 从等级列表构建 OracleCombo
static OracleCombo make_combo(OracleComboType type, Level main_lv,
                        const CardCount& cnts) {
    OracleCombo c;
    c.type = type;
    c.main_level = main_lv;
    for (short i = 0; i < NUM_LEVELS; i++) c.counts[i] = cnts[i];
    return c;
}

// 连续区间检测: 在 [0, MAX_STRAIGHT_LEVEL] 内找所有连续可用段
static std::vector<std::pair<Level, short>>
find_chain_segments(const CardCount& hand, short min_needed) {
    std::vector<std::pair<Level, short>> segs;
    short run_start = -1;
    short run_len = 0;
    for (Level lv = 0; lv <= MAX_STRAIGHT_LEVEL + 1; lv++) {
        bool has = (lv <= MAX_STRAIGHT_LEVEL && hand[lv] >= min_needed);
        if (has) {
            if (run_len == 0) run_start = lv;
            run_len++;
        } else {
            if (run_len >= 2) segs.push_back({run_start, run_len});
            run_len = 0;
        }
    }
    return segs;
}

// 用栈式迭代枚举所有带牌等级组合: 从候选人中选 needed 个
// 每个选中的等级消耗 kicker_use 张牌
// callback(used_counts) 对每个有效组合调用一次
template <typename F>
static void enumerate_kicker_combos(
    const CardCount& hand,
    short needed,
    short kicker_use,
    const std::set<Level>& exclude_set,
    F&& callback) {

    // 收集候选等级
    std::vector<Level> candidates;
    for (Level lv = 0; lv < NUM_LEVELS; lv++) {
        if (exclude_set.count(lv)) continue;
        if (hand[lv] >= kicker_use)
            candidates.push_back(lv);
    }
    if ((short)candidates.size() < needed) return;

    // 栈式组合枚举: combo[i] 是 candidates 的下标
    std::vector<short> combo(needed);
    for (short i = 0; i < needed; i++) combo[i] = i;
    short max_first = (short)candidates.size() - needed;

    while (true) {
        CardCount used = {};
        for (short i = 0; i < needed; i++) {
            Level klv = candidates[combo[i]];
            used[klv] = kicker_use;
        }
        callback(used);

        // 推进到下一组合
        short idx = needed - 1;
        while (idx >= 0 && combo[idx] == max_first + idx)
            idx--;
        if (idx < 0) break;
        combo[idx]++;
        for (short j = idx + 1; j < needed; j++)
            combo[j] = combo[j - 1] + 1;
    }
}

// ── 添加简单牌型 (单张/对子/三条/炸弹/火箭) ─────────────────────────────

static void add_simple_combos(const CardCount& hand,
                              std::vector<OracleCombo>& combos) {
    // 单张: 每个 ≥1 的等级
    for (Level lv = 0; lv < NUM_LEVELS; lv++) {
        if (hand[lv] >= 1) {
            CardCount c = {};
            c[lv] = 1;
            combos.push_back(make_combo(OracleComboType::SINGLE, lv, c));
        }
    }
    // 对子: 每个 ≥2 的等级
    for (Level lv = 0; lv <= MAX_REGULAR_LEVEL; lv++) {
        if (hand[lv] >= 2) {
            CardCount c = {};
            c[lv] = 2;
            combos.push_back(make_combo(OracleComboType::PAIR, lv, c));
        }
    }
    // 三条: 每个 ≥3 的普通等级
    for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++) {
        if (hand[lv] >= 3) {
            CardCount c = {};
            c[lv] = 3;
            combos.push_back(make_combo(OracleComboType::TRIPLE, lv, c));
        }
    }
    // 炸弹: 每个恰有 4 张的普通等级
    for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++) {
        if (hand[lv] == 4) {
            CardCount c = {};
            c[lv] = 4;
            combos.push_back(make_combo(OracleComboType::BOMB, lv, c));
        }
    }
    // 火箭: 同时持有大小王
    if (hand[LEVEL_JOKER_SMALL] >= 1 && hand[LEVEL_JOKER_BIG] >= 1) {
        CardCount c = {};
        c[LEVEL_JOKER_SMALL] = 1;
        c[LEVEL_JOKER_BIG]   = 1;
        combos.push_back(make_combo(OracleComboType::KING_BOMB, LEVEL_JOKER_BIG, c));
    }
}

// ── 添加连续牌型 (顺子/连对/飞机不带翼) ─────────────────────────────────

static void add_serial_combos(const CardCount& hand,
                               std::vector<OracleCombo>& combos) {
    // 单顺 (≥5 连, 每级需 ≥1)
    {
        auto segs = find_chain_segments(hand, 1);
        constexpr short MIN_LEN = 5;
        for (auto [seg_start, max_len] : segs) {
            for (short len = MIN_LEN; len <= max_len; len++) {
                for (short offset = 0; offset + len <= max_len; offset++) {
                    CardCount c = {};
                    Level start = seg_start + offset;
                    for (short i = 0; i < len; i++)
                        c[start + i] = 1;
                    combos.push_back(make_combo(
                        OracleComboType::SERIAL_SINGLE, start + len - 1, c));
                }
            }
        }
    }
    // 连对 (≥3 连对, 每级需 ≥2)
    {
        auto segs = find_chain_segments(hand, 2);
        constexpr short MIN_LEN = 3;
        for (auto [seg_start, max_len] : segs) {
            for (short len = MIN_LEN; len <= max_len; len++) {
                for (short offset = 0; offset + len <= max_len; offset++) {
                    CardCount c = {};
                    Level start = seg_start + offset;
                    for (short i = 0; i < len; i++)
                        c[start + i] = 2;
                    combos.push_back(make_combo(
                        OracleComboType::SERIAL_PAIR, start + len - 1, c));
                }
            }
        }
    }
    // 飞机不带翼 (≥2 连三, 每级需 ≥3)
    {
        auto segs = find_chain_segments(hand, 3);
        constexpr short MIN_LEN = 2;
        for (auto [seg_start, max_len] : segs) {
            for (short len = MIN_LEN; len <= max_len; len++) {
                for (short offset = 0; offset + len <= max_len; offset++) {
                    CardCount c = {};
                    Level start = seg_start + offset;
                    for (short i = 0; i < len; i++)
                        c[start + i] = 3;
                    combos.push_back(make_combo(
                        OracleComboType::SERIAL_TRIPLE, start + len - 1, c));
                }
            }
        }
    }
}

// ── 添加带牌组合 (三带一/二, 四带二, 飞机带单/对) ──────────────────────

static void add_kicker_combos(const CardCount& hand,
                               std::vector<OracleCombo>& combos) {

    // ── 三带一 (TYPE_6) ──
    for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++) {
        if (hand[lv] < 3) continue;
        for (Level klv = 0; klv < NUM_LEVELS; klv++) {
            if (klv == lv || hand[klv] < 1) continue;
            CardCount c = {};
            c[lv]  = 3;
            c[klv] = 1;
            combos.push_back(make_combo(OracleComboType::TRIPLE_1, lv, c));
        }
    }

    // ── 三带二 (TYPE_7) ──
    for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++) {
        if (hand[lv] < 3) continue;
        for (Level klv = 0; klv < NUM_LEVELS; klv++) {
            if (klv == lv || hand[klv] < 2) continue;
            CardCount c = {};
            c[lv]  = 3;
            c[klv] = 2;
            combos.push_back(make_combo(OracleComboType::TRIPLE_2, lv, c));
        }
    }

    // ── 四带两单 (TYPE_13) ──
    for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++) {
        if (hand[lv] < 4) continue;
        std::set<Level> exclude = {lv};
        enumerate_kicker_combos(hand, 2, 1, exclude,
            [&](const CardCount& kickers) {
                CardCount c = kickers;
                c[lv] = 4;
                combos.push_back(make_combo(OracleComboType::QUAD_2_SINGLE, lv, c));
            });
    }

    // ── 四带两对 (TYPE_14) ──
    for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++) {
        if (hand[lv] < 4) continue;
        std::set<Level> exclude = {lv};
        enumerate_kicker_combos(hand, 2, 2, exclude,
            [&](const CardCount& kickers) {
                CardCount c = kickers;
                c[lv] = 4;
                combos.push_back(make_combo(OracleComboType::QUAD_2_PAIR, lv, c));
            });
    }

    // ── 飞机带单 (TYPE_11) ──
    {
        auto segs = find_chain_segments(hand, 3);
        constexpr short MIN_LEN = 2;
        for (auto [seg_start, max_len] : segs) {
            for (short chain_len = MIN_LEN; chain_len <= max_len; chain_len++) {
                for (short offset = 0; offset + chain_len <= max_len; offset++) {
                    Level start = seg_start + offset;
                    std::set<Level> main_set;
                    for (short i = 0; i < chain_len; i++)
                        main_set.insert(start + i);

                    enumerate_kicker_combos(hand, chain_len, 1, main_set,
                        [&](const CardCount& kickers) {
                            CardCount c = kickers;
                            for (Level m : main_set) c[m] = 3;
                            combos.push_back(make_combo(
                                OracleComboType::SERIAL_3_1,
                                start + chain_len - 1, c));
                        });
                }
            }
        }
    }

    // ── 飞机带对 (TYPE_12) ──
    {
        auto segs = find_chain_segments(hand, 3);
        constexpr short MIN_LEN = 2;
        for (auto [seg_start, max_len] : segs) {
            for (short chain_len = MIN_LEN; chain_len <= max_len; chain_len++) {
                for (short offset = 0; offset + chain_len <= max_len; offset++) {
                    Level start = seg_start + offset;
                    std::set<Level> main_set;
                    for (short i = 0; i < chain_len; i++)
                        main_set.insert(start + i);

                    enumerate_kicker_combos(hand, chain_len, 2, main_set,
                        [&](const CardCount& kickers) {
                            CardCount c = kickers;
                            for (Level m : main_set) c[m] = 3;
                            combos.push_back(make_combo(
                                OracleComboType::SERIAL_3_2,
                                start + chain_len - 1, c));
                        });
                }
            }
        }
    }
}

// ── 主组合生成入口 ─────────────────────────────────────────────────────────
// 从手牌计数向量生成全部 14 种标准牌型组合
// 组合总数上界约 1000-3000 (取决于手牌结构)
// 包含简单牌型(单张/对子/三条/炸弹/火箭) + 连续牌型(顺子/连对/飞机)
//      + 带牌组合(三带一/二, 四带二, 飞机带单/对)

std::vector<OracleCombo> generate_all_combos(const CardCount& hand) {
    std::vector<OracleCombo> combos;
    add_simple_combos(hand, combos);
    add_serial_combos(hand, combos);
    add_kicker_combos(hand, combos);
    return combos;
}

// ── 贪心初始上界 (构造一个完整解, 用于 DP 剪枝初始化) ────────────────────
// 与 oracle_core.c greedy_bound 对齐: 每次贪心选最大可用 combo,
// 同时计算 N_active (步数) 和 N_preserve (含拆弹惩罚) 两个上界.
// 炸弹惩罚在修改手牌副本前计算 (与 C 参考一致).

static std::pair<short, short> greedy_bound(const CardCount& hand,
                                             const std::vector<ComboMeta>& metas) {
    CardCount cur = hand;
    short steps = 0;
    short steps_p = 0;
    while (true) {
        short remaining = 0;
        uint16_t hmask = 0;
        for (short i = 0; i < NUM_LEVELS; i++) {
            remaining += cur[i];
            if (cur[i]) hmask |= (uint16_t)(1u << i);
        }
        if (remaining == 0) break;

        // 找能用的最大 combo
        short best_sz = 0;
        short best_idx = -1;
        for (size_t ci = 0; ci < metas.size(); ci++) {
            auto& m = metas[ci];
            if (m.size > remaining) continue;
            if ((m.bitmask & hmask) != m.bitmask) continue;

            bool ok = true;
            for (short i = 0; i < NUM_LEVELS; i++) {
                if (m.counts[i] > cur[i]) { ok = false; break; }
            }
            if (!ok) continue;

            if (m.size > best_sz) {
                best_sz = m.size;
                best_idx = (short)ci;
            }
        }

        if (best_idx < 0) break;  // 不应发生 (至少可出单牌)

        auto& m = metas[best_idx];
        // 炸弹惩罚在修改手牌前计算 (与 oracle_core.c 一致)
        short penalty = 0;
        uint16_t bm = m.bomb_mask;
        while (bm) {
            short r = ctz16(bm);
            bm &= bm - 1;
            if (cur[r] == 4) penalty += 1;
        }
        for (short i = 0; i < NUM_LEVELS; i++)
            cur[i] -= m.counts[i];
        steps++;
        steps_p += 1 + penalty;
    }
    return {steps, steps_p};
}

// ── SplitMix64 哈希终混器 (与 oracle_core.c 一致) ──────────────────────────
// 将 64 位手牌哈希值进一步混淆以改善 memo 槽位分布
static inline uint32_t key_to_slot(uint64_t k) {
    k = (k ^ (k >> 30)) * 0xbf58476d1ce4e5b9ULL;
    k = (k ^ (k >> 27)) * 0x94d049bb133111ebULL;
    k = k ^ (k >> 31);
    return (uint32_t)(k & MEMO_MASK);
}

// ── 合并 hash+hmask 单次遍历 (优化 #10, 与 oracle_core.c 一致) ─────────────

static inline uint64_t hash_and_mask(const CardCount& hand, uint16_t* hmask_out) {
    uint64_t h = 0;
    uint16_t m = 0;
    for (short i = 0; i < NUM_LEVELS; i++) {
        short c = hand[i];
        h |= (uint64_t)(c & 0x7) << (i * 3);
        if (c) m |= (uint16_t)(1u << i);
    }
    *hmask_out = m;
    return h;
}

// ═══════════════════════════════════════════════════════════════════════════
// 统一 DP: 单次递归同时计算 N_active 和 N_preserve
// ═══════════════════════════════════════════════════════════════════════════
// 与 oracle_core.c dp_unified_internal 严格对齐:
//   - N_active:   bomb_penalty=0, 不惩罚拆弹
//   - N_preserve: bomb_penalty=1, 每拆一个4张等级额外+1步
//   - 双输出共享同一递归树, 避免重复搜索
// 关键优化:
//   - 位掩码预过滤 (O(1) 淘汰不可用的 combo)
//   - 下界剪枝 (best ≤ ceil(remaining/max_size) 时直接 break)
//   - 单 combo 快速路径 (combo_sz == remaining 时无需递归)
//   - 合并 hash+hmask 单次遍历 15 ranks (优化 #10)

constexpr short INF_SHORT = 32767;

// 子状态不传播父节点上界 (传递 INF_SHORT), 与 oracle_core.c 传递 FLOAT_INF 一致
// 避免子节点因宽松上界过早剪枝导致返回非最优值
static std::pair<short, short> dp_unified_internal(
    const CardCount& hand,
    const std::vector<ComboMeta>& metas,
    const std::array<short, 21>& first_ci_for_rem,
    short init_a,
    short init_p) {

    // ── 合并 hash + hmask (单次遍历 15 ranks) ──
    uint16_t hmask = 0;
    uint64_t hkey = hash_and_mask(hand, &hmask);

    short remaining = 0;
    for (short i = 0; i < NUM_LEVELS; i++) remaining += hand[i];

    // ── Base case: 空手牌 ──
    if (remaining == 0)
        return {0, 0};

    // ── Memo lookup ──
    uint64_t slot = key_to_slot(hkey);
    for (int probe = 0; probe < MAX_PROBES; probe++) {
        if (g_memo[slot].key == hkey)
            return {g_memo[slot].n_active, g_memo[slot].n_preserve};
        if (g_memo[slot].key == 0) break;
        slot = (slot + 1) & MEMO_MASK;
    }

    short best_a = init_a;
    short best_p = init_p;

    short max_sz = metas[0].size;
    short lb = (remaining + max_sz - 1) / max_sz;

    // ── 遍历 combo ──
    short start_ci = first_ci_for_rem[remaining];
    for (size_t ci = start_ci; ci < metas.size(); ci++) {
        auto& m = metas[ci];

        // 优化 #4: 下界剪枝
        if (lb >= best_a && lb >= best_p)
            break;

        // 优化 #2: 位掩码预过滤
        if ((m.bitmask & hmask) != m.bitmask)
            continue;

        // 检查手牌是否够用
        bool ok = true;
        for (short i = 0; i < NUM_LEVELS; i++) {
            if (m.counts[i] > hand[i]) { ok = false; break; }
        }
        if (!ok) continue;

        // 优化 #11: 单 combo 快速路径
        if (m.size == remaining) {
            if (1 < best_a) best_a = 1;

            short penalty = 0;
            uint16_t bm = m.bomb_mask;
            while (bm) {
                short r = ctz16(bm);
                bm &= bm - 1;
                if (hand[r] == 4) penalty += 1;
            }
            short val_p = 1 + penalty;
            if (val_p < best_p) best_p = val_p;
            continue;
        }

        // 剩余手牌
        CardCount sub = hand;
        for (short i = 0; i < NUM_LEVELS; i++)
            sub[i] -= m.counts[i];

        // 递归 — 不传播父节点上界, 避免子节点过早剪枝返回不正确值
        // (与 oracle_core.c 传递 FLOAT_INF 一致)
        auto [sub_a, sub_p] = dp_unified_internal(sub, metas, first_ci_for_rem,
                                                   INF_SHORT, INF_SHORT);

        // N_active
        short val_a = 1 + sub_a;
        if (val_a < best_a) best_a = val_a;

        // N_preserve: 计算拆弹惩罚 (α=1.0), 在修改手牌前计算
        short penalty = 0;
        uint16_t bm = m.bomb_mask;
        while (bm) {
            short r = ctz16(bm);
            bm &= bm - 1;
            if (hand[r] == 4) penalty += 1;
        }
        short val_p = 1 + sub_p + penalty;
        if (val_p < best_p) best_p = val_p;
    }

    // ── Memo store (线性探测) ──
    slot = key_to_slot(hkey);
    for (int probe = 0; probe < MAX_PROBES; probe++) {
        if (g_memo[slot].key == 0 || g_memo[slot].key == hkey) {
            g_memo[slot] = {hkey, best_a, best_p};
            break;
        }
        slot = (slot + 1) & MEMO_MASK;
    }

    return {best_a, best_p};
}

// ── 统一 DP 包装 (预计算元数据 + 贪心上界 + 递归入口) ─────────────────────

static std::pair<short, short> dp_unified_impl(
    const CardCount& hand,
    const std::vector<OracleCombo>& combos) {

    if (combos.empty())
        return {0, 0};

    // 将 OracleCombo列表转为 ComboMeta 列表
    std::vector<ComboMeta> metas;
    metas.reserve(combos.size());
    for (auto& c : combos)
        metas.push_back(combo_to_meta(c));

    // 优化 #3: 按 size 降序排列
    std::sort(metas.begin(), metas.end(),
        [](const ComboMeta& a, const ComboMeta& b) {
            return a.size > b.size;
        });

    // 优化 #9: 构建 first_ci_for_rem 查找表
    // first_ci_for_rem[r] = 第一个 size <= r 的 combo 下标。
    // metas 按 size 降序，因此必须让 r 也按降序扫描、ci 单调递增
    // （与 Python oracle.py::_precompute_combo_data 一致）；若 r 升序，
    // ci 在 r=0 时已推到 n，表项全为 n，DP 的 combo 循环将一次都不执行。
    std::array<short, 21> first_ci_for_rem;
    first_ci_for_rem.fill((short)metas.size());
    {
        size_t ci = 0;
        for (short r = 20; r >= 0; r--) {
            while (ci < metas.size() && metas[ci].size > r)
                ci++;
            first_ci_for_rem[r] = (short)ci;
        }
    }

    // 优化 #7: 贪心初始上界 (与 oracle_core.c 一致 — 构造完整解得到紧界)
    auto [greedy_a, greedy_p] = greedy_bound(hand, metas);

    return dp_unified_internal(hand, metas, first_ci_for_rem,
                                greedy_a, greedy_p);
}

// ── 公开 API: dp_min_steps ─────────────────────────────────────────────────

short dp_min_steps(const CardCount& hand,
                   const std::vector<OracleCombo>& combos,
                   float bomb_penalty) {
    auto [n_active, n_preserve] = dp_unified_impl(hand, combos);
    return (bomb_penalty == 0.0f) ? n_active : n_preserve;
}

// ── 缓存管理 ───────────────────────────────────────────────────────────────

void dp_clear_cache() {
    std::fill(g_memo.begin(), g_memo.end(), MemoEntry{0, 0, 0});
}

// ═══════════════════════════════════════════════════════════════════════════
// 被动步数启发式
// ═══════════════════════════════════════════════════════════════════════════
// 与 DMC (Deep Monte Carlo) 保持一致: 统计非炸弹牌型中同一类型-主等级的唯一种数,
// 取各类的最大值作为对手在"始终无控牌权"情况下需要的回合数估计.
// 这是推理端简化版 (不依赖对手实际手牌), 用于近似 N_passive.

short dp_passive_steps_heuristic(const CardCount& hand,
                                  const std::vector<OracleCombo>& combos) {
    // 按牌型统计唯一 (type, main_level) 对的数量
    // 对每一种牌型, 统计不同主等级的个数 → 取最大值
    std::set<Level> levels_by_type[18];  // 18 combo types
    short bomb_count = 0;

    for (auto& c : combos) {
        if (c.type == OracleComboType::BOMB || c.type == OracleComboType::KING_BOMB) {
            bomb_count++;
            continue;
        }
        if (c.type == OracleComboType::PASS || c.type == OracleComboType::INVALID)
            continue;

        short tidx = (short)c.type;
        if (tidx >= 0 && tidx < 18)
            levels_by_type[tidx].insert(c.main_level);
    }

    short max_chain = 0;
    for (short t = 0; t < 18; t++) {
        max_chain = std::max(max_chain, (short)levels_by_type[t].size());
    }

    return std::max((short)1, (short)(max_chain - bomb_count));
}

// ═══════════════════════════════════════════════════════════════════════════
// 完整神谕原始值计算 (核心入口)
// ═══════════════════════════════════════════════════════════════════════════
// 从手牌计数向量计算 N_active / N_passive / delta_N / N_preserve
// self_adv 由调用方按角色补充 (main.cpp 中减去 estimate_opponent_n_active)
// 流程: 手牌 → generate_all_combos → dp_unified_impl(N_a+N_p) → dp_passive → ΔN

OracleRaw compute_oracle_raw(const CardCount& hand_counts) {
    OracleRaw raw = {};

    // 检查空手牌
    short total = 0;
    for (short i = 0; i < NUM_LEVELS; i++) total += hand_counts[i];
    if (total == 0) return raw;

    // 生成所有合法组合
    std::vector<OracleCombo> combos = generate_all_combos(hand_counts);

    // 统一 DP: 同时计算 N_active 和 N_preserve
    auto [n_a, n_p] = dp_unified_impl(hand_counts, combos);
    raw.n_active   = n_a;
    raw.n_preserve = n_p;

    // N_passive: 始终使用启发式 (DMC 兼容, 不依赖对手信息)
    raw.n_passive = dp_passive_steps_heuristic(hand_counts, combos);

    // ΔN: 控牌敏感度
    raw.delta_n = std::max((short)0, (short)(raw.n_passive - raw.n_active));

    // self_adv: 由调用方按角色补充, 此处初始化为 0
    raw.self_adv = 0.0f;

    return raw;
}

// ═══════════════════════════════════════════════════════════════════════════
// 辅助函数: Card ID → CardCount / 未知牌池 / 对手估计
// ═══════════════════════════════════════════════════════════════════════════

CardCount build_hand_counts(const std::vector<short>& cards) {
    CardCount cnt = {};
    for (short c : cards)
        cnt[card2level(c)]++;
    return cnt;
}

// ── 对手主动步数估计 (简化启发式) ────────────────────────────────────────
// 假设对手每步平均出 2.8 张牌, 用剩余牌数线性估算 N̂_opp
// 注: unplayed_pool 参数保留用于未来可能的更精确估计, 当前未使用

float estimate_opponent_n_active(short opponent_remaining,
                                  const CardCount& /*unplayed_pool*/) {
    if (opponent_remaining <= 0) return 0.0f;
    const float AVG_CARDS_PER_STEP = 2.8f;
    return (float)opponent_remaining / AVG_CARDS_PER_STEP;
}
