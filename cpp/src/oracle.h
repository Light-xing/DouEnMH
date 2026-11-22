// =============================================================================
// DouEnMH 递推神谕函数 (Oracle Functions) — Botzone C++ 部署
//
// 实现 design/oracle_functions_design.md 中定义的神谕特征计算：
//   N_active   — 主动步数 (始终握有控牌权时的最少清空步数)
//   N_passive  — 被动步数 (始终无控牌权时的最少清空步数)
//   delta_N    — 控牌敏感度 = N_passive - N_active
//   N_preserve — 炸弹保持度 (带拆弹惩罚的最少步数)
//   self_adv   — 优势距离 = N_active(self) - N̂_opp
//
// 所有特征使用与训练一致的饱和归一化，值域严格在 [0,1) 或 (-1,1) 内。
// 仅依赖自手牌信息，无需完美信息，训练与推理输入一致。
// =============================================================================

#pragma once

#include "card_defs.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

// ── Oracle 专用常量 ──────────────────────────────────────────────────────

constexpr Level MAX_REGULAR_LEVEL = 12;  // 2 (普通牌上限, 王的下一级)

// 组合类型 (与 Python move_detector.py 一致)
enum class OracleComboType : short {
    PASS            = 0,
    SINGLE          = 1,
    PAIR            = 2,
    TRIPLE          = 3,
    BOMB            = 4,
    KING_BOMB       = 5,  // 火箭 (大小王)
    TRIPLE_1        = 6,  // 三带一
    TRIPLE_2        = 7,  // 三带二
    SERIAL_SINGLE   = 8,  // 顺子 (≥5 连)
    SERIAL_PAIR     = 9,  // 连对 (≥3 连对)
    SERIAL_TRIPLE   = 10, // 飞机 (≥2 连三条)
    SERIAL_3_1      = 11, // 飞机带单
    SERIAL_3_2      = 12, // 飞机带对
    QUAD_2_SINGLE   = 13, // 四带两单
    QUAD_2_PAIR     = 14, // 四带两对
    INVALID         = 15,
};

// 单个出牌组合: 计数向量 + 类型 + 主等级
struct OracleCombo {
    CardCount counts;   // 该组合使用的各等级张数
    OracleComboType type;
    Level     main_level; // 主牌等级 (用于大小比较)

    bool is_bomb_or_rocket() const {
        return type == OracleComboType::BOMB || type == OracleComboType::KING_BOMB;
    }
};

// ── 神谕特征输出 ─────────────────────────────────────────────────────────

// 原始神谕值 (未归一化)
struct OracleRaw {
    short n_active = 0;    // 主动步数 [0, 20]
    short n_passive = 0;   // 被动步数 [0, 40]
    short delta_n = 0;     // 控牌敏感度 [0, 20]
    short n_preserve = 0;  // 炸弹保持步数 [0, 25]
    float self_adv = 0.0f; // 优势距离 [-20, 20] (可为负)
};

// 归一化神谕特征 (作为 X 向量输入)
struct OracleFeat {
    float n_active = 0.0f;    // N_a / (N_a + 10),        值域 [0, 0.67)
    float n_passive = 0.0f;   // N_p / (N_p + 15),        值域 [0, 0.73)
    float delta_n = 0.0f;     // ΔN / (ΔN + 8),           值域 [0, 0.71)
    float n_preserve = 0.0f;  // N_pres / (N_pres + 10),  值域 [0, ~0.71)
    float self_adv = 0.0f;    // tanh(self_adv / 5),       值域 (-1, 1)

    // 返回 5 维 float 向量，可直接拼接到 X 向量末尾
    std::vector<float> to_vector() const {
        return {n_active, n_passive, delta_n, n_preserve, self_adv};
    }
};

// ── 组合生成 (MovesGener C++ 移植) ──────────────────────────────────────

// 从手牌计数向量生成所有合法出牌组合 (14 种牌型, 不含航天飞机)
// 包含: 单张/对子/三条/炸弹/火箭 + 三带一/三带二 + 四带两单/四带两对
//       + 顺子/连对/飞机/飞机带单/飞机带对
// 注意: 与 move_generator.h 不同, 神谕 DP 仅处理标准 14 种牌型,
//       SSHUTTLE (航天飞机) 类型 15-17 不在 DP 搜索空间内
std::vector<OracleCombo> generate_all_combos(const CardCount& hand);

// ── 递推 DP ─────────────────────────────────────────────────────────────

// 主动步数 / 炸弹保持度统一 DP
//   bomb_penalty = 0.0 → N_active
//   bomb_penalty = 1.0 → N_preserve
short dp_min_steps(const CardCount& hand,
                   const std::vector<OracleCombo>& combos,
                   float bomb_penalty = 0.0f);

// 清空 DP 内部哈希表缓存 (每个新 Episode 开始时调用)
void dp_clear_cache();

// 被动步数 (简化启发式 — 推理用，不依赖对手手牌)
short dp_passive_steps_heuristic(const CardCount& hand,
                                  const std::vector<OracleCombo>& combos);

// ── 神谕特征计算 ────────────────────────────────────────────────────────

// 从手牌计算完整神谕特征 (仅依赖自手牌)
//   hand_counts: 己方当前手牌，按等级的计数向量
//   opponent_cards_remaining: 两位对手的剩余牌数 [opp1, opp2]
//   unplayed_pool: 未知牌池 (总牌 - 己方手牌 - 已出牌) 按等级的计数向量
OracleRaw compute_oracle_raw(const CardCount& hand_counts);

// 将原始值转为归一化特征
inline OracleFeat normalize_oracle(const OracleRaw& raw) {
    OracleFeat feat;
    feat.n_active   = (float)raw.n_active  / ((float)raw.n_active  + 10.0f);
    feat.n_passive  = (float)raw.n_passive / ((float)raw.n_passive + 15.0f);
    feat.delta_n    = (float)raw.delta_n   / ((float)raw.delta_n   + 8.0f);
    feat.n_preserve = (float)raw.n_preserve / ((float)raw.n_preserve + 10.0f);
    feat.self_adv   = std::tanh(raw.self_adv / 5.0f);
    return feat;
}

// 便捷接口: 手牌 → 归一化神谕特征（一步到位）
inline OracleFeat compute_oracle_features(const CardCount& hand_counts) {
    return normalize_oracle(compute_oracle_raw(hand_counts));
}

// ── 辅助函数 ────────────────────────────────────────────────────────────

// 从 card ID 列表构建手牌计数向量 (使用 card_defs.h 中的 card2level)
CardCount build_hand_counts(const std::vector<short>& cards);

// 注: 未知牌池构建由 opponent_modeling.h 的 buildUnknownPool() 统一提供,
//     此处不再重复定义, 避免两套实现偏离

// 手牌计数向量 → 64 位哈希 (用于 DP memoization)
inline uint64_t hash_hand(const CardCount& hand) {
    uint64_t h = 0;
    for (short i = 0; i < NUM_LEVELS; ++i)
        h |= (uint64_t)(hand[i] & 0x7) << (i * 3);
    return h;
}

// 估计对手的主动步数 (简化启发式)
//   opponent_remaining: 对手剩余牌数
//   unplayed_pool: 未知牌池计数
float estimate_opponent_n_active(short opponent_remaining,
                                  const CardCount& unplayed_pool);
