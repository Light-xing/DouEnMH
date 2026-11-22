// =============================================================================
// opponent_modeling.h — DouEnMH 对手建模模块 (C3/C4/C5, 共 22 维)
// =============================================================================
// 与 Python opponent_modeling.py 严格对齐.
// 设计依据: opponent_modeling_design.md
//
// C3 | OHS + OBP  | 4 维 | 对手手牌强度 + 炸弹概率
//     OHS (Opponent Hand Strength):  加权 rank 价值 × (对手牌数/池牌数)
//     OBP (Opponent Bomb Probability): 超几何概率查表 P(对手有炸弹|当前未知牌池)
//     单次构建未知牌池, 复用给 4 个子计算 → 零重复开销
//
// C4 | CTP         | 14 维 | 对手牌型偏好 (Combo-Type Preference)
//     7 类牌型 × 2 对手 = 14 维, EMA γ=0.9, 仅统计"主动领出"动作
//     回放出牌历史 ≤~100 手, O(H) 开销 < 0.05ms
//
// C5 | SWI + TP    | 4 维 | 对手行为风格
//     SWI (Suppression Willingness Index): EMA(实际压制) / EMA(p_can 期望)
//     TP  (Turn Preference):             EMA(首手出牌数 / 6), 值域 [0,1]
//     复用 C3 的未知牌池和 OBP → 零额外查表开销
//
// 总计: 4 + 14 + 4 = 22 维对手建模特征, 全部仅依赖当前已知信息
// =============================================================================

#ifndef OPPONENT_MODELING_H
#define OPPONENT_MODELING_H

#include "card_encoding.h"
#include "combo.h"
#include "game_state.h"
#include <array>
#include <cmath>
#include <vector>
#include <algorithm>

// ═══════════════════════════════════════════════════════════════════════════
// 基础工具
// ═══════════════════════════════════════════════════════════════════════════

// Botzone Card ID → 等级索引 0-14 (3→0, 4→1, …, 2→12, 小王→13, 大王→14)
inline int botzoneCardToRank(Card card) {
    if (card == CARD_JOKER_SMALL) return 13;
    if (card == CARD_JOKER_BIG)   return 14;
    return card / 4;
}

// 牌列表 → 15 维等级计数向量
inline std::array<int, 15> cardsToRankCounts(const std::vector<int>& cards) {
    std::array<int, 15> cnt = {0};
    for (int c : cards) {
        int r = botzoneCardToRank(c);
        if (r >= 0 && r < 15) cnt[r]++;
    }
    return cnt;
}

// 组合数 C(n, k) — double 精度
inline double comb_double(int n, int k) {
    if (k < 0 || k > n) return 0.0;
    if (k > n - k) k = n - k;
    double result = 1.0;
    for (int i = 1; i <= k; ++i)
        result *= (double)(n - k + i) / i;
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// 预计算超几何概率表 — 模块首次使用时初始化 (~12KB)
// ═══════════════════════════════════════════════════════════════════════════

constexpr int OM_MAX_POOL = 37;  // 未知牌池最大 N = 54 - 17
constexpr int OM_MAX_REM  = 20;  // 对手最多 20 张 (地主)

struct HypergeomTables {
    float p_bomb_single[OM_MAX_POOL + 1][OM_MAX_REM + 1] = {};  // P(特定 rank 恰好 4 张)
    float p_rocket[OM_MAX_POOL + 1][OM_MAX_REM + 1] = {};       // P(同时持有大小王)
    bool initialized = false;

    void init() {
        if (initialized) return;
        // P_bomb_single[N][c] = C(N-4, c-4) / C(N, c)
        for (int N = 4; N <= OM_MAX_POOL; N++) {
            for (int c = 4; c <= std::min(N, OM_MAX_REM); c++) {
                p_bomb_single[N][c] = (float)(
                    comb_double(N - 4, c - 4) / comb_double(N, c)
                );
            }
        }
        // P_rocket[N][c] = C(N-2, c-2) / C(N, c)
        for (int N = 2; N <= OM_MAX_POOL; N++) {
            for (int c = 2; c <= std::min(N, OM_MAX_REM); c++) {
                p_rocket[N][c] = (float)(
                    comb_double(N - 2, c - 2) / comb_double(N, c)
                );
            }
        }
        initialized = true;
    }
};

// 全局单例 (在 compute_opponent_features 首次调用时初始化)
inline HypergeomTables& getHypergeomTables() {
    static HypergeomTables tables;
    tables.init();
    return tables;
}

// ═══════════════════════════════════════════════════════════════════════════
// C3: 对手手牌强度 (OHS) + 对手炸弹概率 (OBP) — 4 维
// ═══════════════════════════════════════════════════════════════════════════

// Rank 价值权重: w[r] 对应等级 3,4,5,6,7,8,9,10,J,Q,K,A,2,小王,大王
constexpr float RANK_WEIGHTS[15] = {
    0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 6, 8, 10, 12
};

constexpr float OHS_MAX = 100.0f;

struct OpponentFeat {
    float ohs_up   = 0.0f;
    float ohs_down = 0.0f;
    float obp_up   = 0.0f;
    float obp_down = 0.0f;

    std::vector<float> to_vector() const {
        return {ohs_up, ohs_down, obp_up, obp_down};
    }
};

// 构建未知牌池: pool[r] = 该等级在未知牌池中的张数
// 未知牌池 = 完整牌库 − 己方手牌 − 所有已出牌 − 三张底牌
inline void buildUnknownPool(const GameState& state,
                             std::array<int, 15>& poolVec, int& poolTotal) {
    // 完整牌库: 普通 rank (0-12) 各 4 张, 大小王各 1 张
    for (int i = 0; i < 13; i++) poolVec[i] = 4;
    poolVec[13] = 1;
    poolVec[14] = 1;

    // 减去己方手牌
    for (int c : state.myHandCards) {
        int r = botzoneCardToRank(c);
        if (r >= 0 && r < 15) poolVec[r]--;
    }

    // 减去所有已出牌
    auto subPlayed = [&](const std::vector<int>& played) {
        for (int c : played) {
            int r = botzoneCardToRank(c);
            if (r >= 0 && r < 15) poolVec[r]--;
        }
    };
    subPlayed(state.landlordPlayedCards);
    subPlayed(state.landlordUpPlayedCards);
    subPlayed(state.landlordDownPlayedCards);

    // 减去底牌 —— **仅非地主视角**，且只扣"尚未被打出"的底牌
    //
    // ① 底牌是地主手牌的一部分（设计文档 §1.2 注），地主视角下 myHandCards 已包含它们，
    //    再减一次会让牌池少 1~3 张；且末尾有负数截断，偏差随局面变化（不是固定偏移）。
    // ② 地主打出某张底牌后，该牌已经在"所有已出牌"里扣过一次；若仍按完整 3 张底牌再扣，
    //    农民视角会重复扣减。训练端 infoset.three_landlord_cards 会随地主打出而移除
    //    （game.py），等价于"只扣未打出的底牌"。
    // 与训练端 opponent_modeling._build_unknown_pool(is_landlord=..., 剩余底牌) 保持一致。
    if (state.myPosition != state.landlordPosition) {
        for (int c : state.threeLandlordCards) {
            bool alreadyPlayed = false;
            for (int p : state.landlordPlayedCards) {
                if (p == c) { alreadyPlayed = true; break; }
            }
            if (alreadyPlayed) continue;
            int r = botzoneCardToRank(c);
            if (r >= 0 && r < 15) poolVec[r]--;
        }
    }

    poolTotal = 0;
    for (int i = 0; i < 15; i++) {
        if (poolVec[i] < 0) poolVec[i] = 0;  // 防御
        poolTotal += poolVec[i];
    }
}

// 从未知牌池计算 OHS
inline float computeOHSFromPool(const std::array<int, 15>& poolVec,
                                 int poolTotal, int opponentRemaining) {
    if (opponentRemaining <= 0 || poolTotal <= 0) return 0.0f;
    float weightedSum = 0.0f;
    for (int r = 0; r < 15; r++)
        weightedSum += RANK_WEIGHTS[r] * (float)poolVec[r];
    return (float)opponentRemaining / (float)poolTotal * weightedSum;
}

// 从未知牌池计算 OBP — 纯查表, O(15)
inline float computeOBPFromPool(const std::array<int, 15>& poolVec,
                                 int poolTotal, int opponentRemaining) {
    int c_o = opponentRemaining;
    if (c_o <= 3 || poolTotal < c_o) return 0.0f;

    int N = poolTotal;
    auto& tbl = getHypergeomTables();

    // 火箭概率
    float p_rocket = 0.0f;
    if (poolVec[13] >= 1 && poolVec[14] >= 1
        && N <= OM_MAX_POOL && c_o <= OM_MAX_REM) {
        p_rocket = tbl.p_rocket[N][c_o];
    }

    // 普通炸弹: 遍历所有 n_r = 4 的 rank
    float p_no_bomb = 1.0f;
    float p_bomb_single = (N <= OM_MAX_POOL && c_o <= OM_MAX_REM)
        ? tbl.p_bomb_single[N][c_o] : 0.0f;

    for (int r = 0; r < 13; r++) {
        if (poolVec[r] == 4) {
            p_no_bomb *= (1.0f - p_bomb_single);
        }
    }

    return 1.0f - (1.0f - p_rocket) * p_no_bomb;
}

// C3 主接口: 计算对手信念特征 (4 维: ohs_up, ohs_down, obp_up, obp_down)
// upPos / downPos: 上手/下手绝对位置 (0/1/2)
// outPoolVec / outPoolTotal: 可选输出, 将未知牌池暴露给 C5 复用, 避免重复构建
// 流程: buildUnknownPool (单次) → OHS (加权 rank 分) + OBP (超几何查表) × 2 对手
inline OpponentFeat computeOpponentFeatures(const GameState& state,
                                             int upPos, int downPos,
                                             std::array<int, 15>* outPoolVec = nullptr,
                                             int* outPoolTotal = nullptr) {
    // 获取各玩家剩余牌数
    int ldPos = state.landlordPosition;
    auto getRemaining = [&](int pos) {
        if (pos == ldPos) return state.landlordCardsLeft;
        if (pos == (ldPos + 1) % 3) return state.landlordDownCardsLeft;
        return state.landlordUpCardsLeft;
    };

    int remUp = getRemaining(upPos);
    int remDown = getRemaining(downPos);

    // 单次构建未知牌池, 复用给 4 个子计算
    std::array<int, 15> poolVec;
    int poolTotal = 0;
    buildUnknownPool(state, poolVec, poolTotal);

    if (outPoolVec)  *outPoolVec  = poolVec;
    if (outPoolTotal) *outPoolTotal = poolTotal;

    OpponentFeat feat;
    feat.ohs_up   = std::min(computeOHSFromPool(poolVec, poolTotal, remUp) / OHS_MAX, 1.0f);
    feat.ohs_down = std::min(computeOHSFromPool(poolVec, poolTotal, remDown) / OHS_MAX, 1.0f);
    feat.obp_up   = computeOBPFromPool(poolVec, poolTotal, remUp);
    feat.obp_down = computeOBPFromPool(poolVec, poolTotal, remDown);
    return feat;
}

// ═══════════════════════════════════════════════════════════════════════════
// C4: 对手牌型偏好 CTP (Combo-Type Preference) — 14 维
// ═══════════════════════════════════════════════════════════════════════════

constexpr int NUM_CTP_TYPES = 7;
constexpr float CTP_GAMMA = 0.9f;

// ComboType → 7 类 CTP 映射
inline int classifyCtpCategory(ComboType t) {
    switch (t) {
        case ComboType::SINGLE:                               return 0;
        case ComboType::PAIR:                                 return 1;
        case ComboType::TRIPLE:
        case ComboType::TRIPLE_1:
        case ComboType::TRIPLE_2:                             return 2;
        case ComboType::SERIAL_SINGLE:
        case ComboType::SERIAL_PAIR:
        case ComboType::SERIAL_TRIPLE:
        case ComboType::SERIAL_3_1:
        case ComboType::SERIAL_3_2:                           return 3;
        case ComboType::QUAD_2_SINGLE:
        case ComboType::QUAD_2_PAIR:                          return 4;
        case ComboType::BOMB:                                 return 5;
        case ComboType::KING_BOMB:                            return 6;
        default:                                              return -1;
    }
}

// 判断 action 是否为 "主动领出" (与 Python is_lead_action 对齐)
// cardPlayActionSeq: 不含当前 action 的历史
inline bool isLeadAction(const std::vector<int>& action,
                         const std::vector<std::pair<int, std::vector<int>>>& history) {
    if (action.empty()) return false;
    if (history.empty()) return true;

    // rival_move = 最后一个非 pass 动作 (跳过末尾连续的 pass)
    std::vector<int> rivalMove;
    const auto& last = history.back().second;
    if (last.empty()) {
        // 最后一个动作是 pass
        if (history.size() >= 2) {
            rivalMove = history[history.size() - 2].second;
        } else {
            rivalMove = {};  // 仅一手历史且为 pass — 新回合
        }
    } else {
        rivalMove = last;
    }
    return rivalMove.empty();
}

// 回放历史计算 CTP 状态 (3 位置 × 7 类别), EMA γ=0.9
inline void computeCtpStateFromHistory(
    const std::vector<std::pair<int, std::vector<int>>>& history,
    float ctpState[3][7])
{
    // 初始化为全零
    for (int p = 0; p < 3; p++)
        for (int t = 0; t < NUM_CTP_TYPES; t++)
            ctpState[p][t] = 0.0f;

    // 回放: 模拟当时的历史
    std::vector<std::pair<int, std::vector<int>>> pastHistory;
    for (const auto& entry : history) {
        int pos = entry.first;
        const auto& cards = entry.second;

        if (!cards.empty() && isLeadAction(cards, pastHistory)) {
            Combo combo(cards.begin(), cards.end());
            int cat = classifyCtpCategory(combo.type);
            if (cat >= 0 && cat < NUM_CTP_TYPES) {
                // EMA 更新: CTP = γ·old + (1-γ)·1
                float oldVal = ctpState[pos][cat];
                ctpState[pos][cat] = CTP_GAMMA * oldVal + (1.0f - CTP_GAMMA) * 1.0f;
                // 其余类别纯衰减
                for (int t = 0; t < NUM_CTP_TYPES; t++) {
                    if (t != cat) ctpState[pos][t] *= CTP_GAMMA;
                }
            }
        }
        pastHistory.push_back(entry);
    }
}

// C4 主接口: 从 GameState 回放历史, 提取 CTP 特征 (14 维: 7类×2对手)
// 与 Python compute_ctp_features / update_ctp 严格对齐:
//   1. 回放 cardPlayActionSeq (≤~100手)
//   2. 仅当该手为"主动领出"时更新: CTP[pos][cat] = γ·old + (1-γ)·1,
//      同时对同一位置其余 6 类纯衰减 ×γ
//   3. 非领出回合不更新 (CTP 只统计领出行为, 与 Python 侧一致)
// 注: CTP 不含 SSHUTTLE 类型 (航天飞机不在标准牌型分类中)
inline std::vector<float> computeCtpFeatures(const GameState& state,
                                              int upPos, int downPos) {
    float ctpState[3][7] = {};
    computeCtpStateFromHistory(state.cardPlayActionSeq, ctpState);

    std::vector<float> feat;
    feat.reserve(NUM_CTP_TYPES * 2);
    for (int t = 0; t < NUM_CTP_TYPES; t++)
        feat.push_back(ctpState[upPos][t]);
    for (int t = 0; t < NUM_CTP_TYPES; t++)
        feat.push_back(ctpState[downPos][t]);
    return feat;
}

// ═══════════════════════════════════════════════════════════════════════════
// C5: 对手行为风格 SWI + TP — 4 维
// ═══════════════════════════════════════════════════════════════════════════

constexpr float SWI_GAMMA = 0.9f;
constexpr float TP_GAMMA  = 0.9f;
constexpr float SWI_INIT_NUM = 1.0f;
constexpr float SWI_INIT_DEN = 2.0f;
constexpr float TP_MAX_CARDS = 6.0f;

// need(t): 牌型所需最少同等级张数 (用于 p_same 分母)
inline int needForType(ComboType t) {
    switch (t) {
        case ComboType::SINGLE:        return 1;
        case ComboType::PAIR:          return 2;
        case ComboType::TRIPLE:
        case ComboType::TRIPLE_1:
        case ComboType::TRIPLE_2:      return 3;
        case ComboType::BOMB:          return 4;
        case ComboType::KING_BOMB:     return 4;
        case ComboType::SERIAL_SINGLE: return 1;
        case ComboType::SERIAL_PAIR:   return 2;
        case ComboType::SERIAL_TRIPLE:
        case ComboType::SERIAL_3_1:
        case ComboType::SERIAL_3_2:    return 3;
        case ComboType::QUAD_2_SINGLE:
        case ComboType::QUAD_2_PAIR:   return 4;
        default:                       return 1;
    }
}

// 无"领出等级"的哨兵值（训练端 move_detector 对航天飞机不返回 rank）
constexpr Level kNoLeadLevel = -1;

// 从出牌列表提取 (ComboType, **C5 用的领出等级 ℓ**)
// 与训练端 opponent_modeling._get_combo_type_level 逐类型对齐：
//   - 三带一/三带二: 取**三条**的等级；四带二/四带两对: 取**四条**的等级。
//     （不能用 Combo::main_level —— 它在带牌比主牌大时会取到带牌等级，与训练端不一致；
//       也不能改 Combo::main_level，那是比大小逻辑用的）
//   - 顺子/连对/飞机: Combo::main_level 已经是链条最高等级，与训练端一致
//   - 航天飞机: 训练端没有 rank（lead_level=None）→ 返回 kNoLeadLevel
//   - 其它（单张/对子/三条/炸弹）: main_level 即该等级
inline std::pair<ComboType, Level> getComboTypeLevel(const std::vector<int>& cards) {
    if (cards.empty()) return {ComboType::PASS, kNoLeadLevel};
    Combo combo(cards.begin(), cards.end());
    auto rankCounts = [&](int counts[15]) {
        for (int i = 0; i < 15; i++) counts[i] = 0;
        for (int c : cards) {
            int r = botzoneCardToRank(c);
            if (r >= 0 && r < 15) counts[r]++;
        }
    };
    switch (combo.type) {
        case ComboType::TRIPLE_1:
        case ComboType::TRIPLE_2: {
            int counts[15];
            rankCounts(counts);
            for (int r = 14; r >= 0; r--)
                if (counts[r] >= 3) return {combo.type, (Level)r};
            break;
        }
        case ComboType::QUAD_2_SINGLE:
        case ComboType::QUAD_2_PAIR: {
            int counts[15];
            rankCounts(counts);
            for (int r = 14; r >= 0; r--)
                if (counts[r] >= 4) return {combo.type, (Level)r};
            break;
        }
        case ComboType::SSHUTTLE:
        case ComboType::SSHUTTLE_2:
        case ComboType::SSHUTTLE_4:
            return {combo.type, kNoLeadLevel};
        default:
            break;
    }
    return {combo.type, combo.main_level};
}

// 估计对手能压制当前领出的概率 p_can(t, ℓ)
// p_can = 1 − (1 − p_same) · (1 − OBP_o)
inline float estimatePCan(ComboType leadType, Level leadLevel,
                           const std::array<int, 15>& poolVec, int poolTotal,
                           int opponentRemaining, float opponentObp) {
    if (leadType == ComboType::PASS || leadType == ComboType::INVALID)
        return 1.0f;  // 无法识别 → 保守估计总能压制

    // 火箭不可被压制
    if (leadType == ComboType::KING_BOMB) return 0.0f;

    // 无领出等级（航天飞机）: 与训练端一致 —— move_detector 不给 rank，
    // 训练端 _estimate_p_can 走"无法识别"分支返回 1.0（保守估计总能压制）。
    if (leadLevel < 0) return 1.0f;

    // 炸弹: 仅火箭可压制 (OBP 中约 5% 是火箭)
    if (leadType == ComboType::BOMB)
        return opponentObp * 0.05f;

    if (opponentRemaining <= 0 || poolTotal <= 0) return 0.0f;

    int need = needForType(leadType);

    // p_same = min(1.0, (1/need) · Σ_{r > ℓ, n_r >= need} c_o·n_r/N)
    float pSameSum = 0.0f;
    for (int r = leadLevel + 1; r < 15; r++) {
        int n_r = poolVec[r];
        if (n_r >= need) {
            pSameSum += (float)(opponentRemaining * n_r) / (float)poolTotal;
        }
    }
    float pSame = std::min(1.0f, pSameSum / (float)need);

    return 1.0f - (1.0f - pSame) * (1.0f - opponentObp);
}

// C5 回放状态
struct C5ReplayState {
    float swi_num[3] = {SWI_INIT_NUM, SWI_INIT_NUM, SWI_INIT_NUM};
    float swi_den[3] = {SWI_INIT_DEN, SWI_INIT_DEN, SWI_INIT_DEN};
    float tp_val[3]  = {0.5f, 0.5f, 0.5f};

    bool  roundActive    = false;
    int   prevPassPos    = -1;
    ComboType curLeadType = ComboType::PASS;
    Level   curLeadLevel  = -1;
};

// 回放历史计算 C5 状态 (SWI + TP)
//
// 关键: p_can 必须用**该动作发生时刻**的未知牌池 / 剩余牌数 / OBP,
// 不能用当前时刻的快照 —— 后者会把动作之后才发生的信息 (哪些牌已被打掉)
// 回填到历史动作上, 既与训练端不一致, 也构成特征层面的少量事后信息。
//
// 做法: 先从"出牌阶段开始"的局面重建牌池与剩余牌数, 再沿历史逐手推进;
// 走到历史末尾时, 牌池恰好等于 buildUnknownPool(当前局面) 的结果, 因此与 C3 用的池自洽。
// 这与 Python 端 update_c5_from_history 的增量语义一致: 每个动作在其首次被观测到的
// 那一刻用当刻牌池计算 p_can (自对弈与评估中, 每次观测恰好新增一个动作)。
inline void computeC5StateFromHistory(
    const GameState& state,
    const std::vector<std::pair<int, std::vector<int>>>& history,
    C5ReplayState& st)
{
    st = C5ReplayState{};

    int ldPos = state.landlordPosition;

    auto playedOf = [&](int pos) -> const std::vector<int>& {
        if (pos == ldPos) return state.landlordPlayedCards;
        if (pos == (ldPos + 1) % 3) return state.landlordDownPlayedCards;
        return state.landlordUpPlayedCards;
    };
    auto remainingOf = [&](int pos) {
        if (pos == ldPos) return state.landlordCardsLeft;
        if (pos == (ldPos + 1) % 3) return state.landlordDownCardsLeft;
        return state.landlordUpCardsLeft;
    };

    // ── 1. 重建"出牌阶段开始时"的未知牌池 ──
    // 当时我的手牌 = 当前手牌 + 我已经打出的牌; 当时三家都还没出过牌。
    std::array<int, 15> poolVec{};
    for (int i = 0; i < 13; i++) poolVec[i] = 4;
    poolVec[13] = 1;
    poolVec[14] = 1;

    auto subRank = [&](const std::vector<int>& cards) {
        for (int c : cards) {
            int r = botzoneCardToRank(c);
            if (r >= 0 && r < 15) poolVec[r]--;
        }
    };
    subRank(state.myHandCards);
    // 底牌只对非地主扣（地主手里已含底牌，再扣是重复扣减）
    if (state.myPosition != state.landlordPosition)
        subRank(state.threeLandlordCards);
    subRank(playedOf(state.myPosition));   // 我之后打出的牌, 在开局时还在我手里

    // 开局各位置剩余牌数 = 当前剩余 + 该位置已打出的张数
    int cardsRemaining[3];
    for (int pos = 0; pos < 3; pos++)
        cardsRemaining[pos] = remainingOf(pos) + (int)playedOf(pos).size();

    // ── 2. 沿历史逐手推进 ──
    for (const auto& entry : history) {
        int pos = entry.first;
        const auto& cards = entry.second;

        // 该动作的牌此刻已离手 → 先推进牌池与剩余牌数, 再用当刻快照计算 p_can
        //
        // 牌池只扣**别人**打出的牌: 开局池里已经把我方手牌（含底牌）整手扣掉了,
        // 再把我自己之后打出的牌扣一次就是重复扣减（牌池会比训练端少"我出牌数"张）。
        // 训练端 _build_unknown_pool(hand_self=当前手牌, all_played, ...) 里，
        // 我的手牌与我的已出牌互斥，合计恒等于"开局我方手牌"，等价于只扣一次。
        if (!cards.empty()) {
            if (pos != state.myPosition) {
                // 底牌已在开局整体扣过（且我方手牌已整手扣过）→ 不再重复扣
                for (int c : cards) {
                    bool isBottom = false;
                    for (int b : state.threeLandlordCards) {
                        if (b == c) { isBottom = true; break; }
                    }
                    if (isBottom) continue;
                    int r = botzoneCardToRank(c);
                    if (r >= 0 && r < 15) poolVec[r]--;
                }
            }
            if (pos >= 0 && pos < 3 && cardsRemaining[pos] > 0)
                cardsRemaining[pos] -= (int)cards.size();
        }

        if (!st.roundActive) {
            if (!cards.empty()) {
                // 新一轮: 该玩家获得控牌权 → TP 更新
                st.roundActive = true;
                st.prevPassPos = -1;
                auto [ct, cl] = getComboTypeLevel(cards);
                st.curLeadType = ct;
                st.curLeadLevel = cl;

                float nCards = (float)cards.size();
                float normCards = std::min(nCards / TP_MAX_CARDS, 1.0f);
                st.tp_val[pos] = TP_GAMMA * st.tp_val[pos]
                               + (1.0f - TP_GAMMA) * normCards;
            }
        } else {
            // 轮次激活中: respond 机会 → SWI 更新
            float didSuppress = cards.empty() ? 0.0f : 1.0f;

            // p_can 仅 pass 时计算 (压制时权重=1.0, 不影响分子/分母比例)
            float pCan = 1.0f;
            if (cards.empty() && pos >= 0 && pos < 3) {
                int poolTotal = 0;
                for (int i = 0; i < 15; i++)
                    poolTotal += (poolVec[i] > 0 ? poolVec[i] : 0);  // 防御: 与 buildUnknownPool 一致

                float obp = computeOBPFromPool(poolVec, poolTotal, cardsRemaining[pos]);
                pCan = estimatePCan(st.curLeadType, st.curLeadLevel,
                                    poolVec, poolTotal,
                                    cardsRemaining[pos], obp);
                pCan = std::max(pCan, 0.01f);  // 防止除零
            }

            st.swi_num[pos] = SWI_GAMMA * st.swi_num[pos]
                            + (1.0f - SWI_GAMMA) * didSuppress;
            st.swi_den[pos] = SWI_GAMMA * st.swi_den[pos]
                            + (1.0f - SWI_GAMMA) * pCan;

            if (!cards.empty()) {
                // 压制 — 更新当前领出
                auto [ct, cl] = getComboTypeLevel(cards);
                st.curLeadType = ct;
                st.curLeadLevel = cl;
                st.prevPassPos = -1;
            } else {
                // pass — 检查轮次结束 (连续两次 pass 且非同一人)
                if (st.prevPassPos >= 0 && st.prevPassPos != pos) {
                    st.roundActive = false;
                    st.curLeadType = ComboType::PASS;
                    st.curLeadLevel = -1;
                    st.prevPassPos = -1;
                } else {
                    st.prevPassPos = pos;
                }
            }
        }
    }
}

// C5 主接口: 从 GameState 回放历史, 提取 SWI + TP 特征 (4 维)
// 与 Python 端 compute_c5_features / update_c5_from_history 对齐:
//   SWI = EMA(是否压制) / EMA(期望压制概率 p_can), 初始值 num/den = 1/2 = 0.5
//   TP  = EMA(首手出牌数 / 6), 初始值 0.5
//   仅统计对手回合, 不追踪自己
// 注意: p_can 使用每个动作当刻的牌池, 因此本函数自行重建并推进牌池,
//       不再复用 C3 的"当前池快照"(那会把事后信息回填到历史动作上)。
inline std::vector<float> computeC5Features(const GameState& state,
                                             int upPos, int downPos) {
    C5ReplayState st;
    computeC5StateFromHistory(state, state.cardPlayActionSeq, st);

    float swi_up   = st.swi_num[upPos]   / std::max(st.swi_den[upPos], 0.01f);
    float swi_down = st.swi_num[downPos] / std::max(st.swi_den[downPos], 0.01f);
    float tp_up    = st.tp_val[upPos];
    float tp_down  = st.tp_val[downPos];

    return {swi_up, swi_down, tp_up, tp_down};
}

#endif // OPPONENT_MODELING_H
