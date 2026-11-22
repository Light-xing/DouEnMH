// =============================================================================
// card_encoding.h — DouEnMH 8×15 卡牌编码矩阵 (120维)
// =============================================================================
// 与 Python env.py cards2array_v2() / utils.py _cards2tensor_v2() 严格对齐
//
// 设计依据: input_design.md A节 — 8×15 卡牌编码矩阵
//
// 8 行 (Row) × 15 列 (Col) = 120 维, 展平: 列主序 (Fortran order)
// 列主序使得同一牌点的8行特征在120维向量中连续存储,
// 有利于 Conv1d 逐点投影 (kernel=1 时不会跨列混合信息).
//
//   Row 0: 计数 ≥1  (DouZero 风格增量独热) — 该点数至少持有1张
//   Row 1: 计数 ≥2  — 该点数至少持有2张 (可组成对子)
//   Row 2: 计数 ≥3  — 该点数至少持有3张 (可组成三条)
//   Row 3: 计数 =4  — 该点数恰好持有4张 (可组成炸弹)
//   Row 4: 断点标记 (count[i]≥1 ∧ count[i+1]=0) — 连续持牌段的右边界
//   Row 5: 单顺起点 (≥5连, Row0[i..i+4]全1) — 以该等级为起点的单顺存在性
//   Row 6: 连对起点 (≥3连对, Row1[i..i+2]全1) — 以该等级为起点的连对存在性
//   Row 7: 飞机起点 (≥2连三条, Row2[i..i+1]全1) — 以该等级为起点的飞机存在性
//
// 列 0-12: 3,4,5,6,7,8,9,10,J,Q,K,A,2
// 列 13:   小王 (env=20)
// 列 14:   大王 (env=30)
//
// Joker 约束: 大小王仅 Row 0 可=1 (不存在≥2张), Row 1-7 永为 0
// 链类约束: MAX_CHAIN_RANK=12, 链类牌型仅用列0-11(3-A), 2/joker不可入链
//
// 历史编码约定 (与 input_design.md B节一致):
//   - 有效出牌: cards2array_v2(cards) → 120维
//   - pass (实际不出): 空列表→全零120维向量 (携带战术信息: 无法压制/战术放弃)
//   - padding (序列不足35手): 全-1 哨兵值 (纯占位, 与pass语义截然不同)
//   - 区分理由: Conv1d投影后, pass与padding映射到嵌入空间的不同区域,
//     Transformer自注意力可学习对两者做截然不同的处理.
// =============================================================================

#ifndef CARD_ENCODING_H
#define CARD_ENCODING_H

#include <vector>
#include <array>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>

// ── 常量定义 ──────────────────────────────────────────────────────────────

constexpr int CARD_MATRIX_ROWS   = 8;    // 矩阵行数
constexpr int CARD_MATRIX_COLS   = 15;   // 矩阵列数 (13个普通等级 + 大小王)
constexpr int CARD_MATRIX_DIM    = CARD_MATRIX_ROWS * CARD_MATRIX_COLS;  // 120

constexpr int MIN_SINGLE_CHAIN   = 5;   // 单顺: ≥5连
constexpr int MIN_PAIR_CHAIN     = 3;   // 连对: ≥3连对
constexpr int MIN_TRIPLE_CHAIN   = 2;   // 飞机: ≥2连三条
constexpr int MAX_CHAIN_RANK     = 12;  // 链类牌型可用列0-11 (3-A), 排除2/joker

constexpr int HISTORY_LENGTH     = 35;  // 最近35手出牌历史

// ── Card → Column 映射表 ──────────────────────────────────────────────────
// env_card → column index (0-14)
// {3:0, 4:1, 5:2, 6:3, 7:4, 8:5, 9:6, 10:7, 11:8, 12:9, 13:10, 14:11, 17:12, 20:13, 30:14}
// 注意: 2在env编码中为17(非15), 因为15/16为Botzone兼容预留

inline int card2col(int envCard) {
    if (envCard >= 3 && envCard <= 14) return envCard - 3;  // 3..A → 0..11
    if (envCard == 17) return 12;                            // 2 → 12
    if (envCard == 20) return 13;                            // 小王 → 13
    if (envCard == 30) return 14;                            // 大王 → 14
    return -1;  // 无效编码
}

// Botzone Card ID (0-53) → env_card (3-14, 17, 20, 30)
// 转换公式: level = card/4 + card/53, 再映射到env_card
inline int botzoneCardToEnvCard(int cardID) {
    if (cardID == 52) return 20;       // 小王
    if (cardID == 53) return 30;       // 大王
    int level = cardID / 4;
    if (level <= 11) return level + 3; // 3..14
    return 17;                          // 2
}

// Botzone Card ID 列表批量转换为 env_card 列表
inline std::vector<int> handToEnvCards(const std::vector<int>& botzoneCards) {
    std::vector<int> result;
    for (int c : botzoneCards) result.push_back(botzoneCardToEnvCard(c));
    return result;
}

// ── Row 0-3 增量独热编码查找表 ────────────────────────────────────────────
// NUM_ONES[count][row]: count张时第row行应填的值
// 例如 count=2 → Row0=1, Row1=1, Row2=0, Row3=0
constexpr int NUM_ONES[5][8] = {
    {0, 0, 0, 0, 0, 0, 0, 0},  // count=0
    {1, 0, 0, 0, 0, 0, 0, 0},  // count=1
    {1, 1, 0, 0, 0, 0, 0, 0},  // count=2
    {1, 1, 1, 0, 0, 0, 0, 0},  // count=3
    {1, 1, 1, 1, 0, 0, 0, 0},  // count=4
};


// ═══════════════════════════════════════════════════════════════════════════
// cards2array_v2 — 核心 8×15 编码函数
// ═══════════════════════════════════════════════════════════════════════════
// 设计依据: input_design.md A.2-A.5节
// 输入: env_card 列表 (3-14, 17, 20, 30)
// 输出: 120维 float 向量 (Fortran-order: 逐列展平)
// 空列表返回全零向量 (对应pass编码 — 语义上表示"玩家选择不出牌")

inline std::vector<float> cards2array_v2(const std::vector<int>& envCards) {
    std::vector<float> result(CARD_MATRIX_DIM, 0.0f);

    if (envCards.empty()) return result;

    // ── Step 1: 计数填充 Row 0–3 (来源: DouZero 增量独热) ──
    // 对每种点数统计张数, n张 → Row0..Row(n-1)=1, 上限4行
    // 大小王最多1张, 故仅Row0[13]/[14]=1
    std::unordered_map<int, int> counter;
    for (int c : envCards) counter[c]++;

    for (auto& kv : counter) {
        int card = kv.first;
        int n = kv.second;
        int col = card2col(card);
        if (col < 0) continue;

        int fill_rows = (n < 4) ? n : 4;  // 最多填4行
        for (int row = 0; row < fill_rows; row++)
            result[col * CARD_MATRIX_ROWS + row] = 1.0f;
    }

    // ── Step 2: 断点标记 Row 4 (来源: DouMH, 独立行展开) ──
    // 标记连续持牌段的右边界: count[i]≥1 且 count[i+1]=0
    // 这帮助Transformer感知手牌的"连续段"结构, 无需解耦OR逻辑
    // 仅对普通牌(列0-13)检测断点, 大小王不参与等级断点
    std::unordered_set<int> ranksPresent;
    for (int c : envCards) {
        if (c < 20) ranksPresent.insert(card2col(c));
    }
    for (int i = 0; i < 14; i++) {  // i=13: 检查小王是否有后继(大王不算等级连续)
        if (ranksPresent.count(i) && !ranksPresent.count(i + 1))
            result[i * CARD_MATRIX_ROWS + 4] = 1.0f;
    }

    // ── Step 3: 单顺起点 Row 5 (来源: PerfectDou, ≥5连, 起点0..7) ──
    // 链长不区分: 无论链长5/6/7/.../12, 只要≥5连存在, 起点列即标1
    // Transformer可通过自注意力跨列比较推断真实链长
    for (int i = 0; i <= MAX_CHAIN_RANK - MIN_SINGLE_CHAIN; i++) {
        bool all_one = true;
        for (int k = 0; k < MIN_SINGLE_CHAIN; k++) {
            if (result[(i + k) * CARD_MATRIX_ROWS + 0] != 1.0f) {
                all_one = false;
                break;
            }
        }
        if (all_one) result[i * CARD_MATRIX_ROWS + 5] = 1.0f;
    }

    // ── Step 4: 连对起点 Row 6 (来源: PerfectDou, ≥3连对, 起点0..9) ──
    for (int i = 0; i <= MAX_CHAIN_RANK - MIN_PAIR_CHAIN; i++) {
        bool all_one = true;
        for (int k = 0; k < MIN_PAIR_CHAIN; k++) {
            if (result[(i + k) * CARD_MATRIX_ROWS + 1] != 1.0f) {
                all_one = false;
                break;
            }
        }
        if (all_one) result[i * CARD_MATRIX_ROWS + 6] = 1.0f;
    }

    // ── Step 5: 飞机起点 Row 7 (来源: PerfectDou, ≥2连三条, 起点0..10) ──
    for (int i = 0; i <= MAX_CHAIN_RANK - MIN_TRIPLE_CHAIN; i++) {
        bool all_one = true;
        for (int k = 0; k < MIN_TRIPLE_CHAIN; k++) {
            if (result[(i + k) * CARD_MATRIX_ROWS + 2] != 1.0f) {
                all_one = false;
                break;
            }
        }
        if (all_one) result[i * CARD_MATRIX_ROWS + 7] = 1.0f;
    }

    return result;
}

// 便捷重载: 直接接受 Botzone Card ID (0-53) 列表, 内部转换为env_card
inline std::vector<float> cards2array_v2_from_botzone(const std::vector<int>& botzoneCards) {
    return cards2array_v2(handToEnvCards(botzoneCards));
}


// ═══════════════════════════════════════════════════════════════════════════
// 出牌历史序列编码 — 与 Python _action_seq_list2array() 严格对齐
// ═══════════════════════════════════════════════════════════════════════════

// processActionSeq_typed — 截取最近 N 手 + 反转 + 填充, 区分 pass 与 padding
// seq: [(pos, cards), ...] 其中 cards 为 Botzone Card ID (0-53) 列表
// 返回: length 个 pair<bool, vector<int>>, 最近的在前
//   - first=true:  真实动作 (second 为 env_card 列表, 空列表=pass)
//   - first=false: padding 哨兵 (不足35手的占位)
// 设计依据: input_design.md B节 — 历史编码需区分 pass(全零) 与 padding(全-1)
inline std::vector<std::pair<bool, std::vector<int>>> processActionSeq_typed(
    const std::vector<std::pair<int, std::vector<int>>>& seq,
    int length = HISTORY_LENGTH)
{
    std::vector<std::pair<bool, std::vector<int>>> result;
    int start = std::max(0, (int)seq.size() - length);
    for (int i = start; i < (int)seq.size(); i++) {
        std::vector<int> envCards;
        for (int c : seq[i].second)
            envCards.push_back(botzoneCardToEnvCard(c));
        result.push_back({true, envCards});
    }

    std::reverse(result.begin(), result.end());  // 最近的在前面

    // 序列不足35手时用 padding 填充
    int paddingNeeded = length - (int)result.size();
    for (int i = 0; i < paddingNeeded; i++)
        result.push_back({false, {}});

    return result;
}

// actionSeqList2Array_typed — 区分 pass 与 padding 的序列编码
// 与 Python _action_seq_list2array() 严格对齐:
//   - 有效动作(含pass) → cards2array_v2 编码 (pass=空列表→全零120维)
//   - padding 哨兵 → 全-1 向量
inline std::vector<std::vector<float>> actionSeqList2Array_typed(
    const std::vector<std::pair<bool, std::vector<int>>>& seq)
{
    std::vector<std::vector<float>> arr;
    for (auto& entry : seq) {
        if (entry.first) {
            // 真实动作: 空列表=pass → cards2array_v2返回全零
            arr.push_back(cards2array_v2(entry.second));
        } else {
            // padding 哨兵: 全-1
            arr.push_back(std::vector<float>(CARD_MATRIX_DIM, -1.0f));
        }
    }
    return arr;
}


// ═══════════════════════════════════════════════════════════════════════════
// 辅助编码函数
// ═══════════════════════════════════════════════════════════════════════════

// getOneHotBomb — 已出炸弹数 one-hot 编码 (15维, 索引0-14)
// 设计依据: input_design.md C节 — 炸弹数特征 (属于X向量)
inline std::vector<float> getOneHotBomb(int bombNum) {
    std::vector<float> oneHot(15, 0.0f);
    if (bombNum >= 0 && bombNum < 15)
        oneHot[bombNum] = 1.0f;
    return oneHot;
}

// computeHandStats — 自手牌型统计 (C1, 7维归一化线性标量)
// 设计依据: input_design.md C1节
// 统计当前手牌中各类牌型的"可出"数量:
//   n_solo(可出单张数), n_pair(可出对子数), n_trio(可出三条数),
//   n_bomb(可出炸弹/火箭数), n_chain_solo(可出单顺数),
//   n_chain_pair(可出连对数), n_airplane(可出飞机数)
// 各维度用经验上界做线性归一化, 截断到[0,1]
inline std::vector<float> computeHandStats(const std::vector<int>& envCards) {
    int rankCnt[15] = {0};
    for (int c : envCards) {
        int col = card2col(c);
        if (col >= 0 && col < 15) rankCnt[col]++;
    }

    // 单张/对子/三条/炸弹: 统计持有≥1/≥2/≥3/=4的等级数
    int n_solo = 0, n_pair = 0, n_trio = 0, n_bomb = 0;
    for (int i = 0; i < 15; i++) {
        if (rankCnt[i] >= 1) n_solo++;
        if (i < 13 && rankCnt[i] >= 2) n_pair++;   // 大小王无对子
        if (i < 13 && rankCnt[i] >= 3) n_trio++;   // 大小王无三条
        if (i < 13 && rankCnt[i] == 4) n_bomb++;   // 普通炸弹
    }
    if (rankCnt[13] >= 1 && rankCnt[14] >= 1) n_bomb++;  // 火箭也算炸弹

    // 单顺: 滑动窗口统计≥5连的起点数 (仅列0-11)
    int n_chain_solo = 0;
    for (int i = 0; i <= MAX_CHAIN_RANK - MIN_SINGLE_CHAIN; i++) {
        bool ok = true;
        for (int k = 0; k < MIN_SINGLE_CHAIN; k++)
            if (rankCnt[i + k] < 1) { ok = false; break; }
        if (ok) n_chain_solo++;
    }

    // 连对: 滑动窗口统计≥3连对的起点数
    int n_chain_pair = 0;
    for (int i = 0; i <= MAX_CHAIN_RANK - MIN_PAIR_CHAIN; i++) {
        bool ok = true;
        for (int k = 0; k < MIN_PAIR_CHAIN; k++)
            if (rankCnt[i + k] < 2) { ok = false; break; }
        if (ok) n_chain_pair++;
    }

    // 飞机: 滑动窗口统计≥2连三条的起点数
    int n_airplane = 0;
    for (int i = 0; i <= MAX_CHAIN_RANK - MIN_TRIPLE_CHAIN; i++) {
        bool ok = true;
        for (int k = 0; k < MIN_TRIPLE_CHAIN; k++)
            if (rankCnt[i + k] < 3) { ok = false; break; }
        if (ok) n_airplane++;
    }

    // 线性归一化: 各维度除以经验上界, 截断到[0,1]
    return {
        std::min(n_solo / 15.0f, 1.0f),        // 单张最多~15种等级
        std::min(n_pair / 8.0f, 1.0f),         // 对子最多~8种
        std::min(n_trio / 5.0f, 1.0f),         // 三条最多~5种
        std::min(n_bomb / 6.0f, 1.0f),         // 炸弹最多~6个
        std::min(n_chain_solo / 10.0f, 1.0f),  // 单顺起点最多~10个
        std::min(n_chain_pair / 6.0f, 1.0f),   // 连对起点最多~6个
        std::min(n_airplane / 4.0f, 1.0f),     // 飞机起点最多~4个
    };
}

#endif // CARD_ENCODING_H
