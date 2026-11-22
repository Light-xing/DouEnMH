// =============================================================================
// feature_encoder.h — DouEnMH 特征编码 (8×15=120-dim Z矩阵 + 52/56-dim X向量)
// =============================================================================
// 设计依据: input_design.md B/C节 + Botzone_DouEnMH_推理部署设计文档.md 五章
//
// 基于 card_encoding.h 的 8×15 卡牌矩阵, 组装两大输入模块:
//   Z 矩阵 (40行×120维) — 送入 Transformer Encoder, 编码卡牌结构信息
//   X 向量 (52或56维)    — 标量辅助特征, 在Transformer输出后拼接
//
// Z矩阵行布局 (input_design.md B节):
//   Row 0:    当前手牌 (my_hand, 1×8×15)
//   Row 1:    对手手牌并集 (other_union, 1×8×15, 推理时=未知牌池)
//   Row 2:    自身累计已出牌 (played_self, 1×8×15)
//   Row 3:    上手累计已出牌 (played_up, 1×8×15)
//   Row 4:    下手累计已出牌 (played_down, 1×8×15)
//   Row 5-39: 最近35手出牌历史 (含pass=全零 / padding=-1), 有位置编码
//
// X向量布局 (input_design.md C节):
//   [0-2]:   叫分信息 bidInfo/3
//   [3-5]:   三人剩余牌数/20
//   [6]:     春天标志
//   [7-21]:  已出炸弹数 one-hot(15)
//   [22-28]: C1 自手牌型统计 (7维)
//   [29-33]: C2 神谕特征 (5维: n_active/n_passive/delta_n/n_preserve/self_adv)
//   [34-37]: C3 对手信念 (4维: OHS×2 + OBP×2)
//   [38-51]: C4 对手牌型偏好CTP (14维: 7×2)
//   [52-55]: C5 对手行为风格 (4维: SWI×2 + TP×2, 可选)
//
// 与 Python env.py _get_obs_resnet() / _get_bid_obs_resnet() 严格对齐
// =============================================================================

#ifndef FEATURE_ENCODER_H
#define FEATURE_ENCODER_H

#include "card_encoding.h"
#include "game_state.h"
#include <cstring>
#include <cassert>

// ── X 向量维度定义 ────────────────────────────────────────────────────────
// 不含C5时52维, 含C5时56维 (与 opponent_modeling_design.md 3.3节一致)
constexpr int X_DIM     = 52;  // C1-C4 (不含C5行为风格)
constexpr int X_DIM_C5  = 56;  // C1-C5 (含C5行为风格)
// 各子模块维度
constexpr int X_BID_INFO    = 3;   // 叫分信息 (三人叫分值÷3)
constexpr int X_NUM_CARDS   = 3;   // 三人剩余牌数 (self/up/down ÷20)
constexpr int X_SPRING      = 1;   // 春天标志 (bool)
constexpr int X_BOMB        = 15;  // 已出炸弹数 one-hot (0-14)
constexpr int X_HAND_STATS  = 7;   // C1 自手牌型统计 (n_solo/pair/trio/bomb/chain_solo/chain_pair/airplane)
constexpr int X_ORACLE      = 5;   // C2 神谕特征 (n_active/n_passive/delta_n/n_preserve/self_adv)
constexpr int X_OPPONENT    = 4;   // C3 对手信念特征 (OHS_up/down + OBP_up/down)
constexpr int X_CTP         = 14;  // C4 对手牌型偏好 (CTP_up[7] + CTP_down[7])
constexpr int X_C5          = 4;   // C5 对手行为风格 (SWI_up/down + TP_up/down)

// Z 矩阵行数定义
constexpr int Z_STATIC_ROWS  = 5;   // 静态卡牌状态: hand + other + 3×played
constexpr int Z_HISTORY_ROWS = 35;  // 最近35手出牌历史
constexpr int Z_TOTAL_ROWS   = Z_STATIC_ROWS + Z_HISTORY_ROWS;  // 40

// z_batch 行数: 1个动作行 + 40行Z矩阵 (V1格式)
constexpr int Z_BATCH_ROWS   = Z_TOTAL_ROWS + 1;  // 41


// ═══════════════════════════════════════════════════════════════════════════
// PlayFeatureBatch — 出牌阶段特征批次
// ═══════════════════════════════════════════════════════════════════════════
// V1 格式 (兼容 DouMH Transformer 直接输入, 当前main.cpp使用):
//   z_batch:  (B, 41, 120) = 1 action + 5 static + 35 history
//   x_batch:  (B, 52) or (B, 56)
//
// V2 Cross-Attention 格式 (预留, 对应action_encoder_cross_attention_design.md):
//   state_z:   (1, 5, 120)   — 静态卡牌状态 (hand + other + played×3)
//   action_z:  (B, 1, 120)   — N个候选动作, 每个为单token
//   history_z: (1, 35, 120)  — 最近35手出牌历史
//   x_batch:   (B, 52) or (B, 56)

struct PlayFeatureBatch {
    // V1 格式 — 当前推理管线使用
    std::vector<float> z_batch;   // (B, 41, 120) flat, row-major
    std::vector<float> x_batch;   // (B, X_DIM) flat, row-major

    // V2 拆分 — 预留Cross-Attention接口
    std::vector<float> state_z;   // (1, 5, 120) flat
    std::vector<float> action_z;  // (B, 1, 120) flat
    std::vector<float> history_z; // (1, 35, 120) flat

    int batch_size = 0;
    int x_dim = X_DIM;         // 实际X维度 (52或56, 取决于是否启用C5)
};

// ═══════════════════════════════════════════════════════════════════════════
// BidFeatureBatch — 叫分阶段特征批次
// ═══════════════════════════════════════════════════════════════════════════
// 设计依据: input_design.md E节 — 叫分网络输入
//   z_batch: (B, 2, 120) = Row0叫分动作广播 + Row1手牌
//   x_batch: (B, 3)     = 三人叫分值 (未归一化, 由模型内部处理)

struct BidFeatureBatch {
    std::vector<float> z_batch;  // (B, 2, 120) flat
    std::vector<float> x_batch;  // (B, 3) flat
    int batch_size = 0;
};


// ═══════════════════════════════════════════════════════════════════════════
// buildPlayFeatures — 组装出牌阶段特征 (主入口)
// ═══════════════════════════════════════════════════════════════════════════
// 设计依据: Botzone_DouEnMH_推理部署设计文档.md 五章 + input_design.md B/C节
//
// 组装流程:
//   1. 预计算5行静态Z矩阵: hand/other/self_played/up_played/down_played
//   2. 编码35手出牌历史: 区分真实动作(pass→全零)/padding(全-1)
//   3. 组装X向量: bidInfo + cardsLeft + spring + bombNum + C1-C5
//   4. 对每个合法动作: 编码其卡牌矩阵 → prepend到Z矩阵前方 → 构成z_batch
//   5. 同时输出V2 Cross-Attention拆分格式 (state_z/action_z/history_z)
//
// 参数:
//   state:        游戏状态 (GameState, 包含手牌/已出牌/历史等全部信息)
//   legalActions: 合法动作列表, 每个动作为 Botzone Card ID (0-53) 列表
//   oracleFeat:   C2 神谕特征 (B×5 dim), 每个候选动作的神谕值不同 (逐动作计算, 评估出牌后剩余手牌).
//                 外部由 main.cpp 逐动作调用 oracle.cpp compute_oracle_raw() + normalize_oracle() 计算
//   opponentFeat: C3 对手信念特征 (4维), 外部由 opponent_modeling.h computeOpponentFeatures() 计算
//   ctpFeat:      C4 对手牌型偏好 (14维), 外部由 opponent_modeling.h computeCtpFeatures() 计算
//   c5Feat:       C5 对手行为风格 (4维), 外部由 opponent_modeling.h computeC5Features() 计算;
//                 传空vector则关闭C5, 此时X维度=52, x_dim字段自动调整
// 返回: PlayFeatureBatch (z_batch, x_batch, state_z, action_z, history_z)

inline PlayFeatureBatch buildPlayFeatures(
    const GameState& state,
    const std::vector<std::vector<int>>& legalActions,
    const std::vector<std::vector<float>>& oracleFeat = {},  // C2: (B, 5) per-action
    const std::vector<float>& opponentFeat = {},    // C3: 4 dim
    const std::vector<float>& ctpFeat = {},         // C4: 14 dim
    const std::vector<float>& c5Feat = {})          // C5: 4 dim (optional)
{
    PlayFeatureBatch batch;
    int B = (int)legalActions.size();
    batch.batch_size = B;

    bool hasC5 = (c5Feat.size() >= X_C5);
    batch.x_dim = hasC5 ? X_DIM_C5 : X_DIM;

    // ── 第一步: 预计算所有动作共享的静态Z矩阵行 (5行) ──
    // 这些编码对所有候选动作完全相同, 仅需计算1次

    // Row 0: 当前手牌 — 己方持牌 (Botzone Card ID → env_card → 8×15矩阵)
    auto myEnv = handToEnvCards(state.myHandCards);
    auto myHandcards = cards2array_v2(myEnv);

    // Row 1: 对手手牌并集 — 全集减去己方手牌和所有已出牌
    // (推理时等价于未知牌池, 与Python训练端 other_union 对齐)
    auto otherEnv = computeOtherHandCards(state);
    auto otherHandcards = cards2array_v2(otherEnv);

    // Row 2-4: 三人累计已出牌 — 按角色映射到相对位置(自身/上手/下手)
    // 角色映射: 地主位置ldPos=0, 农民下家=1, 农民上家=2
    int myPos = state.myPosition;
    int ldPos = state.landlordPosition;

    // Row 2: 自身累计已出牌
    auto selfPlayedEnv = handToEnvCards(
        (myPos == ldPos) ? state.landlordPlayedCards :
        (myPos == (ldPos + 1) % 3) ? state.landlordDownPlayedCards :
        state.landlordUpPlayedCards);
    auto selfPlayed = cards2array_v2(selfPlayedEnv);

    // Row 3: 上手累计已出牌 (相对位置: 逆时针上一家)
    int upPos = (myPos + 2) % 3;
    auto upPlayedEnv = handToEnvCards(
        (upPos == ldPos) ? state.landlordPlayedCards :
        (upPos == (ldPos + 1) % 3) ? state.landlordDownPlayedCards :
        state.landlordUpPlayedCards);
    auto upPlayed = cards2array_v2(upPlayedEnv);

    // Row 4: 下手累计已出牌 (相对位置: 逆时针下一家)
    int downPos = (myPos + 1) % 3;
    auto downPlayedEnv = handToEnvCards(
        (downPos == ldPos) ? state.landlordPlayedCards :
        (downPos == (ldPos + 1) % 3) ? state.landlordDownPlayedCards :
        state.landlordUpPlayedCards);
    auto downPlayed = cards2array_v2(downPlayedEnv);

    // ── 第二步: 组装Z矩阵 (5静态行 + 35历史行) ──
    // 将5行静态状态和35行历史拼接为40×120连续内存块
    std::vector<float> z(Z_TOTAL_ROWS * CARD_MATRIX_DIM, 0.0f);
    std::memcpy(&z[0 * CARD_MATRIX_DIM], myHandcards.data(), CARD_MATRIX_DIM * sizeof(float));
    std::memcpy(&z[1 * CARD_MATRIX_DIM], otherHandcards.data(), CARD_MATRIX_DIM * sizeof(float));
    std::memcpy(&z[2 * CARD_MATRIX_DIM], selfPlayed.data(), CARD_MATRIX_DIM * sizeof(float));
    std::memcpy(&z[3 * CARD_MATRIX_DIM], upPlayed.data(), CARD_MATRIX_DIM * sizeof(float));
    std::memcpy(&z[4 * CARD_MATRIX_DIM], downPlayed.data(), CARD_MATRIX_DIM * sizeof(float));

    // Row 5-39: 最近35手出牌历史 — 最近的在前面(索引5=最近, 索引39=最远)
    // pass→全零120维, padding→全-1 哨兵值
    auto processedSeq = processActionSeq_typed(state.cardPlayActionSeq, HISTORY_LENGTH);
    auto historyArr = actionSeqList2Array_typed(processedSeq);
    for (int r = 0; r < HISTORY_LENGTH; r++)
        std::memcpy(&z[(Z_STATIC_ROWS + r) * CARD_MATRIX_DIM],
                    historyArr[r].data(),
                    CARD_MATRIX_DIM * sizeof(float));

    // ── 第三步: 组装X向量 (所有动作共享的标量特征) ──
    // X向量不进Transformer, 在Transformer输出后拼接

    std::vector<float> x_noAction;

    // 叫分信息 (3维): 三人叫分值归一化÷3
    for (int i = 0; i < 3; i++)
        x_noAction.push_back(state.bidInfo[i] / 3.0f);

    // 剩余牌数 (3维): self/up/down, 归一化÷20
    // 使用相对顺序: self → up(上手) → down(下手), 与Python训练端一致
    {
        auto getCardsLeft = [&](int absPos) {
            if (absPos == ldPos) return state.landlordCardsLeft;
            if (absPos == (ldPos + 1) % 3) return state.landlordDownCardsLeft;
            return state.landlordUpCardsLeft;
        };
        x_noAction.push_back(getCardsLeft(myPos) / 20.0f);           // 自身
        x_noAction.push_back(getCardsLeft((myPos + 2) % 3) / 20.0f); // 上手
        x_noAction.push_back(getCardsLeft((myPos + 1) % 3) / 20.0f); // 下手
    }

    // 春天标志 (1维): 当前是否仍是春天
    x_noAction.push_back(state.spring ? 1.0f : 0.0f);

    // 已出炸弹数 (15维): one-hot编码, 索引=已出炸弹数(0-14)
    auto bombOneHot = getOneHotBomb(state.bombNum);
    x_noAction.insert(x_noAction.end(), bombOneHot.begin(), bombOneHot.end());

    // C1 自手牌型统计 (7维): 各牌型可出数量/经验上界
    auto handStats = computeHandStats(myEnv);
    x_noAction.insert(x_noAction.end(), handStats.begin(), handStats.end());

    // C2 神谕特征 (5维): n_active/n_passive/delta_n/n_preserve/self_adv
    // 逐动作变化 (评估出牌后剩余手牌). 此处占位填零,
    // 第五步组装 x_batch 时逐动作覆盖为正确的神谕值.
    bool hasOracle = (oracleFeat.size() >= (size_t)B && oracleFeat[0].size() >= X_ORACLE);
    x_noAction.insert(x_noAction.end(), X_ORACLE, 0.0f);  // 占位, 逐动作覆盖

    // C3 对手信念特征 (4维): OHS_up/OHS_down/OBP_up/OBP_down
    if (opponentFeat.size() >= X_OPPONENT)
        x_noAction.insert(x_noAction.end(), opponentFeat.begin(), opponentFeat.begin() + X_OPPONENT);
    else
        x_noAction.insert(x_noAction.end(), X_OPPONENT, 0.0f);

    // C4 对手牌型偏好 (14维): CTP_up[7] + CTP_down[7], EMA衰减维护
    if (ctpFeat.size() >= X_CTP)
        x_noAction.insert(x_noAction.end(), ctpFeat.begin(), ctpFeat.begin() + X_CTP);
    else
        x_noAction.insert(x_noAction.end(), X_CTP, 0.0f);

    // C5 对手行为风格 (4维, 可选): SWI_up/SWI_down + TP_up/TP_down
    if (hasC5)
        x_noAction.insert(x_noAction.end(), c5Feat.begin(), c5Feat.begin() + X_C5);

    int actualXDim = (int)x_noAction.size();

    // ── 第四步: 组装 z_batch — 每个候选动作前置其动作卡牌矩阵 ──
    // V1格式: z_batch = [action_row | Z_matrix] 共41行×120维
    batch.z_batch.resize(B * Z_BATCH_ROWS * CARD_MATRIX_DIM, 0.0f);
    for (int a = 0; a < B; a++) {
        float* base = &batch.z_batch[a * Z_BATCH_ROWS * CARD_MATRIX_DIM];

        // Row 0: 候选动作的卡牌编码 (action = 实际打出的牌→8×15矩阵)
        auto actionEnv = handToEnvCards(legalActions[a]);
        auto actionVec = cards2array_v2(actionEnv);
        std::memcpy(base, actionVec.data(), CARD_MATRIX_DIM * sizeof(float));

        // Row 1-40: 共享的状态Z矩阵 (5静态 + 35历史)
        std::memcpy(base + CARD_MATRIX_DIM, z.data(),
                    Z_TOTAL_ROWS * CARD_MATRIX_DIM * sizeof(float));
    }

    // ── 第五步: 组装 x_batch — 逐动作覆盖神谕特征 ──
    // X 向量布局: bid_info(3) + cards_left(3) + spring(1) + bomb_num(15)
    //             + hand_stats(7) + oracle(5) + opponent(4) + ctp(14) + [c5(4)]
    // oracle 段 (偏移 29) 对每个动作不同, 其余段所有动作共享
    constexpr int ORACLE_OFFSET = X_BID_INFO + X_NUM_CARDS + X_SPRING + X_BOMB + X_HAND_STATS;  // 29

    batch.x_batch.resize(B * actualXDim, 0.0f);
    for (int a = 0; a < B; a++) {
        float* dest = &batch.x_batch[a * actualXDim];
        // 全量复制共享特征 (含 oracle 占位零)
        std::memcpy(dest, x_noAction.data(), actualXDim * sizeof(float));
        // 逐动作覆盖 oracle 段
        if (hasOracle)
            std::memcpy(dest + ORACLE_OFFSET, oracleFeat[a].data(),
                        X_ORACLE * sizeof(float));
    }

    // ── 第六步: V2 Cross-Attention 拆分格式 (预留接口) ──
    // 与V1的区别: state/action/history独立送入各自的Encoder,
    // 再通过Cross-Attention交互 (详见 action_encoder_cross_attention_design.md)

    // state_z: (1, 5, 120) — 5行静态卡牌状态, 所有动作共享, 模型仅编码1次
    batch.state_z.resize(1 * Z_STATIC_ROWS * CARD_MATRIX_DIM);
    std::memcpy(batch.state_z.data(), z.data(),
                Z_STATIC_ROWS * CARD_MATRIX_DIM * sizeof(float));

    // action_z: (B, 1, 120) — 每个候选动作为单独的1-token
    batch.action_z.resize(B * 1 * CARD_MATRIX_DIM, 0.0f);
    for (int a = 0; a < B; a++) {
        auto actionEnv = handToEnvCards(legalActions[a]);
        auto actionVec = cards2array_v2(actionEnv);
        std::memcpy(&batch.action_z[a * CARD_MATRIX_DIM], actionVec.data(),
                    CARD_MATRIX_DIM * sizeof(float));
    }

    // history_z: (1, 35, 120) — 最近35手出牌历史, 所有动作共享, 模型仅编码1次
    batch.history_z.resize(1 * Z_HISTORY_ROWS * CARD_MATRIX_DIM);
    for (int r = 0; r < Z_HISTORY_ROWS; r++)
        std::memcpy(&batch.history_z[r * CARD_MATRIX_DIM], historyArr[r].data(),
                    CARD_MATRIX_DIM * sizeof(float));

    return batch;
}


// ═══════════════════════════════════════════════════════════════════════════
// buildBidFeatures — 组装叫分阶段特征
// ═══════════════════════════════════════════════════════════════════════════
// handCards:     Botzone Card ID 列表 (0-53)
// bidInfo:       三人叫分值 [p0, p1, p2]
// legalActions:  合法叫分动作 [[val], ...], 如 [[0],[1],[2],[3]]

inline BidFeatureBatch buildBidFeatures(
    const std::vector<int>& handCards,
    const std::vector<int>& bidInfo,
    const std::vector<std::vector<int>>& legalActions)
{
    BidFeatureBatch batch;
    int B = (int)legalActions.size();
    batch.batch_size = B;

    // 手牌编码 (120 dim)
    auto handEnv = handToEnvCards(handCards);
    auto handVec = cards2array_v2(handEnv);

    // ── 组装 z_batch: (B, 2, 120) ──
    batch.z_batch.resize(B * 2 * CARD_MATRIX_DIM, 0.0f);
    for (int a = 0; a < B; a++) {
        float* base = &batch.z_batch[a * 2 * CARD_MATRIX_DIM];

        // Row 0: 叫分动作广播 (120 维全为叫分值)
        float val = (float)legalActions[a][0];
        for (int c = 0; c < CARD_MATRIX_DIM; c++)
            base[c] = val;

        // Row 1: 手牌
        std::memcpy(base + CARD_MATRIX_DIM, handVec.data(),
                    CARD_MATRIX_DIM * sizeof(float));
    }

    // ── 组装 x_batch: (B, 3) ──
    batch.x_batch.resize(B * 3, 0.0f);
    for (int a = 0; a < B; a++) {
        batch.x_batch[a * 3 + 0] = (float)bidInfo[0];
        batch.x_batch[a * 3 + 1] = (float)bidInfo[1];
        batch.x_batch[a * 3 + 2] = (float)bidInfo[2];
    }

    return batch;
}


#endif // FEATURE_ENCODER_H
