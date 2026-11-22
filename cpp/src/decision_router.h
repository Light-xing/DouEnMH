// ============================================================================
// decision_router.h — DouEnMH 决策路由 & 引擎开关
// ============================================================================
// 两个开关独立控制叫分/出牌阶段:
//
//   USE_HEURISTIC_BID   — 1 = MCTS 启发式叫分, 0 = 神经网络模型叫分
//   USE_HEURISTIC_PLAY  — 1 = MCTS Flat MC 出牌, 0 = 神经网络模型出牌
//
// 切换模式只需修改下面两个值然后重新编译。
// 模型在 main.cpp 中仅当对应开关为 0 时才预加载。
// ============================================================================

#ifndef DECISION_ROUTER_H
#define DECISION_ROUTER_H

#include "game_state.h"
#include "card_defs.h"

// ==================== 唯一切换点 ====================
#define USE_HEURISTIC_BID  1   // 1=启发式叫分, 0=神经网络模型叫分
#define USE_HEURISTIC_PLAY 1   // 1=Flat MC出牌, 0=神经网络模型出牌
// =====================================================

// ==================== MCTS 路径 (始终编译) ====================
#include "mcts_globals.h"     // CardCombo, Hand, 全局变量 (需在 mcts_state.h 之前)
#include "mcts_state.h"       // mctsSyncFromGameState() — 状态适配桥
#include "mcts_bid.h"         // decideBid()
#include "mcts_flat_mc.h"     // flat_mc::prep, flat_mc::pickPlay

// ==================== 模型路径 (仅非全启发式模式时编译) ====================
#if !USE_HEURISTIC_BID || !USE_HEURISTIC_PLAY
#include "model_inference.h"  // 模型加载/推理
#include "feature_encoder.h"  // buildBidFeatures / buildPlayFeatures
#include "card_encoding.h"    // env_card 转换
#include "combo.h"            // Combo, ComboType
#include "move_generator.h"   // MoveGenerator
#include "legal_moves.h"      // LegalMoves
#include "oracle.h"           // oracle 特征 (C2)
#include "opponent_modeling.h" // 对手建模 (C3/C4/C5)
#endif

// ==================== 辅助: 叫分合法动作 ====================

inline std::vector<std::vector<int>> legalBidActions(int maxBid) {
    std::vector<std::vector<int>> actions;
    actions.push_back({0});
    if (maxBid < 1) actions.push_back({1});
    if (maxBid < 2) actions.push_back({2});
    if (maxBid < 3) actions.push_back({3});
    return actions;
}

// ==================== DecisionRouter ====================

struct DecisionRouter {
#if !USE_HEURISTIC_BID || !USE_HEURISTIC_PLAY
    ModelInference inference;

    // ── 模型路径配置 ──
    static constexpr const char* MODEL_DIR = "data/models";

#if !USE_HEURISTIC_BID
    bool bidModelsLoaded = false;

    void loadBidModelIfNeeded(const GameState& gs) {
        if (bidModelsLoaded) return;
        std::string bidName = ModelInference::bidModelName(gs.bidPosition);
        std::string bidPath = ModelInference::modelPath(MODEL_DIR, bidName);
        inference.loadModel(bidName, bidPath);
        bidModelsLoaded = true;
    }
#endif

#if !USE_HEURISTIC_PLAY
    bool playModelsLoaded = false;
    std::string loadedPlayModel;

    void loadPlayModelsIfNeeded(const GameState& gs) {
        if (playModelsLoaded) return;
        // 预判角色并预加载
        std::string preloadName = (gs.bidPosition == 0) ? "landlord" : "landlord_down";
        std::string path = ModelInference::modelPath(MODEL_DIR, preloadName);
        if (inference.loadModel(preloadName, path)) {
            loadedPlayModel = preloadName;
        }
        playModelsLoaded = true;
    }

    void ensurePlayModel(const GameState& gs) {
        const char* needed = ModelInference::playModelName(
            gs.myPosition, gs.landlordPosition);
        if (loadedPlayModel == needed) return;
        inference.unloadModel(loadedPlayModel);
        loadedPlayModel.clear();
        std::string path = ModelInference::modelPath(MODEL_DIR, needed);
        if (inference.loadModel(needed, path)) {
            loadedPlayModel = needed;
        }
    }

    // 模型出牌回退 (模型不可用时 → MCTS)
    std::vector<Card> decidePlayFallback(const GameState& gs) {
        mctsSyncFromGameState(gs);
        flat_mc::prep();
        CardCombo myAction = flat_mc::pickPlay();
        return std::vector<Card>(myAction.cards.begin(), myAction.cards.end());
    }
#endif // !USE_HEURISTIC_PLAY
#endif // !USE_HEURISTIC_BID || !USE_HEURISTIC_PLAY

    // ==================== 叫分路由 ====================
    int decideBid(const GameState& gs) {
#if USE_HEURISTIC_BID
        mctsSyncFromGameState(gs);
        return ::decideBid();
#else
        loadBidModelIfNeeded(gs);
        std::string bidName = ModelInference::bidModelName(gs.bidPosition);
        if (!inference.isLoaded(bidName)) {
            // 模型不可用 → 回退 MCTS 叫分
            mctsSyncFromGameState(gs);
            return ::decideBid();
        }
        int maxBid = 0;
        for (int b : gs.bidInput) maxBid = std::max(maxBid, b);
        auto legalActions = legalBidActions(maxBid);
        auto batch = buildBidFeatures(gs.myHandCards, gs.bidInfo, legalActions);
        int bestIdx = inference.predictBid(bidName, batch.z_batch, batch.x_batch,
                                           batch.batch_size);
        return legalActions[bestIdx][0];
#endif
    }

    // ==================== 出牌路由 ====================
    std::vector<Card> decidePlay(const GameState& gs) {
#if USE_HEURISTIC_PLAY
        mctsSyncFromGameState(gs);
        flat_mc::prep();
        CardCombo myAction = flat_mc::pickPlay();
        return std::vector<Card>(myAction.cards.begin(), myAction.cards.end());
#else
        loadPlayModelsIfNeeded(gs);
        ensurePlayModel(gs);

        std::vector<Card> handCards(gs.myHandCards.begin(),
                                     gs.myHandCards.end());
        Combo lastCombo(gs.lastValidCombo.begin(), gs.lastValidCombo.end());
        LegalMoves legal(handCards, lastCombo);
        auto moves = legal.generate();

        const char* playName = ModelInference::playModelName(
            gs.myPosition, gs.landlordPosition);

        if (!inference.isLoaded(playName)) {
            return decidePlayFallback(gs);
        }

        // 构建 DouEnMH 特征 (含 C2 oracle + C3/C4/C5 对手建模)
        // 1. 神谕特征 (C2)
        CardCount handCounts = {};
        for (int c : handCards) handCounts[card2level(c)]++;
        OracleRaw oracleRaw = compute_oracle_raw(handCounts);

        int upPos = (gs.myPosition + 2) % 3;
        int downPos = (gs.myPosition + 1) % 3;

        auto getLeft = [&](int pos) {
            if (pos == gs.landlordPosition) return gs.landlordCardsLeft;
            if (pos == (gs.landlordPosition + 1) % 3) return gs.landlordDownCardsLeft;
            return gs.landlordUpCardsLeft;
        };
        short oppRemaining;
        if (gs.myPosition == gs.landlordPosition) {
            oppRemaining = std::min(getLeft(upPos), getLeft(downPos));
        } else {
            oppRemaining = getLeft(gs.landlordPosition);
        }
        oracleRaw.self_adv = (float)oracleRaw.n_active -
                             estimate_opponent_n_active(oppRemaining, CardCount{});
        OracleFeat oracleFeat = normalize_oracle(oracleRaw);
        auto oracleVec = oracleFeat.to_vector();

        // 2. 对手建模特征 (C3/C4/C5)
        std::array<int, 15> poolVec;
        int poolTotal = 0;
        auto oppFeat = computeOpponentFeatures(gs, upPos, downPos,
                                                &poolVec, &poolTotal);
        auto oppVec = oppFeat.to_vector();
        auto ctpVec = computeCtpFeatures(gs, upPos, downPos);
        auto c5Vec = computeC5Features(gs, upPos, downPos,
                                        poolVec, poolTotal,
                                        oppFeat.obp_up, oppFeat.obp_down);

        // 3. 组装合法动作 int 列表
        std::vector<std::vector<int>> legalInts;
        legalInts.reserve(moves.size());
        for (auto& m : moves) {
            legalInts.emplace_back(m.begin(), m.end());
        }

        // 4. 构建特征 → 批量推理
        auto batch = buildPlayFeatures(gs, legalInts,
                                        oracleVec, oppVec, ctpVec, c5Vec);
        int bestIdx = inference.predictPlay(playName, batch.z_batch, batch.x_batch,
                                             batch.batch_size);

        return moves[bestIdx];
#endif
    }
};

#endif // DECISION_ROUTER_H
