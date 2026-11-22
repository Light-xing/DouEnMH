// =============================================================================
// main.cpp — DouEnMH Botzone C++ 智能体 统一入口
// =============================================================================
// 基于 DouZero+MultiHead 架构的斗地主 AI (V2: 8×15 120-dim 编码)
//
// 推理管线 (5 阶段):
//   ① JSON 解析 & 游戏状态重建  (game_state.h)
//   ② 合法动作生成             (card_defs.h + combo.h + move_generator.h + legal_moves.h)
//   ③ 特征编码                 (feature_encoder.h + card_encoding.h + oracle + opponent)
//   ④ LibTorch 模型推理        (model_inference.h)
//   ⑤ 决策输出 + 状态更新闭环   (main.cpp self-update)
//
// KEEP_RUNNING 长时运行机制:
//   首回合输出 >>>BOTZONE_REQUEST_KEEP_RUNNING<<< 后进程不退出,
//   回合间被 Botzone 判题端 SIGSTOP/SIGCONT 挂起/唤醒,
//   模型与 GameState 保持加载, 回合间仅做增量状态更新 (applyIncrementalUpdate).
//
// 状态更新闭环 (每手出牌后):
//   removeFromMyHand → cardPlayActionSeq 追加 → addToPlayedCards
//   → bombNum 更新 → spring 检查 → lastValidCombo 更新
// =============================================================================

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cassert>
#include "jsoncpp/json.h"
#include "game_state.h"
#include "card_defs.h"
#include "combo.h"
#include "move_generator.h"
#include "legal_moves.h"
#include "feature_encoder.h"
#include "model_inference.h"
#include "oracle.h"
#include "opponent_modeling.h"

using namespace std;

// 模型路径配置
static const string MODEL_DIR = "data/models";

// 全局持久状态 — KEEP_RUNNING 模式下跨回合存活
static GameState g_state;
static ModelInference g_inference;
static bool g_modelsLoaded = false;
static string g_loadedPlayModel;
static bool g_firstPlayOfEpisode = true;

// 叫分合法动作生成
static vector<vector<int>> legalBidActions(int maxBid) {
    vector<vector<int>> actions;
    actions.push_back({0});  // 不叫
    if (maxBid < 1) actions.push_back({1});
    if (maxBid < 2) actions.push_back({2});
    if (maxBid < 3) actions.push_back({3});
    return actions;
}

// 首回合模型加载: 叫分模型 (按bidPosition) + 预判出牌模型 (默认landlord_down)
// 后续回合通过 ensurePlayModel() 按实际角色动态切换出牌模型
static void loadModelsFirstRound() {
    if (g_modelsLoaded) return;

    string bidName = ModelInference::bidModelName(g_state.bidPosition);
    string bidPath = ModelInference::modelPath(MODEL_DIR, bidName);
    g_inference.loadModel(bidName, bidPath);

    string preloadName = ModelInference::predictPlayModel(g_state.bidPosition);
    string preloadPath = ModelInference::modelPath(MODEL_DIR, preloadName);
    if (g_inference.loadModel(preloadName, preloadPath)) {
        g_loadedPlayModel = preloadName;
    }

    g_modelsLoaded = true;
}

// 确保正确的出牌模型已加载 (按 myPosition 与 landlord 的关系)
// 若当前加载模型与所需不一致, 先卸载旧模型再加载新模型 (节省内存)
static void ensurePlayModel() {
    const char* needed = ModelInference::playModelName(
        g_state.myPosition, g_state.landlordPosition);

    if (g_loadedPlayModel == needed) return;

    g_inference.unloadModel(g_loadedPlayModel);
    g_loadedPlayModel.clear();

    string path = ModelInference::modelPath(MODEL_DIR, needed);
    if (g_inference.loadModel(needed, path)) {
        g_loadedPlayModel = needed;
    }
}

static int decideBidFallback(const GameState& state);

// 模型驱动叫分决策
static int decideBidModel(const GameState& state) {
    string bidName = ModelInference::bidModelName(state.bidPosition);
    if (!g_inference.isLoaded(bidName)) return decideBidFallback(state);

    int maxBid = 0;
    for (int b : state.bidInput) maxBid = max(maxBid, b);
    auto legalActions = legalBidActions(maxBid);

    auto batch = buildBidFeatures(state.myHandCards, state.bidInfo, legalActions);

    int bestIdx = g_inference.predictBid(bidName, batch.z_batch, batch.x_batch,
                                          batch.batch_size);
    return legalActions[bestIdx][0];
}

// 启发式叫分回退 (模型未加载或推理失败时使用)
// 规则: 大小王各值4分, 每张2值2分 → handStrength
//   handStrength≥4 且无人叫 → 叫3; ≥2 且叫分≤1 → +1; ≥6 且叫分≤2 → 叫3
static int decideBidFallback(const GameState& state) {
    int maxBid = 0;
    for (int b : state.bidInput) maxBid = max(maxBid, b);

    int numJokers = 0, numTwos = 0;
    for (int c : state.myHandCards) {
        if (c == 52 || c == 53) numJokers++;
        if (c >= 48 && c <= 51) numTwos++;
    }
    int handStrength = numJokers * 4 + numTwos * 2;

    if (maxBid == 0 && handStrength >= 4) return 3;
    if (maxBid <= 1 && handStrength >= 2) return maxBid + 1;
    if (maxBid <= 2 && handStrength >= 6) return 3;
    return 0;
}

// 模型驱动出牌决策 (主流程)
// 1. 生成当前局面下所有合法动作 (LegalMoves)
// 2. 确保正确角色模型已加载 (landlord/landlord_down/landlord_up)
// 3. 构建完整特征: oracle(C2) + opponent C3(OHS+OBP) + C4(CTP) + C5(SWI+TP)
// 4. buildPlayFeatures 组装 Z 矩阵 (36×120) + X 向量 (B×52/56)
// 5. 批量推理 → argmax Q 值 → 返回最优动作
static vector<Card> decidePlayModel(const vector<Card>& handCards,
                                     const vector<int>& lastValidCombo) {
    // 1. 生成合法动作
    Combo lastCombo(lastValidCombo.begin(), lastValidCombo.end());
    LegalMoves legal(handCards, lastCombo);
    auto moves = legal.generate();
    assert(!moves.empty());

    // 2. 确保正确出牌模型已加载
    ensurePlayModel();
    const char* playName = ModelInference::playModelName(
        g_state.myPosition, g_state.landlordPosition);

    if (!g_inference.isLoaded(playName)) {
        // 启发式回退
        if (lastCombo.is_pass()) {
            return moves[0];
        }
        for (auto& m : moves) {
            if (m.empty()) continue;
            Combo c(m.begin(), m.end());
            if (!c.is_invalid() && lastCombo.can_be_beaten_by(c)) return m;
        }
        return {};
    }

    // 3. 构建特征
    vector<vector<int>> legalInts;
    legalInts.reserve(moves.size());
    for (auto& m : moves) {
        legalInts.emplace_back(m.begin(), m.end());
    }

    int upPos = (g_state.myPosition + 2) % 3;
    int downPos = (g_state.myPosition + 1) % 3;

    // 计算神谕特征 (C2) — 逐动作计算（出牌后剩余手牌）
    // 先对完整手牌做一次 DP 预热 g_memo, 后续每个动作的子状态查询直接 memo 命中
    CardCount handCounts = {};
    for (int c : handCards) handCounts[card2level(c)]++;
    compute_oracle_raw(handCounts);  // 结果丢弃，仅用于填充 DP 缓存

    // self_adv 的对手基准 (对手剩余牌数, 仅取决于状态, 不随动作改变)
    short oppRemaining;
    {
        auto getLeft = [&](int pos) {
            if (pos == g_state.landlordPosition) return g_state.landlordCardsLeft;
            if (pos == (g_state.landlordPosition + 1) % 3) return g_state.landlordDownCardsLeft;
            return g_state.landlordUpCardsLeft;
        };
        if (g_state.myPosition == g_state.landlordPosition) {
            oppRemaining = std::min(getLeft(upPos), getLeft(downPos));
        } else {
            oppRemaining = getLeft(g_state.landlordPosition);
        }
    }

    // 逐动作计算剩余手牌的神谕特征 (DP 缓存命中, O(1) per action)
    vector<vector<float>> perActionOracle;
    perActionOracle.reserve(legalInts.size());
    for (auto& action : legalInts) {
        CardCount actionCounts = {};
        for (int c : action) actionCounts[card2level(c)]++;

        CardCount remaining = handCounts;
        for (int i = 0; i < NUM_LEVELS; i++) remaining[i] -= actionCounts[i];

        OracleRaw raw = compute_oracle_raw(remaining);  // memo 命中, 快速
        raw.self_adv = (float)raw.n_active -
                       estimate_opponent_n_active(oppRemaining, CardCount{});
        perActionOracle.push_back(normalize_oracle(raw).to_vector());
    }

    // ── 对手建模特征 (C3/C4/C5) ──

    // C3: 对手信念特征 (OHS + OBP, 4 dim)
    auto oppFeat = computeOpponentFeatures(g_state, upPos, downPos);
    auto oppVec = oppFeat.to_vector();

    // C4: 对手牌型偏好 (CTP, 14 dim) — 回放出牌历史计算
    auto ctpVec = computeCtpFeatures(g_state, upPos, downPos);

    // C5: 对手行为风格 (SWI + TP, 4 dim) — 内部按时间推进重建牌池
    // (p_can 必须用各动作当刻的快照, 故不再传入当前池)
    auto c5Vec = computeC5Features(g_state, upPos, downPos);

    auto batch = buildPlayFeatures(g_state, legalInts,
                                   perActionOracle, oppVec, ctpVec, c5Vec);

    // 4. 批量推理 → argmax
    int bestIdx = g_inference.predictPlay(playName, batch.z_batch, batch.x_batch,
                                           batch.batch_size);

    return moves[bestIdx];
}

// ============================================================================
// KEEP_RUNNING 长时运行主循环
// ============================================================================
// 主循环:
//   1. BotzoneIO::read() 阻塞读取 JSON → 重建/增量更新 GameState
//   2. 若是叫分阶段 → decideBidModel(), 输出叫分
//   3. 若是出牌阶段 → decidePlayModel(), 输出出牌, 然后状态更新闭环
//   4. 输出 >>>BOTZONE_REQUEST_KEEP_RUNNING<<< 保持进程存活
// 异常安全: 任何阶段异常都输出保守动作 (叫0 / 不出), 不会崩溃
int main() {

    while (true) {
        if (!BotzoneIO::read(g_state)) break;

        if (!g_modelsLoaded) {
            loadModelsFirstRound();
        }

        try {
        if (g_state.stage == Stage::BIDDING) {
            g_firstPlayOfEpisode = true;
            int bidValue = decideBidModel(g_state);
            g_state.bidInfo[g_state.bidPosition] = bidValue;
            g_state.bidInput.push_back(bidValue);
            BotzoneIO::bid(bidValue);

        } else if (g_state.stage == Stage::PLAYING) {
            if (g_firstPlayOfEpisode) {
                dp_clear_cache();
                g_firstPlayOfEpisode = false;
            }
            vector<Card> handCards(g_state.myHandCards.begin(),
                                    g_state.myHandCards.end());
            vector<Card> chosenCards = decidePlayModel(handCards,
                                                        g_state.lastValidCombo);

            vector<int> chosenInts(chosenCards.begin(), chosenCards.end());

            // 状态更新闭环
            removeFromMyHand(g_state, chosenInts);
            g_state.cardPlayActionSeq.push_back({g_state.myPosition, chosenInts});
            addToPlayedCards(g_state, g_state.myPosition, chosenInts);
            if (isBombCards(chosenInts)) g_state.bombNum++;
            updateSpring(g_state, g_state.myPosition, chosenInts);
            if (!chosenInts.empty()) {
                g_state.lastValidCombo = chosenInts;
            }

            BotzoneIO::play(chosenInts);
        }
        } catch (const std::exception&) {
            if (g_state.stage == Stage::BIDDING) {
                BotzoneIO::bid(0);
            } else {
                BotzoneIO::play({});
            }
        } catch (...) {
            if (g_state.stage == Stage::BIDDING) {
                BotzoneIO::bid(0);
            } else {
                BotzoneIO::play({});
            }
        }

        cout << "\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n" << flush;
    }
    return 0;
}
