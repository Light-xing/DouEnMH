// =============================================================================
// game_state.h — 游戏状态管理 & Botzone IO
// =============================================================================
// 职责:
//   GameState    — 斗地主对局的完整状态
//   rebuildFromFullHistory() — 首回合从完整 JSON 历史全量重建 (O(N), <5ms)
//   applyIncrementalUpdate() — 后续回合增量更新 (O(1), <1ms)
//   BotzoneIO    — 输入解析 & 决策输出 (自动分发全量/增量模式)
//
// 两种输入模式:
//   - 首回合: {"requests": [...], "responses": [...]}
//     → rebuildFromFullHistory() 遍历全部历史
//   - 后续回合 (KEEP_RUNNING): {"history": [[...], [...]]}
//     → applyIncrementalUpdate() 原地增量更新
//
// 判断依据: input.isMember("requests") — 有则全量, 无则增量
// =============================================================================

#ifndef GAME_STATE_H
#define GAME_STATE_H

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <unordered_set>
#include <algorithm>
#include "jsoncpp/json.h"
#include "card_encoding.h"

// 游戏阶段
enum class Stage { BIDDING, PLAYING };

// 游戏完整状态 — 所有字段在 KEEP_RUNNING 长时运行模式下跨回合持久化
// 首回合 rebuildFromFullHistory 全量构建, 后续回合 applyIncrementalUpdate 增量更新
struct GameState {
    // ---- 身份信息 ----
    int myPosition = -1;         // 出牌阶段的玩家位置 (0=地主, 1=地主下手, 2=地主上手)
    int landlordPosition = -1;   // 地主位置 (0/1/2), -1=未确定
    int bidPosition = -1;        // 叫分阶段的玩家位置 (由首回合 JSON bid 数组长度确定)
    Stage stage = Stage::BIDDING;

    // ---- 手牌 (Botzone Card ID 0-53) ----
    std::vector<int> myHandCards;

    // ---- 地主底牌 (三张, 公开信息) ----
    std::vector<int> threeLandlordCards;

    // ---- 已出牌 (按角色累计) ----
    std::vector<int> landlordPlayedCards;     // 地主累计已出
    std::vector<int> landlordUpPlayedCards;   // 地主上手(农民)累计已出
    std::vector<int> landlordDownPlayedCards; // 地主下手(农民)累计已出

    // ---- 剩余牌数 (初始: 地主20, 农民各17) ----
    int landlordCardsLeft = 20;
    int landlordUpCardsLeft = 17;
    int landlordDownCardsLeft = 17;

    // ---- 出牌历史序列 ----
    // [(出牌人 position 0/1/2, [Card ID]), ...]
    // 特征编码时 processActionSeq_typed 截取最近 35 手并反转 (最近的在前面)
    std::vector<std::pair<int, std::vector<int>>> cardPlayActionSeq;

    // ---- 炸弹计数 (全局累计, 含火箭) ----
    int bombNum = 0;

    // ---- 春天标记 ----
    // 初始 true; 任意农民出牌→false; 地主出牌>1手→false (不可逆)
    bool spring = true;
    int landlordPlayCount = 0;     // 地主有效出牌次数 (PASS 不计)

    // ---- 叫分信息 ----
    std::vector<int> bidInfo;      // [p0, p1, p2] 三人叫分值, 未叫=-1
    int finalBid = 0;              // 最终底分 (1/2/3)
    std::vector<int> bidInput;     // 已有叫分序列 (用于推算当前 maxBid)

    // ---- 当前需压的牌 (空 = PASS / 新一轮自由出牌) ----
    std::vector<int> lastValidCombo;
};

// ============================================================================
// 辅助函数
// ============================================================================

// 按角色将出牌追加到已出牌累计
inline void addToPlayedCards(GameState& state, int playerPos,
                             const std::vector<int>& cards) {
    if (cards.empty()) return;
    std::vector<int>* target = nullptr;
    if (playerPos == state.landlordPosition)
        target = &state.landlordPlayedCards;
    else if (playerPos == (state.landlordPosition + 1) % 3)
        target = &state.landlordUpPlayedCards;
    else
        target = &state.landlordDownPlayedCards;
    for (int c : cards) target->push_back(c);
}

// 判定出牌是否为炸弹或火箭
inline bool isBombCards(const std::vector<int>& cards) {
    if (cards.empty()) return false;
    // 火箭: 恰好两张且为一王一后 (不依赖顺序)
    if (cards.size() == 2) {
        return (cards[0] == 52 && cards[1] == 53)
            || (cards[0] == 53 && cards[1] == 52);
    }
    // 普通炸弹: 4 张同点数
    if (cards.size() != 4) return false;
    int level = cards[0] / 4;
    for (int c : cards)
        if (c / 4 != level) return false;
    return true;
}

// 更新春天状态 (不可逆: 一旦 false 永不恢复)
inline void updateSpring(GameState& state, int playerPos,
                         const std::vector<int>& cards) {
    if (cards.empty()) return;  // PASS 不影响春天
    if (playerPos == state.landlordPosition) {
        state.landlordPlayCount++;
        if (state.landlordPlayCount > 1) state.spring = false;
    } else {
        state.spring = false;
    }
}

// 出牌后从手牌中删除
inline void removeFromMyHand(GameState& state,
                             const std::vector<int>& playedCards) {
    for (int c : playedCards) {
        auto it = std::find(state.myHandCards.begin(),
                            state.myHandCards.end(), c);
        if (it != state.myHandCards.end())
            state.myHandCards.erase(it);
    }
    // 同步更新剩余牌数
    int& myRemaining = (state.myPosition == state.landlordPosition)
        ? state.landlordCardsLeft
        : (state.myPosition == (state.landlordPosition + 1) % 3
            ? state.landlordUpCardsLeft : state.landlordDownCardsLeft);
    myRemaining -= (int)playedCards.size();
}

// 推算对手手牌并集 (54张全集 - 我手牌 - 全部已出牌) → env_card 列表
// 用于特征编码 Z 矩阵的 other_union 行 (第2行)
inline std::vector<int> computeOtherHandCards(const GameState& state) {
    std::unordered_set<int> knownCards;
    for (int c : state.myHandCards) knownCards.insert(c);
    for (int c : state.landlordPlayedCards) knownCards.insert(c);
    for (int c : state.landlordUpPlayedCards) knownCards.insert(c);
    for (int c : state.landlordDownPlayedCards) knownCards.insert(c);

    std::vector<int> result;
    for (int c = 0; c < 54; c++) {
        if (!knownCards.count(c))
            result.push_back(botzoneCardToEnvCard(c));
    }
    return result;
}

// ============================================================================
// 全量状态重建 — 从 Botzone 完整 JSON 历史重建 GameState
// ============================================================================
// 适用: KEEP_RUNNING 首回合 (input 含 "requests" 键)
// 算法三步:
//   ① 首回合初始化: 获取初始手牌 (firstReq["own"]), 从 bid 数组长度推断 bidPosition
//   ② 逐回合回放: 遍历 requests[0..N-1]
//      - 首次出现 publiccard → 确定地主身份, 更新 position, 导入底牌
//      - history[0..1]: 回放两个对手的出牌 (更新剩余牌数/已出/炸弹/春天)
//      - responses[i]: 回放自己过往出牌/叫分
//   ③ 收尾计算: 确定 stage, 未确定地主时 myPosition = bidPosition
// 复杂度: O(N), N ≤ ~100 手, < 5ms
// ============================================================================

inline GameState rebuildFromFullHistory(const Json::Value& requests,
                                        const Json::Value& responses) {
    GameState state;

    // Step 1: 首回合初始化 — 获取初始手牌和叫分 position
    const Json::Value& firstReq = requests[0u];
    for (unsigned i = 0; i < firstReq["own"].size(); i++)
        state.myHandCards.push_back(firstReq["own"][i].asInt());

    // 从 "bid" 数组长度推断叫分 position
    const Json::Value& bidArr = firstReq["bid"];
    state.bidPosition = bidArr.size();

    // 初始化叫分信息
    state.bidInfo = {-1, -1, -1};
    for (unsigned i = 0; i < bidArr.size(); i++) {
        int bv = bidArr[i].asInt();
        state.bidInfo[i] = bv;
        state.bidInput.push_back(bv);
    }

    state.spring = true;

    // Step 2: 逐回合回放
    // whoInHistory: 上上家和上家的 position (初始基于 bidPosition 推算)
    int whoInHistory[2] = {
        (state.bidPosition - 2 + 3) % 3,
        (state.bidPosition - 1 + 3) % 3
    };

    int numTurns = (int)requests.size();

    for (int i = 0; i < numTurns; i++) {
        const Json::Value& req = requests[i];

        // 首次出现 publiccard → 确定地主身份, 更新 position
        const Json::Value& pub = req["publiccard"];
        if (!pub.isNull()) {
            state.landlordPosition = req["landlord"].asInt();
            state.finalBid = req["finalbid"].asInt();
            state.myPosition = req["pos"].asInt();

            // 基于出牌阶段 position 更新 whoInHistory
            whoInHistory[0] = (state.myPosition - 2 + 3) % 3;
            whoInHistory[1] = (state.myPosition - 1 + 3) % 3;

            // 地主导入底牌
            state.landlordCardsLeft = 20;
            for (unsigned j = 0; j < pub.size(); j++) {
                int c = pub[j].asInt();
                state.threeLandlordCards.push_back(c);
                if (state.landlordPosition == state.myPosition)
                    state.myHandCards.push_back(c);
            }
        }

        // 出牌历史回放 (仅在 playing 阶段)
        const Json::Value& hist = req["history"];
        if (hist.isNull()) continue;  // 叫分阶段跳过

        state.stage = Stage::PLAYING;

        int passCount = 0;
        for (int p = 0; p < 2; p++) {
            int player = whoInHistory[p];
            const Json::Value& action = hist[p];
            std::vector<int> played;
            for (unsigned k = 0; k < action.size(); k++)
                played.push_back(action[k].asInt());

            state.cardPlayActionSeq.push_back({player, played});

            // 更新剩余牌数
            int& remaining = (player == state.landlordPosition)
                ? state.landlordCardsLeft
                : (player == (state.landlordPosition + 1) % 3
                    ? state.landlordUpCardsLeft : state.landlordDownCardsLeft);
            remaining -= (int)played.size();

            addToPlayedCards(state, player, played);

            if (isBombCards(played)) state.bombNum++;
            updateSpring(state, player, played);

            if (played.empty()) passCount++;
            else state.lastValidCombo = played;
        }

        // 两人都 PASS → 新一轮, 清空待压牌
        if (passCount == 2) state.lastValidCombo.clear();

        // 回放自己过往的出牌 (除去最后一回合)
        if (i < (int)responses.size()) {
            const Json::Value& resp = responses[i];
            // 叫分 response 是整数, 出牌 response 是数组
            if (resp.isArray()) {
                std::vector<int> myPlayed;
                for (unsigned k = 0; k < resp.size(); k++) {
                    int c = resp[k].asInt();
                    myPlayed.push_back(c);
                    auto it = std::find(state.myHandCards.begin(),
                                        state.myHandCards.end(), c);
                    if (it != state.myHandCards.end())
                        state.myHandCards.erase(it);
                }

                state.cardPlayActionSeq.push_back({state.myPosition, myPlayed});

                int& myRemaining = (state.myPosition == state.landlordPosition)
                    ? state.landlordCardsLeft
                    : (state.myPosition == (state.landlordPosition + 1) % 3
                        ? state.landlordUpCardsLeft : state.landlordDownCardsLeft);
                myRemaining -= (int)myPlayed.size();

                addToPlayedCards(state, state.myPosition, myPlayed);
                if (isBombCards(myPlayed)) state.bombNum++;
                updateSpring(state, state.myPosition, myPlayed);
            } else {
                // 叫分 response
                int bidValue = resp.asInt();
                state.bidInfo[state.bidPosition] = bidValue;
                state.bidInput.push_back(bidValue);
                if (bidValue > state.finalBid) state.finalBid = bidValue;
            }
        }
    }

    // Step 3: 收尾计算
    if (numTurns > 0) {
        const Json::Value& lastReq = requests[numTurns - 1];
        if (!lastReq["history"].isNull())
            state.stage = Stage::PLAYING;
    }

    // 地主未确定时, myPosition = bidPosition
    if (state.landlordPosition == -1)
        state.myPosition = state.bidPosition;

    return state;
}

// ============================================================================
// 增量状态更新 — KEEP_RUNNING 后续回合专用 (input 不含 "requests" 键)
// ============================================================================
// 仅处理本回合两个对手的最新出牌 (history[0..1]), 原地更新 GameState
// 处理逻辑: 出现 publiccard → 确定地主/切换阶段/补全 bidInfo
//           history 回放 → 更新剩余牌数/已出/炸弹/春天/lastValidCombo
// 复杂度: O(1), < 1ms (仅处理 2 手对手出牌)
// 注: 自己的出牌由 main.cpp 决策后自行写入 cardPlayActionSeq

inline void applyIncrementalUpdate(GameState& state,
                                   const Json::Value& currentRequest) {
    int whoInHistory[2] = {
        (state.myPosition - 2 + 3) % 3,
        (state.myPosition - 1 + 3) % 3
    };

    // 检查是否从叫分切换到出牌 (出现 publiccard)
    const Json::Value& pub = currentRequest["publiccard"];
    if (!pub.isNull()) {
        state.landlordPosition = currentRequest["landlord"].asInt();
        state.finalBid = currentRequest["finalbid"].asInt();
        state.myPosition = currentRequest["pos"].asInt();

        whoInHistory[0] = (state.myPosition - 2 + 3) % 3;
        whoInHistory[1] = (state.myPosition - 1 + 3) % 3;

        state.landlordCardsLeft = 20;
        for (unsigned j = 0; j < pub.size(); j++) {
            int c = pub[j].asInt();
            state.threeLandlordCards.push_back(c);
            if (state.landlordPosition == state.myPosition)
                state.myHandCards.push_back(c);
        }

        // 从 finalBid/landlord 推算缺失的 bidInfo (自己的 bid 已在 main.cpp 中写回)
        if ((int)state.bidInfo.size() < 3) state.bidInfo.resize(3);
        for (int i = 0; i < 3; i++) {
            if (state.bidInfo[i] < 0) {
                // 地主至少叫了 finalBid; 其他玩家保守估计为不叫
                state.bidInfo[i] = (i == state.landlordPosition) ? state.finalBid : 0;
            }
        }
    }

    const Json::Value& hist = currentRequest["history"];
    if (hist.isNull()) return;  // 叫分阶段 (KEEP_RUNNING 下罕见, 防御性处理)

    state.stage = Stage::PLAYING;

    int passCount = 0;
    for (int p = 0; p < 2; p++) {
        int player = whoInHistory[p];
        const Json::Value& action = hist[p];
        std::vector<int> played;
        for (unsigned k = 0; k < action.size(); k++)
            played.push_back(action[k].asInt());

        state.cardPlayActionSeq.push_back({player, played});

        int& remaining = (player == state.landlordPosition)
            ? state.landlordCardsLeft
            : (player == (state.landlordPosition + 1) % 3
                ? state.landlordUpCardsLeft : state.landlordDownCardsLeft);
        remaining -= (int)played.size();

        addToPlayedCards(state, player, played);

        if (isBombCards(played)) state.bombNum++;
        updateSpring(state, player, played);

        if (played.empty()) passCount++;
        else state.lastValidCombo = played;
    }

    if (passCount == 2) state.lastValidCombo.clear();
}

// ============================================================================
// BotzoneIO — 输入解析 & 决策输出 (读写 Botzone JSON 协议)
// ============================================================================
// read(): 阻塞读取 stdin 一行 JSON, 自动判断模式:
//   - 含 "requests" 键 → rebuildFromFullHistory (首回合全量重建)
//   - 不含 "requests" 键 → applyIncrementalUpdate (后续回合增量)
// bid(): 输出 {"response": int}
// play(): 输出 {"response": [card_ids]} (空数组 = PASS)
// ============================================================================

namespace BotzoneIO {

// 读取一行 STDIN, 自动判断全量重建 vs 增量更新
// 返回 false 表示 EOF 或解析错误 (主循环退出)
inline bool read(GameState& state) {
    std::string line;
    if (!std::getline(std::cin, line)) return false;
    if (line.empty()) return false;
    Json::Value input;
    Json::CharReaderBuilder readerBuilder;
    std::string parseErrs;
    std::istringstream iss(line);
    if (!Json::parseFromStream(readerBuilder, iss, &input, &parseErrs))
        return false;

    if (input.isMember("requests")) {
        // 完整历史模式 (首回合)
        const Json::Value& requests = input["requests"];
        const Json::Value& responses = input["responses"];
        state = rebuildFromFullHistory(requests, responses);
    } else {
        // 增量模式 (KEEP_RUNNING 后续回合)
        applyIncrementalUpdate(state, input);
    }
    return true;
}

// 输出叫分决策
inline void bid(int value) {
    Json::Value result;
    result["response"] = value;
    Json::StreamWriterBuilder writerBuilder;
    writerBuilder["indentation"] = "";
    std::cout << Json::writeString(writerBuilder, result) << std::endl;
}

// 输出出牌决策 (空数组 = PASS)
inline void play(const std::vector<int>& cards) {
    Json::Value result, response(Json::arrayValue);
    for (int c : cards)
        response.append(c);
    result["response"] = response;
    Json::StreamWriterBuilder writerBuilder;
    writerBuilder["indentation"] = "";
    std::cout << Json::writeString(writerBuilder, result) << std::endl;
}

} // namespace BotzoneIO

#endif // GAME_STATE_H
