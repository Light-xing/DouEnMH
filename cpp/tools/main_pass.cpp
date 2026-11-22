// =============================================================================
// main_pass.cpp — 简易基准 Bot (无 LibTorch 依赖, 仅用于语法/逻辑验证)
// =============================================================================
// 策略: 叫分=0 (不叫), 自由出牌时出最小单张, 压牌时=PASS

#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include "jsoncpp/json.h"

using namespace std;

int main() {
    bool firstRound = true;
    bool isPlaying = false;
    vector<int> myHand;

    Json::CharReaderBuilder readerBuilder;
    string parseErrs;

    Json::StreamWriterBuilder writerBuilder;
    writerBuilder["indentation"] = "";

    while (true) {
        string line;
        if (!getline(cin, line) || line.empty()) break;

        Json::Value input;
        istringstream iss(line);
        if (!Json::parseFromStream(readerBuilder, iss, &input, &parseErrs))
            break;

        // 首回合: 从完整 JSON 历史重建手牌状态
        if (firstRound && input.isMember("requests")) {
            const Json::Value& requests = input["requests"];
            const Json::Value& responses = input["responses"];

            const Json::Value& firstReq = requests[0U];
            for (const auto& c : firstReq["own"])
                myHand.push_back(c.asInt());
            sort(myHand.begin(), myHand.end());

            int myPos = -1, landlordPos = -1;
            for (unsigned i = 0; i < requests.size(); i++) {
                const Json::Value& pub = requests[i]["publiccard"];
                if (!pub.isNull()) {
                    landlordPos = requests[i]["landlord"].asInt();
                    myPos = requests[i]["pos"].asInt();
                    if (myPos == landlordPos) {
                        for (const auto& c : pub) myHand.push_back(c.asInt());
                        sort(myHand.begin(), myHand.end());
                    }
                    break;
                }
            }

            // 减去已出的牌
            for (unsigned i = 0; i < responses.size(); i++) {
                const Json::Value& resp = responses[i];
                if (!resp.isArray()) continue;
                for (const auto& c : resp) {
                    int card = c.asInt();
                    auto it = find(myHand.begin(), myHand.end(), card);
                    if (it != myHand.end()) myHand.erase(it);
                }
            }

            if (requests.size() > 0) {
                const Json::Value& lastReq = requests[requests.size() - 1];
                isPlaying = !lastReq["history"].isNull();
            }
            firstRound = false;
        } else {
            isPlaying = !input["history"].isNull();
        }

        // 决策输出
        Json::Value result;
        if (isPlaying) {
            const Json::Value& history = input.isMember("requests")
                ? input["requests"][input["requests"].size() - 1]["history"]
                : input["history"];

            bool bothPassed = history[0U].empty() && history[1U].empty();

            if (bothPassed && !myHand.empty()) {
                // 自由出牌: 出最小单张
                int card = myHand[0];
                myHand.erase(myHand.begin());
                Json::Value arr(Json::arrayValue);
                arr.append(card);
                result["response"] = arr;
            } else {
                // 压牌: 直接 PASS
                result["response"] = Json::Value(Json::arrayValue);
            }
        } else {
            // 叫分: 不叫
            result["response"] = 0;
        }

        cout << Json::writeString(writerBuilder, result) << endl;
        cout << "\n>>>BOTZONE_REQUEST_KEEP_RUNNING<<<\n" << flush;
    }
    return 0;
}
