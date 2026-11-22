// =============================================================================
// move_generator.h — 从手牌生成所有可能的出牌组合
// =============================================================================
// 枚举全部 17 种牌型 (含 SSHUTTLE 航天飞机类型 15-17), 复合牌型遍历全带牌组合
//
// 核心设计:
//   - 按点数 (Level) 分组存储 Card ID (cards_by_level_[15]), 单次构造
//   - 连续牌型: find_seqs() 检测连续可用段 → 枚举子序列 → materialize() 提取 Card
//   - 带牌组合: gen_kicker_combos() 栈式迭代枚举 C(n,k) 带牌等级组合 (无递归)
//   - 压牌模式: repeat_num≠0 时只生成指定长度的连续序列, 大幅减少候选数
// =============================================================================

#ifndef MOVE_GENERATOR_H
#define MOVE_GENERATOR_H

#include "card_defs.h"
#include "combo.h"
#include <algorithm>
#include <vector>

class MoveGenerator {
public:
    explicit MoveGenerator(const std::vector<Card>& hand) {
        cnt_ = {};
        for (short lv = 0; lv < NUM_LEVELS; lv++)
            cards_by_level_[lv].clear();

        for (Card c : hand) {
            Level lv = card2level(c);
            cnt_[lv]++;
            cards_by_level_[lv].push_back(c);
        }
        // 每组内排序保证确定性
        for (short lv = 0; lv < NUM_LEVELS; lv++)
            std::sort(cards_by_level_[lv].begin(), cards_by_level_[lv].end());
    }

    // ---- 各牌型生成器 (返回 Card ID 向量列表) ----

    // 单张: 每种有点数的牌取一张
    std::vector<std::vector<Card>> gen_type_1_single() {
        std::vector<std::vector<Card>> res;
        for (Level lv = 0; lv < NUM_LEVELS; lv++)
            if (cnt_[lv] >= 1)
                res.push_back({cards_by_level_[lv][0]});
        return res;
    }

    // 对子: 每种 ≥2 张的点数取两张
    std::vector<std::vector<Card>> gen_type_2_pair() {
        std::vector<std::vector<Card>> res;
        for (Level lv = 0; lv <= MAX_LEVEL; lv++)
            if (cnt_[lv] >= 2)
                res.push_back({cards_by_level_[lv][0], cards_by_level_[lv][1]});
        return res;
    }

    // 三条: 每种 ≥3 张的点数取三张
    std::vector<std::vector<Card>> gen_type_3_triple() {
        std::vector<std::vector<Card>> res;
        for (Level lv = 0; lv <= MAX_LEVEL; lv++)
            if (cnt_[lv] >= 3)
                res.push_back({cards_by_level_[lv][0],
                               cards_by_level_[lv][1],
                               cards_by_level_[lv][2]});
        return res;
    }

    // 炸弹: 每种恰好 4 张的点数 (不含王)
    std::vector<std::vector<Card>> gen_type_4_bomb() {
        std::vector<std::vector<Card>> res;
        for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++)
            if (cnt_[lv] == 4)
                res.push_back({cards_by_level_[lv][0],
                               cards_by_level_[lv][1],
                               cards_by_level_[lv][2],
                               cards_by_level_[lv][3]});
        return res;
    }

    // 火箭: 同时有大小王
    std::vector<std::vector<Card>> gen_type_5_king_bomb() {
        std::vector<std::vector<Card>> res;
        if (cnt_[LEVEL_JOKER_SMALL] >= 1 && cnt_[LEVEL_JOKER_BIG] >= 1)
            res.push_back({CARD_JOKER_SMALL, CARD_JOKER_BIG});
        return res;
    }

    // 三带一: 每种三条 + 所有可用单张
    std::vector<std::vector<Card>> gen_type_6_3_1() {
        std::vector<std::vector<Card>> res;
        for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++) {
            if (cnt_[lv] < 3) continue;
            for (Level k = 0; k < NUM_LEVELS; k++) {
                if (k == lv || cnt_[k] < 1) continue;
                CardCount used = {};
                used[lv] = 3; used[k] = 1;
                auto cards = materialize(used);
                if (!cards.empty()) res.push_back(cards);
            }
        }
        return res;
    }

    // 三带二: 每种三条 + 所有可用对子
    std::vector<std::vector<Card>> gen_type_7_3_2() {
        std::vector<std::vector<Card>> res;
        for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++) {
            if (cnt_[lv] < 3) continue;
            for (Level k = 0; k < NUM_LEVELS; k++) {
                if (k == lv || cnt_[k] < 2) continue;
                CardCount used = {};
                used[lv] = 3; used[k] = 2;
                auto cards = materialize(used);
                if (!cards.empty()) res.push_back(cards);
            }
        }
        return res;
    }

    // 顺子 (≥5 连单)
    std::vector<std::vector<Card>> gen_type_8_serial_single(short repeat_num = 0) {
        return gen_serial(1, MIN_SINGLE_CARDS, repeat_num);
    }

    // 双顺 (≥3 连对)
    std::vector<std::vector<Card>> gen_type_9_serial_pair(short repeat_num = 0) {
        return gen_serial(2, MIN_PAIRS, repeat_num);
    }

    // 飞机不带翼 (≥2 连三)
    std::vector<std::vector<Card>> gen_type_10_serial_triple(short repeat_num = 0) {
        return gen_serial(3, MIN_TRIPLES, repeat_num);
    }

    // 飞机带单 (≥2 连三 + 等量单张)
    std::vector<std::vector<Card>> gen_type_11_serial_3_1(short repeat_num = 0) {
        return gen_serial_with_kicker(3, 1, MIN_TRIPLES, repeat_num);
    }

    // 飞机带对 (≥2 连三 + 等量对子)
    std::vector<std::vector<Card>> gen_type_12_serial_3_2(short repeat_num = 0) {
        return gen_serial_with_kicker(3, 2, MIN_TRIPLES, repeat_num);
    }

    // 四带两单: 遍历全部 C(n,2) 单张组合
    std::vector<std::vector<Card>> gen_type_13_4_2() {
        std::vector<std::vector<Card>> res;
        for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++) {
            if (cnt_[lv] < 4) continue;
            CardCount base_used = {};
            base_used[lv] = 4;
            gen_kicker_combos(base_used, 2, 1, res);
        }
        return res;
    }

    // 四带两对: 遍历全部 C(m,2) 对子组合
    std::vector<std::vector<Card>> gen_type_14_4_22() {
        std::vector<std::vector<Card>> res;
        for (Level lv = 0; lv < LEVEL_JOKER_SMALL; lv++) {
            if (cnt_[lv] < 4) continue;
            CardCount base_used = {};
            base_used[lv] = 4;
            gen_kicker_combos(base_used, 2, 2, res);
        }
        return res;
    }

    // 航天飞机不带翼 (≥2 连四)
    std::vector<std::vector<Card>> gen_type_15_sshuttle() {
        return gen_serial(4, 2, 0);
    }
    // 航天飞机带单
    std::vector<std::vector<Card>> gen_type_16_sshuttle_2() {
        return gen_serial_with_kicker(4, 1, 2, 0);
    }
    // 航天飞机带对
    std::vector<std::vector<Card>> gen_type_17_sshuttle_4() {
        return gen_serial_with_kicker(4, 2, 2, 0);
    }

    // 全量生成: 自由出牌时所有 17 种牌型的并集
    std::vector<std::vector<Card>> gen_all_moves() {
        std::vector<std::vector<Card>> all;
        auto add = [&](auto&& v) { all.insert(all.end(), v.begin(), v.end()); };
        add(gen_type_1_single());
        add(gen_type_2_pair());
        add(gen_type_3_triple());
        add(gen_type_4_bomb());
        add(gen_type_5_king_bomb());
        add(gen_type_6_3_1());
        add(gen_type_7_3_2());
        add(gen_type_8_serial_single());
        add(gen_type_9_serial_pair());
        add(gen_type_10_serial_triple());
        add(gen_type_11_serial_3_1());
        add(gen_type_12_serial_3_2());
        add(gen_type_13_4_2());
        add(gen_type_14_4_22());
        add(gen_type_15_sshuttle());
        add(gen_type_16_sshuttle_2());
        add(gen_type_17_sshuttle_4());
        return all;
    }

private:
    CardCount cnt_;                                  // 手牌按点数计数
    std::vector<Card> cards_by_level_[NUM_LEVELS];   // 实际 Card ID 按点数分组

    // 按 used 计数从手牌中选取 Card ID
    std::vector<Card> materialize(const CardCount& used) const {
        std::vector<Card> res;
        for (Level lv = 0; lv < NUM_LEVELS; lv++) {
            short need = used[lv];
            if (need == 0) continue;
            for (Card c : cards_by_level_[lv]) {
                if (need == 0) break;
                res.push_back(c);
                need--;
            }
            if (need > 0) return {};  // 牌不够 (不应发生)
        }
        return res;
    }

    // 栈式迭代组合生成: 从剩余牌中选 needed 组带牌 (每组 kicker_repeat 张)
    void gen_kicker_combos(const CardCount& base_used, short needed,
                           short kicker_repeat,
                           std::vector<std::vector<Card>>& res) {
        if (needed <= 0) return;

        // 收集候选等级
        std::vector<Level> candidates;
        for (Level k = 0; k < NUM_LEVELS; k++)
            if (cnt_[k] - base_used[k] >= kicker_repeat)
                candidates.push_back(k);

        if ((short)candidates.size() < needed) return;

        // 栈式组合枚举: combo[i] 是 candidates 的下标
        std::vector<short> combo(needed);
        for (short i = 0; i < needed; i++) combo[i] = i;
        short maxIdx = (short)candidates.size() - 1;

        while (true) {
            CardCount used = base_used;
            for (short i = 0; i < needed; i++)
                used[candidates[combo[i]]] += kicker_repeat;
            auto cards = materialize(used);
            if (!cards.empty()) res.push_back(cards);

            // 推进到下一组合
            short idx = needed - 1;
            while (idx >= 0 && combo[idx] == maxIdx - (needed - 1 - idx))
                idx--;
            if (idx < 0) break;
            combo[idx]++;
            for (short j = idx + 1; j < needed; j++)
                combo[j] = combo[j - 1] + 1;
        }
    }

    // 连续区间检测: 在 [0, MAX_STRAIGHT_LEVEL] 内找连续可用段
    // avail[lv] = true 表示该点数有足够张数
    // 返回: [(起始点数, 长度), ...]
    static std::vector<std::pair<Level, short>>
    find_seqs(const bool* avail, short min_len) {
        std::vector<std::pair<Level, short>> segs;
        short run = 0, start = 0;
        for (Level lv = 0; lv <= MAX_STRAIGHT_LEVEL + 1; lv++) {
            if (lv <= MAX_STRAIGHT_LEVEL && avail[lv]) {
                if (run == 0) start = lv;
                run++;
            } else {
                if (run >= min_len) segs.push_back({start, run});
                run = 0;
            }
        }
        return segs;
    }

    // 连续牌型生成 (顺子/双顺/飞机不带翼/航天飞机不带翼)
    std::vector<std::vector<Card>> gen_serial(short repeat, short min_len,
                                               short repeat_num) {
        std::vector<std::vector<Card>> res;
        bool avail[MAX_STRAIGHT_LEVEL + 1] = {};
        for (Level lv = 0; lv <= MAX_STRAIGHT_LEVEL; lv++)
            avail[lv] = (cnt_[lv] >= repeat);

        auto segs = find_seqs(avail, min_len);
        for (auto& seg : segs) {
            Level seg_start = seg.first;
            short max_len = seg.second;

            if (repeat_num == 0) {
                // 自由出牌: 枚举所有长度 ≥ min_len 的子序列
                for (short len = min_len; len <= max_len; len++)
                    for (short offset = 0; offset + len <= max_len; offset++) {
                        CardCount used = {};
                        for (short i = 0; i < len; i++)
                            used[seg_start + offset + i] = repeat;
                        auto cards = materialize(used);
                        if (!cards.empty()) res.push_back(cards);
                    }
            } else {
                // 压牌: 只生成指定长度的序列
                if (repeat_num <= max_len)
                    for (short offset = 0; offset + repeat_num <= max_len; offset++) {
                        CardCount used = {};
                        for (short i = 0; i < repeat_num; i++)
                            used[seg_start + offset + i] = repeat;
                        auto cards = materialize(used);
                        if (!cards.empty()) res.push_back(cards);
                    }
            }
        }
        return res;
    }

    // 带翼连续牌型生成 (飞机带单/对, 航天飞机带单/对)
    std::vector<std::vector<Card>> gen_serial_with_kicker(
        short repeat, short kicker_repeat, short min_len, short repeat_num) {

        std::vector<std::vector<Card>> res;
        bool avail[MAX_STRAIGHT_LEVEL + 1] = {};
        for (Level lv = 0; lv <= MAX_STRAIGHT_LEVEL; lv++)
            avail[lv] = (cnt_[lv] >= repeat);

        auto segs = find_seqs(avail, min_len);
        for (auto& seg : segs) {
            Level seg_start = seg.first;
            short max_len = seg.second;

            auto try_len = [&](short offset, short len) {
                CardCount base_used = {};
                Level s = seg_start + offset;
                for (short i = 0; i < len; i++) base_used[s + i] = repeat;
                gen_kicker_combos(base_used, len, kicker_repeat, res);
            };

            if (repeat_num == 0) {
                for (short len = min_len; len <= max_len; len++)
                    for (short offset = 0; offset + len <= max_len; offset++)
                        try_len(offset, len);
            } else {
                if (repeat_num <= max_len)
                    for (short offset = 0; offset + repeat_num <= max_len; offset++)
                        try_len(offset, repeat_num);
            }
        }
        return res;
    }
};

#endif // MOVE_GENERATOR_H
