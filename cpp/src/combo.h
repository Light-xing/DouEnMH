// =============================================================================
// combo.h — 牌型检测 & 大小比较
// =============================================================================
// 支持 Botzone FightTheLandlord2 全部 18 种牌型 (含 PASS / INVALID)
// ComboType 0-14 与 Python 训练端 move_detector 严格对齐, 15-17 为航天飞机扩展
//
// 核心接口:
//   Combo(begin, end) — 从 Card 列表构造, 自动检测牌型 (detect)
//   can_be_beaten_by(b) — this 能否被 b 大过 (Combo::can_be_beaten_by)
//
// 牌型判定优先级:
//   火箭 > 炸弹/航天飞机 (同级比点数) > 普通牌型
//   普通牌型: 同类型 + 同长度(连续序列长度或总牌数) + 主牌点数更高 才能大过
//
// 牌型检测算法 (detect):
//   1. 按点数统计每级张数 → count_of_count[k] = 出现k张的等级种数
//   2. 单一类别 (kinds=1): detect_kind1 — 单张/对子/三条/炸弹/顺子/连对/飞机/航天飞机
//   3. 双类别 (kinds=2): detect_kind2 — 三带一/三带二/四带二/飞机带翼/航天飞机带翼
//   4. 后补检测: detect_serial — 处理含缺口的非标准飞机带翼填充
// =============================================================================

#ifndef COMBO_H
#define COMBO_H

#include "card_defs.h"
#include <algorithm>
#include <vector>

// 牌型枚举 (0-18, 与 Python move_detector.py 一致)
// 注: 15-17 SSHUTTLE (航天飞机) 为非标准 Botzone FightTheLandlord2 扩展牌型
//     oracle DP 搜索空间仅有 0-14, 不包含 SSHUTTLE
enum class ComboType : short {
    PASS           = 0,   // 过
    SINGLE         = 1,   // 单张
    PAIR           = 2,   // 对子
    TRIPLE         = 3,   // 三条
    BOMB           = 4,   // 炸弹 (4 张同点)
    KING_BOMB      = 5,   // 火箭 (大小王)
    TRIPLE_1       = 6,   // 三带一
    TRIPLE_2       = 7,   // 三带二
    SERIAL_SINGLE  = 8,   // 顺子 (≥5 连单)
    SERIAL_PAIR    = 9,   // 双顺 (≥3 连对)
    SERIAL_TRIPLE  = 10,  // 飞机不带翼 (≥2 连三)
    SERIAL_3_1     = 11,  // 飞机带单
    SERIAL_3_2     = 12,  // 飞机带对
    QUAD_2_SINGLE  = 13,  // 四带两单
    QUAD_2_PAIR    = 14,  // 四带两对
    SSHUTTLE       = 15,  // 航天飞机 (≥2 连四)
    SSHUTTLE_2     = 16,  // 航天飞机带单
    SSHUTTLE_4     = 17,  // 航天飞机带对
    INVALID        = 18,  // 非法牌型
};

// 连续牌型最小长度 (与 Python utils.py 一致)
constexpr short MIN_SINGLE_CARDS = 5;   // 顺子最少 5 张
constexpr short MIN_PAIRS        = 3;   // 双顺最少 3 对
constexpr short MIN_TRIPLES      = 2;   // 飞机最少 2 连三

// 一手出牌的牌型结构体
struct Combo {
    std::vector<Card> cards;     // 牌张列表 (Botzone Card ID)
    ComboType type = ComboType::PASS;
    Level     main_level = 0;    // 主牌点数 (比大小用)
    short     length     = 1;    // 连续序列长度 (顺子/飞机/航天飞机用)

    Combo() : type(ComboType::PASS) {}

    // 从 Card 迭代器构造, 自动检测牌型
    template <typename IT>
    Combo(IT begin, IT end) { detect(begin, end); }

    // ---- 快捷牌型判断 ----
    bool is_bomb_or_rocket() const {
        return type == ComboType::BOMB || type == ComboType::KING_BOMB;
    }
    bool is_bomb_type() const {
        return type == ComboType::BOMB || type == ComboType::KING_BOMB
            || type == ComboType::SSHUTTLE || type == ComboType::SSHUTTLE_2
            || type == ComboType::SSHUTTLE_4;
    }
    bool is_pass()    const { return type == ComboType::PASS; }
    bool is_invalid() const { return type == ComboType::INVALID; }

    // 大小比较: this 能否被 b 大过
    bool can_be_beaten_by(const Combo& b) const {
        if (is_invalid() || b.is_invalid()) return false;
        // 火箭最大, 可压任何牌
        if (b.type == ComboType::KING_BOMB) return true;
        // 炸弹/航天飞机: 可压非炸弹; 同为炸弹类型比点数
        if (b.type == ComboType::BOMB || b.type == ComboType::SSHUTTLE
            || b.type == ComboType::SSHUTTLE_2 || b.type == ComboType::SSHUTTLE_4) {
            if (type == ComboType::KING_BOMB) return false;
            if (is_bomb_type())
                return b.main_level > main_level;
            return true;
        }
        // 普通牌型: 同类型 + 同长度 + 主牌点数更高
        return b.type == type && b.cards.size() == cards.size()
            && b.main_level > main_level;
    }

    // 牌型检测 (从 Card 迭代器自动识别牌型)
    template <typename IT>
    void detect(IT begin, IT end) {
        cards.assign(begin, end);
        if (cards.empty()) { type = ComboType::PASS; return; }

        // 按点数统计每级牌张数
        short counts[MAX_LEVEL + 1] = {};
        for (Card c : cards) counts[card2level(c)]++;

        // count_of_count[k] = 出现 k 张的等级种数
        short count_of_count[5] = {};
        std::vector<std::pair<Level, short>> packs;  // (点数, 张数)

        for (Level lv = 0; lv <= MAX_LEVEL; lv++) {
            if (counts[lv]) {
                packs.push_back({lv, counts[lv]});
                if (counts[lv] <= 4) count_of_count[counts[lv]]++;
            }
        }
        // 按张数降序、点数降序排列 (主牌在前)
        std::sort(packs.begin(), packs.end(),
            [](auto& a, auto& b) {
                return a.second != b.second ? a.second > b.second : a.first > b.first;
            });

        main_level = packs[0].first;
        short sz = (short)cards.size();

        // 提取出现的同牌张数类别
        std::vector<short> kinds;
        for (short i = 1; i <= 4; i++)
            if (count_of_count[i]) kinds.push_back(i);

        // 分两大类检测: 单一类别 / 双类别
        if (kinds.size() == 1) {
            detect_kind1(packs, counts, count_of_count, kinds[0]);
        } else if (kinds.size() == 2) {
            detect_kind2(packs, count_of_count, kinds);
        }

        // 仍未匹配 → 尝试连续牌型后补检测 (含缺口飞机)
        if (type == ComboType::INVALID && sz >= 5) {
            detect_serial(packs, count_of_count);
        }
    }

private:
    // 计算 packs 中前部同 count 的最大连续长度
    static short find_max_seq(const std::vector<std::pair<Level, short>>& packs,
                               short count) {
        short seq = 1;
        for (size_t i = 1; i < packs.size(); i++) {
            if (packs[i].second != count) break;
            if (packs[i].first != packs[i - 1].first - 1) break;
            seq++;
        }
        return seq;
    }

    // 单一类别检测 (全是单张 / 全是对子 / 全是三条 / 全是四条)
    void detect_kind1(const std::vector<std::pair<Level, short>>& packs,
                      const short counts[],
                      const short count_of_count[5], short kind) {
        short cnt = count_of_count[kind];
        switch (kind) {
        case 1:  // 全是单张
            if (cnt == 1) { type = ComboType::SINGLE; return; }
            if (cnt == 2 && counts[LEVEL_JOKER_SMALL] && counts[LEVEL_JOKER_BIG])
                { type = ComboType::KING_BOMB; return; }
            if (cnt >= MIN_SINGLE_CARDS && find_max_seq(packs, 1) == cnt
                && packs[0].first <= MAX_STRAIGHT_LEVEL)
                { type = ComboType::SERIAL_SINGLE; length = cnt; return; }
            break;
        case 2:  // 全是对子
            if (cnt == 1) { type = ComboType::PAIR; return; }
            if (cnt >= MIN_PAIRS && find_max_seq(packs, 2) == cnt
                && packs[0].first <= MAX_STRAIGHT_LEVEL)
                { type = ComboType::SERIAL_PAIR; length = cnt; return; }
            break;
        case 3:  // 全是三条
            if (cnt == 1) { type = ComboType::TRIPLE; return; }
            if (cnt >= MIN_TRIPLES && find_max_seq(packs, 3) == cnt
                && packs[0].first <= MAX_STRAIGHT_LEVEL)
                { type = ComboType::SERIAL_TRIPLE; length = cnt; return; }
            break;
        case 4:  // 全是四条
            if (cnt == 1) { type = ComboType::BOMB; return; }
            if (cnt >= 2 && find_max_seq(packs, 4) == cnt
                && packs[0].first <= MAX_STRAIGHT_LEVEL)
                { type = ComboType::SSHUTTLE; length = cnt; return; }
            break;
        }
        type = ComboType::INVALID;
    }

    // 双类别检测 (三带一/二、四带二、飞机带翼、航天飞机带翼)
    void detect_kind2(const std::vector<std::pair<Level, short>>& packs,
                      const short count_of_count[5],
                      const std::vector<short>& kinds) {
        short hi  = kinds[1];  // 较多的同牌数 (主牌)
        short lo  = kinds[0];  // 较少的同牌数 (带牌)
        short cnt_hi = count_of_count[hi];
        short cnt_lo = count_of_count[lo];

        if (hi == 3) {
            if (lo == 1) {
                if (cnt_hi == 1 && cnt_lo == 1) { type = ComboType::TRIPLE_1; return; }
                if (cnt_hi >= MIN_TRIPLES && cnt_lo == cnt_hi
                    && find_max_seq(packs, 3) == cnt_hi
                    && packs[0].first <= MAX_STRAIGHT_LEVEL)
                    { type = ComboType::SERIAL_3_1; length = cnt_hi; return; }
            }
            if (lo == 2) {
                if (cnt_hi == 1 && cnt_lo == 1) { type = ComboType::TRIPLE_2; return; }
                if (cnt_hi >= MIN_TRIPLES && cnt_lo == cnt_hi
                    && find_max_seq(packs, 3) == cnt_hi
                    && packs[0].first <= MAX_STRAIGHT_LEVEL)
                    { type = ComboType::SERIAL_3_2; length = cnt_hi; return; }
            }
        }
        if (hi == 4) {
            if (lo == 1) {
                if (cnt_hi == 1 && cnt_lo == 2) { type = ComboType::QUAD_2_SINGLE; return; }
                if (cnt_hi >= 2 && cnt_lo == cnt_hi * 2
                    && find_max_seq(packs, 4) == cnt_hi
                    && packs[0].first <= MAX_STRAIGHT_LEVEL)
                    { type = ComboType::SSHUTTLE_2; length = cnt_hi; return; }
            }
            if (lo == 2) {
                if (cnt_hi == 1 && cnt_lo == 2) { type = ComboType::QUAD_2_PAIR; return; }
                if (cnt_hi >= 2 && cnt_lo == cnt_hi * 2
                    && find_max_seq(packs, 4) == cnt_hi
                    && packs[0].first <= MAX_STRAIGHT_LEVEL)
                    { type = ComboType::SSHUTTLE_4; length = cnt_hi; return; }
            }
        }
        type = ComboType::INVALID;
    }

    // 连续牌型后补检测 (处理含缺口或非标准填充的飞机带翼)
    void detect_serial(const std::vector<std::pair<Level, short>>& packs,
                       const short count_of_count[5]) {
        if (count_of_count[3] >= MIN_TRIPLES) {
            // 提取所有三条的点数, 降序排列
            std::vector<Level> triples;
            for (auto& p : packs) if (p.second == 3) triples.push_back(p.first);
            std::sort(triples.begin(), triples.end(), std::greater<Level>());

            short num_singles = count_of_count[1];
            short num_pairs   = count_of_count[2];

            // 检查是否全连续
            auto is_cont = [](const std::vector<Level>& v) {
                for (size_t i = 1; i < v.size(); i++)
                    if (v[i - 1] - v[i] != 1) return false;
                return true;
            };

            if (is_cont(triples) && triples[0] <= MAX_STRAIGHT_LEVEL) {
                short n3 = (short)triples.size();
                if (n3 == num_singles + num_pairs * 2 && num_singles >= 0) {
                    type = ComboType::SERIAL_3_1; length = n3; main_level = triples[0];
                    return;
                }
                if (n3 == num_pairs && (short)packs.size() == n3 * 2) {
                    type = ComboType::SERIAL_3_2; length = n3; main_level = triples[0];
                    return;
                }
            }
            // 四个连三中存在一个缺口: 尝试取前三个或后三个
            if ((short)triples.size() == 4) {
                std::vector<Level> sub(triples.begin(), triples.end() - 1);
                if (is_cont(sub) && sub[0] <= MAX_STRAIGHT_LEVEL) {
                    type = ComboType::SERIAL_3_1; length = 3; main_level = sub[0];
                    return;
                }
                sub.assign(triples.begin() + 1, triples.end());
                if (is_cont(sub) && sub[0] <= MAX_STRAIGHT_LEVEL) {
                    type = ComboType::SERIAL_3_1; length = 3; main_level = sub[0];
                    return;
                }
            }
        }
        type = ComboType::INVALID;
    }
};

#endif // COMBO_H
