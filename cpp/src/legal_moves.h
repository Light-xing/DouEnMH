// =============================================================================
// legal_moves.h — 合法动作过滤 (给定手牌 + 需要压过的上一手牌)
// =============================================================================
//
// 核心机制 (generate):
//   1. 自由出牌 (last_combo 为 PASS): 输出 MoveGenerator::gen_all_moves() 全部 17 种牌型
//   2. 必须压牌: switch lastCombo.type → 路由到 MoveGenerator 对应压牌方法
//      (repeat_num = lastCombo.length, 只生成同长度连续牌型)
//   3. 过滤: filter_by_rank / filter_serial / filter_serial_kicker 按主牌点数>last_level 保留
//   4. 自动追加: 炸弹 + 火箭 (可压任何非炸弹牌型) + PASS (自选不出)
// =============================================================================

#ifndef LEGAL_MOVES_H
#define LEGAL_MOVES_H

#include "card_defs.h"
#include "combo.h"
#include "move_generator.h"
#include <algorithm>
#include <vector>

class LegalMoves {
public:
    LegalMoves(const std::vector<Card>& hand,
               const Combo& last_move = Combo())
        : last_(last_move), mg_(hand) {}

    // 生成所有合法动作
    std::vector<std::vector<Card>> generate() {
        std::vector<std::vector<Card>> moves;

        if (last_.is_pass()) {
            // 自由出牌: 全部牌型均可出
            moves = mg_.gen_all_moves();
        } else {
            // 必须大过 lastCombo
            switch (last_.type) {
            case ComboType::SINGLE:
                moves = filter_by_rank(mg_.gen_type_1_single(), last_.main_level);
                break;
            case ComboType::PAIR:
                moves = filter_by_rank(mg_.gen_type_2_pair(), last_.main_level);
                break;
            case ComboType::TRIPLE:
                moves = filter_by_rank(mg_.gen_type_3_triple(), last_.main_level);
                break;
            case ComboType::BOMB:
                {
                    auto bombs = mg_.gen_type_4_bomb();
                    auto kings = mg_.gen_type_5_king_bomb();
                    moves = filter_by_rank(bombs, last_.main_level);
                    moves.insert(moves.end(), kings.begin(), kings.end());
                }
                break;
            case ComboType::KING_BOMB:
                break;  // 火箭最大, 无牌可压
            case ComboType::TRIPLE_1:
                moves = filter_triple_kicker(mg_.gen_type_6_3_1(), 3);
                break;
            case ComboType::TRIPLE_2:
                moves = filter_triple_kicker(mg_.gen_type_7_3_2(), 3);
                break;
            case ComboType::SERIAL_SINGLE:
                moves = filter_serial(mg_.gen_type_8_serial_single(last_.length),
                                      last_.main_level);
                break;
            case ComboType::SERIAL_PAIR:
                moves = filter_serial(mg_.gen_type_9_serial_pair(last_.length),
                                      last_.main_level);
                break;
            case ComboType::SERIAL_TRIPLE:
                moves = filter_serial(mg_.gen_type_10_serial_triple(last_.length),
                                      last_.main_level);
                break;
            case ComboType::SERIAL_3_1:
                moves = filter_serial_kicker(
                    mg_.gen_type_11_serial_3_1(last_.length), 3, last_.main_level);
                break;
            case ComboType::SERIAL_3_2:
                moves = filter_serial_kicker(
                    mg_.gen_type_12_serial_3_2(last_.length), 3, last_.main_level);
                break;
            case ComboType::QUAD_2_SINGLE:
                moves = filter_quad_kicker(mg_.gen_type_13_4_2(), 4);
                break;
            case ComboType::QUAD_2_PAIR:
                moves = filter_quad_kicker(mg_.gen_type_14_4_22(), 4);
                break;
            case ComboType::SSHUTTLE:
                moves = filter_serial(mg_.gen_type_15_sshuttle(), last_.main_level);
                break;
            case ComboType::SSHUTTLE_2:
                moves = filter_serial_kicker(
                    mg_.gen_type_16_sshuttle_2(), 4, last_.main_level);
                break;
            case ComboType::SSHUTTLE_4:
                moves = filter_serial_kicker(
                    mg_.gen_type_17_sshuttle_4(), 4, last_.main_level);
                break;
            default:
                break;
            }

            // 追加炸弹/火箭: 炸弹可压任何非炸弹牌型
            if (!last_.is_bomb_or_rocket()) {
                auto bombs = mg_.gen_type_4_bomb();
                auto kings = mg_.gen_type_5_king_bomb();
                moves.insert(moves.end(), bombs.begin(), bombs.end());
                moves.insert(moves.end(), kings.begin(), kings.end());
            }

            // PASS 总是合法 (只要上一手非 PASS)
            moves.push_back({});
        }

        // 组内排序保证确定性
        for (auto& m : moves) std::sort(m.begin(), m.end());
        return moves;
    }

private:
    Combo last_;
    MoveGenerator mg_;

    // 按主牌点数过滤 (单张/对子/三条/炸弹/顺子/双顺/飞机)
    static std::vector<std::vector<Card>>
    filter_by_rank(const std::vector<std::vector<Card>>& moves, Level min_level) {
        std::vector<std::vector<Card>> out;
        for (auto& m : moves) {
            Combo c(m.begin(), m.end());
            if (!c.is_invalid() && c.main_level > min_level)
                out.push_back(m);
        }
        return out;
    }

    // 三带一/二: 找出主牌(三条)点数并比较
    std::vector<std::vector<Card>>
    filter_triple_kicker(const std::vector<std::vector<Card>>& moves, short trip_count) {
        std::vector<std::vector<Card>> out;
        for (auto& m : moves) {
            Combo c(m.begin(), m.end());
            if (c.is_invalid()) continue;
            Level my_main = 0;
            short cnts[MAX_LEVEL + 1] = {};
            for (Card card : m) cnts[card2level(card)]++;
            for (Level lv = 0; lv < NUM_LEVELS; lv++)
                if (cnts[lv] == trip_count) my_main = lv;
            if (my_main > last_.main_level) out.push_back(m);
        }
        return out;
    }

    // 连续牌型 (顺子/双顺/飞机/航天飞机): 比较主牌点数
    static std::vector<std::vector<Card>>
    filter_serial(const std::vector<std::vector<Card>>& moves, Level min_main) {
        std::vector<std::vector<Card>> out;
        for (auto& m : moves) {
            Combo c(m.begin(), m.end());
            if (!c.is_invalid() && c.main_level > min_main)
                out.push_back(m);
        }
        return out;
    }

    // 带翼连续牌型: 找出主牌最高点数并比较
    std::vector<std::vector<Card>>
    filter_serial_kicker(const std::vector<std::vector<Card>>& moves,
                         short main_repeat, Level min_main) {
        std::vector<std::vector<Card>> out;
        for (auto& m : moves) {
            Combo c(m.begin(), m.end());
            if (c.is_invalid()) continue;
            short cnts[MAX_LEVEL + 1] = {};
            for (Card card : m) cnts[card2level(card)]++;
            Level my_main = 0;
            for (Level lv = 0; lv < NUM_LEVELS; lv++)
                if (cnts[lv] == main_repeat && lv > my_main) my_main = lv;
            if (my_main > min_main) out.push_back(m);
        }
        return out;
    }

    // 四带二: 找出四张点数并比较
    std::vector<std::vector<Card>>
    filter_quad_kicker(const std::vector<std::vector<Card>>& moves, short quad_count) {
        std::vector<std::vector<Card>> out;
        for (auto& m : moves) {
            Combo c(m.begin(), m.end());
            if (c.is_invalid()) continue;
            short cnts[MAX_LEVEL + 1] = {};
            for (Card card : m) cnts[card2level(card)]++;
            Level my_main = 0;
            for (Level lv = 0; lv < NUM_LEVELS; lv++)
                if (cnts[lv] == quad_count) { my_main = lv; break; }
            if (my_main > last_.main_level) out.push_back(m);
        }
        return out;
    }
};

#endif // LEGAL_MOVES_H
