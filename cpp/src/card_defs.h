// =============================================================================
// card_defs.h — Botzone 斗地主 Card / Level 双编码基础定义
// =============================================================================
// 两套编码共存:
//   Card (0-53):  Botzone 原生 ID, 每张牌唯一标识 (含花色), 用于动作生成与输出
//   Level (0-14): 仅看点数忽略花色, 用于牌型检测 & 大小比较
//
// 转换公式: card2level(c) = c/4 + c/53
//   c = 0~51 → level 0~12 (3,4,5,6,7,8,9,10,J,Q,K,A,2)
//   c = 52   → level 13 (小王)
//   c = 53   → level 14 (大王)
//
// 约束:
//   连续牌型 (顺子/连对/飞机) 上限为 MAX_STRAIGHT_LEVEL=11 (A), 不能含 2 和王
//   大小王仅 Row 0 可为 1 (不存在 ≥2 张), 不参与链类牌型
// =============================================================================

#ifndef CARD_DEFS_H
#define CARD_DEFS_H

#include <array>
#include <cstdint>
#include <vector>

using Card  = short;   // Botzone 卡牌 ID (0-53)
using Level = short;   // 点数等级 (0-14)

// ---- 特殊牌常量 ----
constexpr Card  CARD_JOKER_SMALL = 52;
constexpr Card  CARD_JOKER_BIG   = 53;
constexpr Level NUM_LEVELS       = 15;    // 0-14 共 15 个等级
constexpr Level MAX_LEVEL        = 14;
constexpr Level MAX_STRAIGHT_LEVEL = 11;  // A, 顺子/飞机上限
constexpr Level LEVEL_JOKER_SMALL = 13;
constexpr Level LEVEL_JOKER_BIG   = 14;

// Card → Level: 扣去花色, 只留点数
inline constexpr Level card2level(Card card) {
    return card / 4 + card / 53;
}

// 按等级统计牌张数 (15 维)
using CardCount = std::array<short, NUM_LEVELS>;

#endif // CARD_DEFS_H
