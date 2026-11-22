from collections import Counter
import numpy as np
import torch

from douzero.env.game import GameEnv
from douzero.env import oracle
from douzero.env import opponent_modeling


Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])


class Env:
    """DouEnMH 训练环境 — 包装 GameEnv, 提供 DMC 兼容的 reset/step 接口。

    通过 DummyAgent 注入模型决策, 观测编码使用 8×15 120-dim 矩阵。
    """

    def __init__(self, objective):
        self.objective = objective

        # 初始化玩家
        # 为目标位置使用三个虚拟玩家
        self.players = {}
        for position in ['first', 'second', 'third', 'landlord', 'landlord_down', 'landlord_up']:
            self.players[position] = DummyAgent(position)

        # 初始化内部环境
        self._env = GameEnv(self.players)
        self.total_round = 0
        self.infoset = None

        # 出牌历史编码增量缓存 (35 行 × 120 维)
        # 避免每步重编码全部 35 手历史 — 仅编码新增动作后移位
        self._history_cache = np.full((35, CARD_MATRIX_DIM), -1, dtype=np.int8)
        self._history_len = 0

    def reset(self, model, device, flags=None):
        self._env.reset()
        oracle._dp_global_cache_clear()  # 新局清空神谕 DP 缓存
        if opponent_modeling.ENABLE_C5:
            opponent_modeling.reset_c5_state()  # 新局清空 C5 行为风格统计

        # 重置历史缓存
        self._history_cache.fill(-1)
        self._history_len = 0

        # 随机洗牌
        _deck = deck.copy()
        np.random.shuffle(_deck)
        card_play_data = {'first': _deck[:17],
                          'second': _deck[20:37],
                          'third': _deck[37:],
                          'three_landlord_cards': _deck[17:20]
                          }
        for key in card_play_data:
            card_play_data[key].sort()
        # 重置叫牌
        self._env.bid_init(card_play_data)

        bid_over = self._bid_over
        self.infoset = self._bid_infoset
        return get_obs(self.infoset, bid_over, self._history_cache)

    def _update_history_cache(self, card_play_action_seq):
        """增量更新出牌历史编码缓存 — 仅编码新动作, 旧行移位.

        与全量重算 (_action_seq_list2array + _process_action_seq) 等价:
          Row 0 = 最近一手, Row 1-34 = 历史 (padding=-1).
        """
        if card_play_action_seq is None:
            self._history_cache.fill(-1)
            self._history_len = 0
            return self._history_cache

        new_len = len(card_play_action_seq)
        if new_len == self._history_len:
            return self._history_cache

        if new_len == 0:
            self._history_cache.fill(-1)
            self._history_len = 0
            return self._history_cache

        # 仅编码新增的动作 (通常 1 个/步)
        for _, cards in card_play_action_seq[self._history_len:]:
            self._history_cache[1:] = self._history_cache[:-1]
            if cards:
                self._history_cache[0] = cards2array_v2(cards)
            else:
                self._history_cache[0] = 0

        self._history_len = new_len
        return self._history_cache

    def step(self, action):
        if not self._draw:
            if self._bid_over:
                pos = self._acting_player_position
            else:
                pos = self._bidding_player_position

            self.players[pos].set_action(action)
            self._env.step()
        if not self._draw:
            if self._bid_over:
                self.infoset = self._game_infoset
            else:
                self.infoset = self._bid_infoset

        done = False
        reward = 0.0
        if self._game_over:
            done = True
            reward = {
                "play": {
                    "landlord": self._get_reward("landlord") / 2,
                    "landlord_down": self._get_reward("landlord_down"),
                    "landlord_up": self._get_reward("landlord_up"),
                    "first": self._get_reward_bidding("first"),
                    "second": self._get_reward_bidding("second"),
                    "third": self._get_reward_bidding("third"),
                }
            }
            obs = None
        elif self._draw:
            obs = None
        else:
            if self._bid_over:
                history_array = self._update_history_cache(
                    self.infoset.card_play_action_seq)
            else:
                history_array = self._history_cache
            obs = get_obs(self.infoset, self._bid_over, history_array)
        return obs, reward, done, self._draw, {}

    def _get_reward(self, pos):
        winner = self._game_winner

        bomb_num = self._game_bomb_num + 1 if self._env.spring else self._game_bomb_num
        bid_count = self._env.bid_count

        multiply = 2
        if pos != 'landlord':
            pos = 'farmer'
            multiply = 1

        if bomb_num == 0:
            r = bid_count
        elif bomb_num == 1:
            r = bid_count * (1 + bomb_num)
        else:
            r = bid_count * (2 + 2 * (bomb_num - 1))

        if pos == winner:
            return r * multiply / 24
        else:
            return -r * multiply / 24

    def _get_reward_bidding(self, pos):
        if self._draw:
            return 0
        winner = self._bid_winner
        bomb_num = self._game_bomb_num + 1 if self._env.spring else self._game_bomb_num
        bid_count = self._env.bid_count
        multiply = 1 if '&' in winner else 2
        _multiply = 2 if multiply == 1 else 1
        if pos in winner:
            if bomb_num == 0:
                r = bid_count
            elif bomb_num == 1:
                r = bid_count * (1 + bomb_num)
            else:
                r = bid_count * (2 + 2 * (bomb_num - 1))
            return r * multiply / 24
        else:
            if bomb_num == 0:
                r = bid_count
            elif bomb_num == 1:
                r = bid_count * (1 + bomb_num)
            else:
                r = bid_count * (2 + 2 * (bomb_num - 1))
            return -r * _multiply / 24

    @property
    def _game_infoset(self):
        return self._env.game_infoset

    @property
    def _bid_infoset(self):
        return self._env.bid_infoset

    @property
    def _game_bomb_num(self):
        return self._env.get_bomb_num()


    @property
    def _acting_player_position(self):
        return self._env.acting_player_position

    @property
    def _bidding_player_position(self):
        return self._env.bidding_player_position

    @property
    def _game_over(self):
        return self._env.game_over

    @property
    def _bid_over(self):
        return self._env.bid_over

    @property
    def _game_winner(self):
        return self._env.get_winner()

    @property
    def _bid_winner(self):
        return self._env.get_winner_bid()

    @property
    def _draw(self):
        return self._env.draw


class DummyAgent(object):
    """占位玩家 — 接收外部设定的动作, 供 GameEnv 调用 act() 获取."""

    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        self.action = action


def get_obs(infoset, bid_over, history_array=None):
    """观测构造入口 — 根据叫分/出牌阶段分派到对应的观测编码函数.

    history_array: 增量缓存的历史编码 ((35,120)), 避免每步重算.
                   仅在出牌阶段使用; 叫分阶段忽略.
    """
    if infoset.player_position not in ["first", 'second', 'third', 'landlord', 'landlord_down', 'landlord_up']:
        raise ValueError('')
    if bid_over:
        return _get_obs_resnet(infoset, history_array)
    elif infoset.player_position in ["first", 'second', 'third']:
        return _get_bid_obs_resnet(infoset)
    else:
        return _get_obs_resnet(infoset, history_array)


# ── DouEnMH 8×15 卡牌编码矩阵 (120-dim) ──────────────────────────────

# 8×15 矩阵各行的常量定义
CARD_MATRIX_ROWS = 8
CARD_MATRIX_COLS = 15
CARD_MATRIX_DIM = CARD_MATRIX_ROWS * CARD_MATRIX_COLS  # 120

# 链起点检测的阈值
MIN_SINGLE_CHAIN = 5   # 单顺: ≥5 连
MIN_PAIR_CHAIN = 3     # 连对: ≥3 连对
MIN_TRIPLE_CHAIN = 2   # 飞机: ≥2 连三条
MAX_CHAIN_RANK = 12    # 链类牌型可用 rank 数: 0-11 (3-A), 排除 2/joker

# Row 0-3 的 8×15 计数递增编码（各等级最多4张，大小王固定 row=0）
NumOnes2ArrayV2 = {
    0: np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    1: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    2: np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    3: np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8),
    4: np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8),
}


def cards2array_v2(list_cards):
    """DouEnMH 8×15 卡牌编码矩阵 → 列主序展平为 120 维向量.

    Row 0-3: 计数行 (≥1, ≥2, ≥3, =4) — DouZero 风格
    Row 4:   断点标记 — DouMH 独立行展开
    Row 5:   单顺起点 — PerfectDou (≥5 连)
    Row 6:   连对起点 — PerfectDou (≥3 连对)
    Row 7:   飞机起点 — PerfectDou (≥2 连三条)

    列 0-12: 3→2; 列 13: 小王; 列 14: 大王.
    """
    if len(list_cards) == 0:
        return np.zeros(CARD_MATRIX_DIM, dtype=np.int8)

    matrix = np.zeros([CARD_MATRIX_ROWS, CARD_MATRIX_COLS], dtype=np.int8)
    counter = Counter(list_cards)

    # ── Step 1: 计数填充 Row 0–3 ──
    for card, n in counter.items():
        if card < 20:
            col = Card2Column[card]
        elif card == 20:
            col = 13
        else:  # card == 30
            col = 14
        matrix[0:4, col] = NumOnes2ArrayV2[min(n, 4)][0:4]

    # ── Step 2: 断点标记 Row 4 ──
    # 判定: Row0[i]=1 AND Row0[i+1]=0 (等级 i 有牌但 i+1 无牌)
    ranks_present = set(Card2Column[c] for c in list_cards if c < 20)
    for i in range(14):
        if i in ranks_present and (i + 1) not in ranks_present:
            matrix[4, i] = 1
    # Row4[14]=0 (大王无后继，且大小王不参与等级断点)

    # ── Step 3: 单顺起点 Row 5 (≥5 连, 起点 0..7, 链不入 rank 12+) ──
    for i in range(MAX_CHAIN_RANK - MIN_SINGLE_CHAIN + 1):  # 0..7
        if all(matrix[0, i + k] == 1 for k in range(MIN_SINGLE_CHAIN)):
            matrix[5, i] = 1

    # ── Step 4: 连对起点 Row 6 (≥3 连对, 起点 0..9, 链不入 rank 12+) ──
    for i in range(MAX_CHAIN_RANK - MIN_PAIR_CHAIN + 1):  # 0..9
        if all(matrix[1, i + k] == 1 for k in range(MIN_PAIR_CHAIN)):
            matrix[6, i] = 1

    # ── Step 5: 飞机起点 Row 7 (≥2 连三条, 起点 0..10, 链不入 rank 12+) ──
    for i in range(MAX_CHAIN_RANK - MIN_TRIPLE_CHAIN + 1):  # 0..10
        if all(matrix[2, i + k] == 1 for k in range(MIN_TRIPLE_CHAIN)):
            matrix[7, i] = 1

    return matrix.flatten('F')


def _action_seq_list2array(action_seq_list):
    """将出牌历史序列编码为 (N, 120) 矩阵 (v2 编码).

    每行编码约定 (与设计文档 B 节一致):
      - 有效出牌: cards2array_v2(cards) → 8×15 矩阵展平
      - pass (实际发生的不出): 全零 120 维向量 (携带"无法/选择不出"信息)
      - padding (序列不足 35 手的占位): 全 -1 标记 (哨兵值, 表示"该位置无数据")

    通过 isinstance(action, tuple) 区分 pass 与 padding:
      - tuple (pos, cards): 真实动作 (含 pass, cards 为空列表)
      - list []: 由 _process_action_seq 生成的占位哨兵
    """
    action_seq_array = np.full((len(action_seq_list), CARD_MATRIX_DIM), -1, dtype=np.int8)
    for row, action in enumerate(action_seq_list):
        if isinstance(action, tuple):
            _, cards = action
            if cards:
                action_seq_array[row, :] = cards2array_v2(cards)
            else:
                # pass: 全零编码
                action_seq_array[row, :] = 0
        # else: 保持 -1 (padding 哨兵)
    return action_seq_array


def _process_action_seq(sequence, length=35):
    """截取最近 `length` 手出牌历史，反转使最近的在前，不足用空列表填充."""
    sequence = sequence[-length:].copy()
    sequence = sequence[::-1]
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        sequence.extend(empty_sequence)
    return sequence


def _get_one_hot_bomb(bomb_num):
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot


def _compute_hand_stats(hand_cards):
    """计算自手牌型统计特征 (C1, 7 维) — 归一化线性标量.

    统计手牌中每种基础牌型可出的独立等级/链起点数量。
    """
    rank_cnt = [0] * 15
    for card in hand_cards:
        if card < 20:
            r = Card2Column[card]
        elif card == 20:
            r = 13
        else:
            r = 14
        rank_cnt[r] += 1

    n_solo = sum(1 for c in rank_cnt if c >= 1)
    n_pair = sum(1 for c in rank_cnt[:13] if c >= 2)
    n_trio = sum(1 for c in rank_cnt[:13] if c >= 3)
    n_bomb = sum(1 for c in rank_cnt[:13] if c == 4)
    if rank_cnt[13] >= 1 and rank_cnt[14] >= 1:
        n_bomb += 1  # 火箭

    # 链类牌型仅能使用 rank 0–11 (3–A), 2/joker 不可入链
    n_chain_solo = 0
    for i in range(MAX_CHAIN_RANK - MIN_SINGLE_CHAIN + 1):
        if all(rank_cnt[i + k] >= 1 for k in range(MIN_SINGLE_CHAIN)):
            n_chain_solo += 1

    n_chain_pair = 0
    for i in range(MAX_CHAIN_RANK - MIN_PAIR_CHAIN + 1):
        if all(rank_cnt[i + k] >= 2 for k in range(MIN_PAIR_CHAIN)):
            n_chain_pair += 1

    n_airplane = 0
    for i in range(MAX_CHAIN_RANK - MIN_TRIPLE_CHAIN + 1):
        if all(rank_cnt[i + k] >= 3 for k in range(MIN_TRIPLE_CHAIN)):
            n_airplane += 1

    raw = np.array([
        n_solo / 15.0,
        n_pair / 8.0,
        n_trio / 5.0,
        n_bomb / 6.0,
        n_chain_solo / 10.0,
        n_chain_pair / 6.0,
        n_airplane / 4.0,
    ], dtype=np.float32)
    return np.minimum(raw, 1.0)


def _get_obs_resnet(infoset, history_array=None):
    """组装 DouEnMH 观测字典 (8×15 编码, 120-dim card matrices).

    Z 矩阵 (40 行 × 120 维):
      行 0:  当前手牌 (my_hand)
      行 1:  对手手牌并集 (other_union)
      行 2:  自身累计已出牌 (self_played)
      行 3:  上手累计已出牌 (up_played)
      行 4:  下手累计已出牌 (down_played)
      行 5-39: 最近 35 手出牌历史 (含 pass=全零)

    X 向量 (52 或 56 维):
      bid_info(3) + num_cards_left(3, 归一化÷20) + spring(1) + bomb_num(15)
      + hand_stats(7)       — C1 自手牌型统计
      + oracle_features(5)  — C2 神谕特征
      + opponent_features(4) — C3 对手信念特征 (OHS_up/down + OBP_up/down)
      + ctp_features(14)    — C4 对手牌型偏好 (CTP_up ×7 + CTP_down ×7)
      + c5_features(4)      — C5 对手行为风格 (SWI_up/down + TP_up/down) [ENABLE_C5 时]

    动作矩阵同样为 8×15 → 120 维，prepend 到 Z 前方形成 z_batch.
    """
    num_legal_actions = len(infoset.legal_actions)
    pos = infoset.player_position

    # ── 卡牌矩阵行 (8×15 → 120-dim) ──
    my_handcards = cards2array_v2(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = cards2array_v2(infoset.other_hand_cards)

    # 角色相对位置映射: 从绝对位置(landlord/up/down) → 相对(自身/上手/下手)
    # 出牌顺序: landlord → landlord_down → landlord_up → landlord → ...
    relative_order = {'landlord': ['landlord', 'landlord_up', 'landlord_down'],
                      'landlord_down': ['landlord_down', 'landlord', 'landlord_up'],
                      'landlord_up': ['landlord_up', 'landlord_down', 'landlord']}
    self_pos, up_pos, down_pos = relative_order[pos]

    self_played = cards2array_v2(infoset.played_cards[self_pos])
    up_played = cards2array_v2(infoset.played_cards[up_pos])
    down_played = cards2array_v2(infoset.played_cards[down_pos])

    # ── 候选动作编码 (8×15 → 120-dim) ──
    my_action_batch = np.zeros((num_legal_actions, CARD_MATRIX_DIM), dtype=np.int8)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = cards2array_v2(action)

    # ── 出牌历史 (最近 35 手) ──
    if history_array is None:
        history_array = _action_seq_list2array(
            _process_action_seq(infoset.card_play_action_seq, length=35))

    # ── Z 矩阵堆叠 (40 行 × 120 维) ──
    z = np.vstack((
        my_handcards,          # 行 0: 当前手牌
        other_handcards,       # 行 1: 对手手牌并集
        self_played,           # 行 2: 自身已出牌
        up_played,             # 行 3: 上手已出牌
        down_played,           # 行 4: 下手已出牌
        history_array,         # 行 5-39: 最近 35 手历史
    ))  # shape: (40, 120)

    # ── X 向量 (辅助标量特征, 52 或 56 维) ──
    bid_info = np.array(infoset.bid_info).flatten().astype(np.float32) / 3.0
    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)

    num_cards_self = np.array([infoset.num_cards_left_dict[self_pos] / 20.0],
                               dtype=np.float32)
    num_cards_up = np.array([infoset.num_cards_left_dict[up_pos] / 20.0],
                             dtype=np.float32)
    num_cards_down = np.array([infoset.num_cards_left_dict[down_pos] / 20.0],
                               dtype=np.float32)
    num_cards = np.concatenate([num_cards_self, num_cards_up, num_cards_down])
    num_cards_batch = np.repeat(num_cards[np.newaxis, :],
                                num_legal_actions, axis=0)

    spring = np.array([1.0], dtype=np.float32) if infoset.spring else np.array([0.0], dtype=np.float32)
    spring_batch = np.repeat(spring[np.newaxis, :],
                             num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(infoset.bomb_num).astype(np.float32)
    bomb_num_batch = np.repeat(bomb_num[np.newaxis, :],
                               num_legal_actions, axis=0)

    # ── 自手牌型统计 (C1, 7 维) ──
    hand_stats = _compute_hand_stats(infoset.player_hand_cards)
    hand_stats_batch = np.repeat(hand_stats[np.newaxis, :],
                                 num_legal_actions, axis=0)

    # ── 神谕特征 (C2, 5 维) + 监督标签 (LOR, 3 维) — 逐动作计算 ──
    # 设计意图: 对齐 DouMH 多头机制 — 每个出牌动作都评估出牌后剩余手牌的神谕值。
    # 先对完整手牌做一次 DP 预热缓存, 后续每个动作的子状态查询直接 memo 命中 (O(1))。
    #
    # LOR 训练标签使用对手感知 N_passive 以提高精度 (训练时有完美信息)
    all_handcards = getattr(infoset, 'all_handcards', None)
    hand_opp = None
    if all_handcards is not None:
        if pos == 'landlord':
            hand_opp = all_handcards['landlord_up'] + all_handcards['landlord_down']
        else:
            hand_opp = all_handcards['landlord']

    # self_adv 的对手基准 (对手剩余牌数, 仅取决于状态, 不随动作改变)
    opp_up_rem = infoset.num_cards_left_dict[up_pos]
    opp_down_rem = infoset.num_cards_left_dict[down_pos]
    if pos == 'landlord':
        opponent_for_adv = min(opp_up_rem, opp_down_rem)
    else:
        opponent_for_adv = infoset.num_cards_left_dict['landlord']

    oracle_feat_list, oracle_labels = oracle.compute_oracle_batch(
        infoset.player_hand_cards, infoset.legal_actions,
        hand_opp=hand_opp, opponent_remaining=opponent_for_adv)
    oracle_feat_batch = np.stack(oracle_feat_list)  # (B, 5)

    # ── 对手信念特征 (C3, 4 维) ──
    opponent_feat, pool_vec, pool_total = opponent_modeling.compute_opponent_features(
        infoset.player_hand_cards,
        infoset.num_cards_left_dict[up_pos],
        infoset.num_cards_left_dict[down_pos],
        infoset.played_cards,
        infoset.three_landlord_cards,
        # 地主视角不能把底牌重复扣除（底牌本就在地主手牌里）
        is_landlord=(pos == 'landlord'),
    )  # (4,): [ohs_up, ohs_down, obp_up, obp_down]; + pool for C5 reuse
    opponent_feat_batch = np.repeat(opponent_feat[np.newaxis, :],
                                    num_legal_actions, axis=0)

    # ── 对手牌型偏好 (C4, 14 维) ──
    ctp_state = getattr(infoset, 'ctp_state', None)
    if ctp_state is None:
        ctp_state = opponent_modeling.init_ctp_state()
    ctp_feat = opponent_modeling.get_ctp_features(
        ctp_state, self_pos, up_pos, down_pos)
    ctp_feat_batch = np.repeat(ctp_feat[np.newaxis, :],
                               num_legal_actions, axis=0)

    # ── 对手行为风格 (C5, 4 维: SWI_up/down + TP_up/down) ──
    if opponent_modeling.ENABLE_C5:
        c5_state = opponent_modeling.get_c5_state()
        # pool_vec, pool_total 复用 C3 已构建的未知牌池 (避免重复建池)
        c5_cards_remaining = {
            up_pos: infoset.num_cards_left_dict[up_pos],
            down_pos: infoset.num_cards_left_dict[down_pos],
            self_pos: infoset.num_cards_left_dict[self_pos],
        }
        c5_obp = {
            up_pos: float(opponent_feat[2]),
            down_pos: float(opponent_feat[3]),
        }
        opponent_modeling.update_c5_from_history(
            infoset.card_play_action_seq, c5_state,
            pool_vec, pool_total, c5_cards_remaining, c5_obp)
        c5_feat = opponent_modeling.get_c5_features(
            c5_state, self_pos, up_pos, down_pos)
        c5_feat_batch = np.repeat(c5_feat[np.newaxis, :],
                                  num_legal_actions, axis=0)

    if opponent_modeling.ENABLE_C5:
        x_batch = np.hstack((
            bid_info_batch,         # 3
            num_cards_batch,        # 3
            spring_batch,           # 1
            bomb_num_batch,         # 15
            hand_stats_batch,       # 7   — C1 自手牌型统计
            oracle_feat_batch,      # 5   — C2 神谕特征
            opponent_feat_batch,    # 4   — C3 对手信念特征
            ctp_feat_batch,         # 14  — C4 对手牌型偏好
            c5_feat_batch,          # 4   — C5 对手行为风格
        ))  # (B, 56)

        x_no_action = np.hstack((
            bid_info,
            num_cards,
            spring,
            bomb_num,
            hand_stats,             # 7   — C1 自手牌型统计
            np.zeros(5, dtype=np.float32),  # 5   — C2 神谕占位 (逐动作, 由 utils 注入)
            opponent_feat,          # 4   — C3 对手信念特征
            ctp_feat,               # 14  — C4 对手牌型偏好
            c5_feat,                # 4   — C5 对手行为风格
        ))  # (56,)
    else:
        x_batch = np.hstack((
            bid_info_batch,         # 3
            num_cards_batch,        # 3
            spring_batch,           # 1
            bomb_num_batch,         # 15
            hand_stats_batch,       # 7   — C1 自手牌型统计
            oracle_feat_batch,      # 5   — C2 神谕特征
            opponent_feat_batch,    # 4   — C3 对手信念特征
            ctp_feat_batch,         # 14  — C4 对手牌型偏好
        ))  # (B, 52)

        x_no_action = np.hstack((
            bid_info,
            num_cards,
            spring,
            bomb_num,
            hand_stats,             # 7   — C1 自手牌型统计
            np.zeros(5, dtype=np.float32),  # 5   — C2 神谕占位 (逐动作, 由 utils 注入)
            opponent_feat,          # 4   — C3 对手信念特征
            ctp_feat,               # 14  — C4 对手牌型偏好
        ))  # (52,)

    # ── z_batch: 候选动作行 prepend 到 Z 前方 ──
    _z_batch = np.repeat(z[np.newaxis, :, :], num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:, np.newaxis, :]
    z_batch = np.concatenate((my_action_batch, _z_batch), axis=1)
    # z_batch shape: (B, 41, 120) — 1 动作行 + 5 静态行 + 35 历史行

    # ── V2 Cross-Attention 拆分数据 (零额外计算, 仅 reshape/reference) ──
    # state_z: (1, 5, 120) — 仅 5 行静态卡牌状态 (my_hand + other_union + played_×3)
    state_z = z[0:5, :][np.newaxis, :, :].astype(np.float32)
    # action_z: (B, 1, 120) — 候选动作矩阵
    action_z = my_action_batch.astype(np.float32)
    # history_z: (1, 35, 120) — 最近 35 手出牌历史 (用于 V2 History Encoder)
    history_z = history_array[np.newaxis, :, :].astype(np.float32)

    obs = {
        'position': pos,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.float32),
        'z': z.astype(np.int8),
        'oracle_labels': oracle_labels.astype(np.float32),  # (B, 3) LOR 监督标签
        'oracle_feat_batch': oracle_feat_batch.astype(np.float32),  # (B, 5) 逐动作神谕
        # V2 Cross-Attention 拆分数据
        'state_z': state_z,        # (1, 5, 120)
        'action_z': action_z,      # (B, 1, 120)
        'history_z': history_z,    # (1, 35, 120)
    }
    return obs


def _get_bid_obs_resnet(infoset):
    """叫分阶段观测组装 (8×15 编码)."""
    num_legal_actions = len(infoset.legal_actions)

    my_handcards = cards2array_v2(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    bid_info = np.array(infoset.bid_info).astype(np.float32)
    bid_info_batch = np.repeat(bid_info[np.newaxis, :],
                               num_legal_actions, axis=0)

    # 叫分动作编码: 将标量叫分值广播为 120 维向量
    my_action_batch = np.zeros((num_legal_actions, CARD_MATRIX_DIM), dtype=np.float32)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = np.full(CARD_MATRIX_DIM, action[0], dtype=np.float32)

    x_batch = bid_info_batch  # (B, 3)
    x_no_action = bid_info    # (3,)

    z = my_handcards[np.newaxis, :]  # (1, 120)

    _z_batch = np.repeat(z[np.newaxis, :, :], num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:, np.newaxis, :]
    z_batch = np.concatenate((my_action_batch, _z_batch), axis=1)
    # z_batch shape: (B, 2, 120) — 1 动作行 + 1 手牌行

    obs = {
        'position': infoset.player_position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.float32),
        'z': z.astype(np.int8),
        # 叫分模型无 LOR 头, 但 rollout buffer 仍尝试索引 oracle_labels[_action_idx],
        # 填充与合法动作数一致的 dummy 标签以避免 IndexError (loss 端不会被消费)
        'oracle_labels': np.zeros((num_legal_actions, 3), dtype=np.float32),
    }
    return obs

