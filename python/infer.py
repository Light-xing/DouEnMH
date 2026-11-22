"""轻量推理脚本 — 加载模型对单局游戏进行决策。

用法:
    # 出牌推理（地主）
    python infer.py --position landlord --model checkpoints/landlord.ckpt --hand "3,3,5,6,7,7,7,9,10,10,11,12,13,14,17,20,30" --history "[]"

    # 叫牌推理
    python infer.py --position first --model checkpoints/first.ckpt --hand "3,4,5,6,7,8,9,10,11,12,13,14,17,20,30" --bid-info "0,0"
"""
import argparse
import numpy as np
import torch


def parse_cards(s):
    """解析牌列表: '3,3,5,7,20' -> [3,3,5,7,20]. 支持空字符串和 '[]'."""
    s = s.strip()
    if not s or s == '[]':
        return []
    return [int(x.strip()) for x in s.strip('[]').split(',') if x.strip()]


def main():
    parser = argparse.ArgumentParser(description='DouEnMH 轻量推理')
    parser.add_argument('--position', type=str, required=True,
                        choices=['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down'],
                        help='玩家位置')
    parser.add_argument('--model', type=str, required=True, help='模型 .ckpt 路径')
    parser.add_argument('--hand', type=str, required=True, help='手牌, 逗号分隔, 如 "3,3,5,7,20"')
    parser.add_argument('--history', type=str, default='', help='最近出牌序列 (JSON格式列表的列表)')
    parser.add_argument('--other-hand', type=str, default='', help='其他玩家手牌联合 (训练时用完美信息)')
    parser.add_argument('--played-landlord', type=str, default='', help='地主已出牌')
    parser.add_argument('--played-up', type=str, default='', help='上家已出牌')
    parser.add_argument('--played-down', type=str, default='', help='下家已出牌')
    parser.add_argument('--three-cards', type=str, default='', help='三张底牌')
    parser.add_argument('--bid-info', type=str, default='0,0,0', help='叫牌信息 (逗号分隔3个值)')
    parser.add_argument('--bomb-num', type=int, default=0, help='已出炸弹数')
    parser.add_argument('--cards-left-landlord', type=int, default=20, help='地主剩余牌数')
    parser.add_argument('--cards-left-up', type=int, default=17, help='上家剩余牌数')
    parser.add_argument('--cards-left-down', type=int, default=17, help='下家剩余牌数')
    parser.add_argument('--spring', action='store_true', help='是否春天')
    parser.add_argument('--top-k', type=int, default=5, help='输出前K个最优动作')
    parser.add_argument('--device', type=str, default='cpu', help='推理设备 (cpu/cuda)')
    parser.add_argument('--use-v2', action='store_true', help='使用 V2 Cross-Attention 模型')
    parser.add_argument('--actions', type=str, default='',
                        help='完整出牌序列 JSON [[role, cards],...]（与 card_play_action_seq 同构）')
    parser.add_argument('--legal', type=str, default='',
                        help='显式指定合法动作(JSON: [[card,...],...], MH 编码); 给定则不自动生成')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # ---- 加载模型（自动检测维度） ----
    from douzero.dmc.models import model_dict, model_dict_v2, GeneralModelTransformer, GeneralModelTransformerV2
    use_v2 = args.use_v2
    state_dict = torch.load(args.model, map_location=device)

    if args.position in ('first', 'second', 'third'):
        model = model_dict[args.position]()
    else:
        # 检测 V2 模型: 看是否有 V2 特有的参数 (cross_attn, state_proj 替代 state_embed 等)
        has_v2_keys = any(k.startswith('cross_attn.') or k.startswith('state_proj.')
                          for k in state_dict.keys())
        if has_v2_keys:
            use_v2 = True
            print('[检测] 权重包含 V2 特征 key (cross_attn / state_proj)，自动启用 use_v2')
        # 从 state_dict 推断 Transformer 维度
        # V2 用 state_proj, V1 用 state_embed
        if use_v2:
            embed_key = 'state_proj.weight'
            TransformerCls = GeneralModelTransformerV2
        else:
            embed_key = 'state_embed.weight'
            TransformerCls = GeneralModelTransformer
        embed_weight = state_dict.get(embed_key)
        d_model = embed_weight.shape[0]  # Conv1d out_channels

        # 计数编码器层数 (state_encoder)
        num_layers = 0
        while f'state_encoder.layers.{num_layers}.self_attn.in_proj_weight' in state_dict:
            num_layers += 1

        ff_weight = state_dict['state_encoder.layers.0.linear1.weight']
        dim_feedforward = ff_weight.shape[0]

        # nhead 无法从权重精确推导，默认 8；可通过 CLI 覆盖
        nhead = getattr(args, 'nhead', None) or 8

        model = TransformerCls(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_layers,
            dim_feedforward=dim_feedforward)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f'[模型] {args.model} 已加载到 {device} ({"V2" if use_v2 else "V1"})')
    if args.position not in ('first', 'second', 'third'):
        print(f'[架构] d_model={d_model} nhead={nhead} layers={num_layers} d_ff={dim_feedforward}')

    # ---- 构建 InfoSet ----
    from douzero.env.game import InfoSet
    from douzero.env.env import get_obs
    from douzero.env.move_generator import MovesGener

    hand = parse_cards(args.hand)
    print(f'[手牌] {hand}')

    # 解析历史出牌序列
    history = []
    if args.history:
        import json
        history_raw = json.loads(args.history)
        for item in history_raw:
            pos = item[0]
            cards_raw = item[1]
            if isinstance(cards_raw, list):
                cards = cards_raw
            else:
                cards = parse_cards(str(cards_raw))
            history.append((pos, cards))

    # --actions: 完整出牌序列（含我方与两家的全部动作），覆盖 --history
    # 训练端 get_obs 的历史编码取 card_play_action_seq 的最近 35 手，
    # 只喂最近两手会得到完全不同的历史特征。
    if args.actions:
        import json as _json
        _seq = _json.loads(args.actions)
        history = [(item[0], [int(c) for c in (item[1] or [])]) for item in _seq]

    infoset = InfoSet(args.position)
    infoset.player_hand_cards = hand
    infoset.other_hand_cards = parse_cards(args.other_hand)
    infoset.three_landlord_cards = parse_cards(args.three_cards)
    infoset.played_cards = {
        'landlord': parse_cards(args.played_landlord),
        'landlord_up': parse_cards(args.played_up),
        'landlord_down': parse_cards(args.played_down),
    }
    infoset.card_play_action_seq = history
    infoset.num_cards_left_dict = {
        'landlord': args.cards_left_landlord,
        'landlord_up': args.cards_left_up,
        'landlord_down': args.cards_left_down,
    }
    infoset.bid_info = [int(x) for x in args.bid_info.split(',')]
    infoset.bomb_num = args.bomb_num
    infoset.spring = args.spring
    infoset.last_move_dict = {}
    infoset.last_move = None

    # 生成合法动作
    if args.position in ('first', 'second', 'third'):
        # 叫牌阶段: 合法动作为标量叫分值
        bid_info = [int(x) for x in args.bid_info.split(',')]
        max_bid = max(bid_info) if bid_info else 0
        legal_bids = [[0]]  # 不叫
        for b in range(1, 4):
            if b > max_bid:
                legal_bids.append([b])
        infoset.legal_actions = legal_bids
    else:
        # 待压牌: 与训练环境一致 —— 看最后一手, 若为过牌则取倒数第二手
        rival = []
        if history:
            if len(history[-1][1]) == 0:
                rival = history[-2][1] if len(history) >= 2 else []
            else:
                rival = history[-1][1]
        infoset.last_move = rival

        if args.legal:
            import json as _json
            raw_legal = _json.loads(args.legal)
            infoset.legal_actions = [[int(c) for c in a] for a in raw_legal]
            if rival and [] not in infoset.legal_actions:
                infoset.legal_actions.append([])
        else:
            # 旧逻辑: 未传 --legal 时按"领出"枚举（跟牌局面不保证正确, 会打印告警）
            if rival:
                print('[警告] 未传 --legal, 跟牌局面的候选集可能不正确')
            mg = MovesGener(hand)
            all_moves = mg.gen_moves()
            if rival:
                all_moves.append([])
            infoset.legal_actions = all_moves

    # 叫牌位置 (first/second/third) 走叫牌阶段编码
    bid_over = args.position not in ('first', 'second', 'third')

    # 编码并推理
    obs = get_obs(infoset, bid_over=bid_over)
    z_batch = torch.from_numpy(obs['z_batch']).float().to(device)
    x_batch = torch.from_numpy(obs['x_batch']).float().to(device)

    with torch.no_grad():
        if use_v2 and args.position not in ('first', 'second', 'third'):
            state_z = torch.from_numpy(obs['state_z']).float().to(device)
            action_z = torch.from_numpy(obs['action_z']).float().to(device)
            history_z = torch.from_numpy(obs['history_z']).float().to(device)
            result = model.forward(state_z, action_z, history_z, x_batch, return_value=True)
        else:
            result = model.forward(z_batch, x_batch, return_value=True)
        win_rate, win, lose = result['values']
        # LOR oracle heads (lor_a, lor_p, lor_d) are available in result['oracle']
        # but not needed for action selection — Q = wr * win + (1-wr) * lose as before
        _wr = (win_rate + 1) / 2
        q_values = (_wr * win + (1 - _wr) * lose).squeeze(-1).cpu().numpy()

    # 排序输出
    sorted_idx = np.argsort(q_values)[::-1]
    top_k = min(args.top_k, len(sorted_idx))
    legal_actions = infoset.legal_actions

    print(f'\n[推理结果] 共 {len(legal_actions)} 个合法动作')
    print(f'{"排名":<4} {"动作":<30} {"Q值":>10}')
    print('-' * 48)
    for rank, idx in enumerate(sorted_idx[:top_k]):
        action = legal_actions[idx]
        if args.position in ('first', 'second', 'third'):
            action_str = f'叫{action[0]}分' if action[0] > 0 else '不叫'
        elif action:
            action_str = str(action)
        else:
            action_str = '过牌'
        print(f'{rank+1:<4} {action_str:<30} {q_values[idx]:>10.6f}')

    best = legal_actions[sorted_idx[0]]
    if args.position in ('first', 'second', 'third'):
        print(f'\n[最优动作] {"叫" + str(best[0]) + "分" if best[0] > 0 else "不叫"}')
    else:
        print(f'\n[最优动作] {best if best else "过牌"}')


if __name__ == '__main__':
    main()
