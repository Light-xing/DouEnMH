r"""
Python ↔ C++ 交叉验证：用真实游戏状态生成输入张量，保存为二进制，
供 C++ 端加载并比对推理输出。

在 PyTorch 1.5.0 桥接环境中运行:
    h:\bot\environments\bridge_python\python38_full\python.exe ^
      h:\bot\DouEnMH-design\bridge\cross_validate\cross_validate.py ^
      --checkpoint-dir <ckpt 目录> --output-dir <输出目录>

⚠️ 尚未迁移到当前架构（本文件当前**不可用于验证**）:
   本脚本的输入构造函数 build_bid_inputs / build_play_inputs 仍是**上一代 54 维架构**
   (4×13 + 双王 = 54 维卡牌编码、Z 矩阵 44 行、X 向量 18 维、历史 32 手)。
   当前模型为 8×15 = 120 维编码、Z 矩阵 41 行、X 向量 52 或 56 维、历史 35 手。
   因此用当前 checkpoint 运行会在 load_state_dict 或首层卷积处**报错**（不会静默给出错误数据），
   但报错信息本身不会说明原因 —— 故本脚本会在启动时直接退出并提示。

   迁移方式（避免再次重复实现编码逻辑）：把这两个构造函数改为**复用训练端特征** ——
   构造 douzero.env.game.InfoSet 后调用 douzero.env.env.get_obs()，
   可参考 DouEnMH/DouEnMH-new/infer.py 的用法。改完后把 ARCH_MIGRATED 置为 True。
"""
import os
import re
import sys
import argparse
import numpy as np
import torch

# 本脚本的输入构造是否已迁移到当前 120 维架构
ARCH_MIGRATED = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compat_models_douenmh import GeneralModelBid, GeneralModelTransformer

# ---- 配置 (均可由 CLI 覆盖) ----
DEFAULT_CHECKPOINT_DIR = r'h:\bot\DouEnMH-design\DouEnMH\DouEnMH-new\douzero_checkpoints\DouEnMH_v1.2'
CKPT_NAMES = ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']

CHECKPOINT_DIR = DEFAULT_CHECKPOINT_DIR
SUFFIX = None
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
TRANSFORMER_KWARGS = dict(d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024)


def detect_suffix(checkpoint_dir):
    """返回 6 个位置共有的最大训练帧数后缀; 找不到返回 None。"""
    pat = re.compile(r'^([a-z_]+)_(\d+)\.ckpt$')
    seen = {}
    for f in os.listdir(checkpoint_dir):
        m = pat.match(f)
        if m:
            seen.setdefault(m.group(1), set()).add(int(m.group(2)))
    if not all(n in seen for n in CKPT_NAMES):
        return None
    common = set.intersection(*[seen[n] for n in CKPT_NAMES])
    return max(common) if common else None


def parse_args():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser(description='Python ↔ C++ 交叉验证数据生成')
    p.add_argument('--checkpoint-dir', default=DEFAULT_CHECKPOINT_DIR,
                   help='训练 checkpoint 目录 (6 个位置各一个 .ckpt)')
    p.add_argument('--suffix', default=None,
                   help='训练帧数后缀; 缺省时自动探测 6 个位置共有的最大后缀')
    p.add_argument('--output-dir', default=os.path.join(here, 'data'))
    p.add_argument('--d-model', default=256, type=int)
    p.add_argument('--nhead', default=8, type=int)
    p.add_argument('--num-encoder-layers', default=6, type=int)
    p.add_argument('--dim-feedforward', default=1024, type=int)
    return p.parse_args()

# ---- 辅助函数（与 env.py 完全一致）----
Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7, 11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]), 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]), 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

def cards2array(list_cards, include_breaks=False):
    if len(list_cards) == 0:
        return np.zeros(54, dtype=np.float32)
    matrix = np.zeros([4, 13], dtype=np.float32)
    jokers = np.zeros(2, dtype=np.float32)
    from collections import Counter
    counter = Counter(list_cards)
    for card, num_times in counter.items():
        if card < 20:
            matrix[:, Card2Column[card]] = NumOnes2Array[num_times]
        elif card == 20:
            jokers[0] = 1.0
        elif card == 30:
            jokers[1] = 1.0
    result = np.concatenate((matrix.flatten('F'), jokers)).astype(np.float32)
    if include_breaks:
        ranks_present = set()
        for card in list_cards:
            if card < 20:
                ranks_present.add(Card2Column[card])
        num_breaks = 0
        for r in range(11):
            if r in ranks_present and (r + 1) not in ranks_present:
                num_breaks += 1
        has_breaks = 1 if num_breaks > 0 else 0
        many_breaks = 1 if num_breaks > 3 else 0
        result[52] = jokers[0] + (has_breaks << 1)
        result[53] = jokers[1] + (many_breaks << 1)
    return result

def save_tensor(path, t):
    """保存 float32 tensor 为原始二进制文件，格式: [n_dims(4 bytes)][dim0][dim1]...[data]"""
    arr = t.cpu().detach().numpy().astype(np.float32)
    with open(path, 'wb') as f:
        # 写入维度数
        ndim = np.array([len(arr.shape)], dtype=np.int32)
        ndim.tofile(f)
        # 写入各维度大小
        dims = np.array(arr.shape, dtype=np.int32)
        dims.tofile(f)
        # 写入数据
        arr.tofile(f)
    print(f"  Saved: {path}  shape={arr.shape}  dtype=float32")
    return arr


def build_bid_inputs(hand_cards, bid_info, legal_actions):
    """构建叫分模型输入，完全对齐 env.py _get_bid_obs_resnet"""
    B = len(legal_actions)
    my_handcards = cards2array(hand_cards, include_breaks=True)

    # bid_info_z: (3, 54) = bid_info broadcast to 54 dims
    bid_info_z = np.multiply(np.array(bid_info, dtype=np.float32),
                            np.ones((54, 3))).transpose((1, 0))

    # z_batch: (B, 5, 54)
    z_batch = np.zeros((B, 5, 54), dtype=np.float32)
    for j, action in enumerate(legal_actions):
        # Row 0: action encoding (bid value broadcast to 54 dims)
        z_batch[j, 0, :] = float(action[0])
        # Row 1: my_handcards
        z_batch[j, 1, :] = my_handcards
        # Rows 2-4: bid_info_z (3×54)
        z_batch[j, 2:5, :] = bid_info_z

    # x_batch: (B, 3) — bid_info for each action
    x_batch = np.repeat(np.array(bid_info, dtype=np.float32)[np.newaxis, :], B, axis=0)

    return torch.from_numpy(z_batch), torch.from_numpy(x_batch)


def build_play_inputs(hand_cards, other_hand_cards, three_landlord_cards,
                       num_cards_left_dict, played_cards_dict,
                       bid_info, spring, bomb_num, card_play_action_seq,
                       legal_actions):
    """构建出牌模型输入，完全对齐 env.py _get_obs_resnet"""
    B = len(legal_actions)
    my_handcards = cards2array(hand_cards, include_breaks=True)
    other_hc = cards2array(other_hand_cards, include_breaks=False)
    three_lc = cards2array(three_landlord_cards, include_breaks=False)
    landlord_played = cards2array(played_cards_dict.get('landlord', []), include_breaks=False)
    up_played = cards2array(played_cards_dict.get('landlord_up', []), include_breaks=False)
    down_played = cards2array(played_cards_dict.get('landlord_down', []), include_breaks=False)

    spring_arr = np.ones(54, dtype=np.float32) if spring else np.zeros(54, dtype=np.float32)

    bid_info_arr = np.array(bid_info, dtype=np.float32)
    bid_info_z = np.multiply(bid_info_arr, np.ones((54, 3))).transpose((1, 0))

    # num_cards_left: 20 + 17 + 17
    def one_hot(num_left, max_cards):
        v = np.zeros(max_cards, dtype=np.float32)
        if num_left > 0:
            v[num_left - 1] = 1.0
        return v
    ncl = np.concatenate([
        one_hot(num_cards_left_dict.get('landlord', 20), 20),
        one_hot(num_cards_left_dict.get('landlord_up', 17), 17),
        one_hot(num_cards_left_dict.get('landlord_down', 17), 17),
    ])

    # action_seq: last 32, reversed, pad
    seq = card_play_action_seq[-32:].copy()
    seq = seq[::-1]
    while len(seq) < 32:
        seq.append([])
    action_seq_arr = np.zeros((32, 54), dtype=np.float32)
    for i, entry in enumerate(seq):
        if entry:
            action_seq_arr[i] = cards2array(entry[1], include_breaks=False)

    # bomb_num one-hot
    bomb_onehot = np.zeros(15, dtype=np.float32)
    if 0 <= bomb_num < 15:
        bomb_onehot[bomb_num] = 1.0

    # 组装 x_batch = (B, 18) = bid_info(3) + bomb_onehot(15)
    x_no_action = np.concatenate([bid_info_arr / 3.0, bomb_onehot]).astype(np.float32)
    x_batch = np.repeat(x_no_action[np.newaxis, :], B, axis=0)

    # 组装 z_batch = (B, 44, 54)
    z_batch = np.zeros((B, 44, 54), dtype=np.float32)
    for j, action in enumerate(legal_actions):
        row = 0
        # Row 0: action encoding
        z_batch[j, row, :] = cards2array(action, include_breaks=False); row += 1
        # Row 1: num_cards_left
        z_batch[j, row, :] = ncl; row += 1
        # Row 2: my_handcards
        z_batch[j, row, :] = my_handcards; row += 1
        # Row 3: other_handcards
        z_batch[j, row, :] = other_hc; row += 1
        # Row 4: three_landlord_cards
        z_batch[j, row, :] = three_lc; row += 1
        # Row 5: landlord_played
        z_batch[j, row, :] = landlord_played; row += 1
        # Row 6: up_played
        z_batch[j, row, :] = up_played; row += 1
        # Row 7: down_played
        z_batch[j, row, :] = down_played; row += 1
        # Row 8-10: bid_info_z
        z_batch[j, row:row+3, :] = bid_info_z; row += 3
        # Row 11: spring
        z_batch[j, row, :] = spring_arr; row += 1
        # Row 12-43: action_seq
        z_batch[j, row:row+32, :] = action_seq_arr; row += 32

    return torch.from_numpy(z_batch), torch.from_numpy(x_batch)


def validate_bid_model(name, checkpoint_path):
    """验证一个叫分模型：加载权重 → 真实输入推理 → 保存张量"""
    print(f"\n{'='*60}")
    print(f"验证叫分模型: {name}")

    model = GeneralModelBid()
    model.eval()
    sd = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(sd)
    print(f"  权重加载: {len(sd)} keys")

    # 构造真实输入：3个合法叫分动作 (0, 1, 2, 3)
    hand_cards = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 14, 14, 17, 17, 20]  # 17张
    bid_info = [-1, -1, -1]      # 初始叫分
    legal_actions = [[0], [1], [2], [3]]

    z, x = build_bid_inputs(hand_cards, bid_info, legal_actions)

    with torch.no_grad():
        output = model(z, x)  # (4, 1)

    print(f"  z shape: {z.shape}  x shape: {x.shape}  output shape: {output.shape}")
    print(f"  Q values: {output.squeeze().tolist()}")
    print(f"  Best action (argmax): {output.argmax().item()}")

    # 保存
    prefix = os.path.join(OUTPUT_DIR, f"{name}")
    save_tensor(f"{prefix}_z.bin", z)
    save_tensor(f"{prefix}_x.bin", x)
    save_tensor(f"{prefix}_out.bin", output)

    # 验证不同输入下模型能区分不同动作
    q_vals = output.squeeze().tolist()
    if len(set(q_vals)) > 1:
        print(f"  [OK] 模型对{bold_good('不同动作')}给出了{bold_good('不同Q值')}（有区分度）")
    else:
        print(f"  [WARN] 所有动作Q值相同，模型可能未正确加载")

    return output


def validate_play_model(name, checkpoint_path):
    """验证一个出牌模型：加载权重 → 真实输入推理 → 保存张量"""
    print(f"\n{'='*60}")
    print(f"验证出牌模型: {name}")

    model = GeneralModelTransformer(**TRANSFORMER_KWARGS)
    model.eval()
    sd = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(sd)
    print(f"  权重加载: {len(sd)} keys")

    # 构造真实出牌状态
    hand_cards = [3,4,5,6,6,6,6, 7,8,9,10,10,10, 11,12,13,14,14,17,20]
    other_cards = [3,3,4,5,7,8,9,11,12,13,13,13,14,17,17,30]
    landlord_cards = [20, 30, 17]
    played = {'landlord': [3,3,3], 'landlord_up': [4,4,4,4], 'landlord_down': [8,8]}
    num_left = {'landlord': 17, 'landlord_up': 13, 'landlord_down': 15}
    bid_info = [-1, 2, 0]
    spring = False
    bomb_num = 2
    # action_seq format: [player_pos, [card_list]] per entry (matching env.py)
    action_seq = [
        [0, [3,3,3]], [], [1, [4,4,4,4]], [2, [8,8]], [],
        [0, [5,6,7,8,9]], [], [],
    ]

    # 合法动作（简化：PASS + 几个单张）
    legal_actions = [
        [],
        [10],
        [10,10],
        [6,6,6,6],
        [14,14],
        [14],
        [17],
    ]

    z, x = build_play_inputs(hand_cards, other_cards, landlord_cards,
                              num_left, played, bid_info, spring, bomb_num,
                              action_seq, legal_actions)

    with torch.no_grad():
        output = model(z, x)  # (B, 1)

    print(f"  src1 shape: {z.shape}  src2 shape: {x.shape}  output shape: {output.shape}")
    print(f"  Q values: {[f'{v:.4f}' for v in output.squeeze().tolist()]}")
    print(f"  Best action (argmax): {output.argmax().item()}")

    # 保存
    prefix = os.path.join(OUTPUT_DIR, f"{name}")
    save_tensor(f"{prefix}_src1.bin", z)
    save_tensor(f"{prefix}_src2.bin", x)
    save_tensor(f"{prefix}_out.bin", output)

    q_vals = output.squeeze().tolist()
    if len(set([f'{v:.4f}' for v in q_vals])) > 1:
        print(f"  [OK] 模型对不同动作给出了不同Q值（有区分度）")
    else:
        print(f"  [WARN] 所有动作Q值相同")

    return output


def bold_good(s):
    return s


def main():
    global CHECKPOINT_DIR, SUFFIX, OUTPUT_DIR, TRANSFORMER_KWARGS

    if not ARCH_MIGRATED:
        print("=" * 68)
        print("本脚本尚未迁移到当前 120 维架构, 不能用于跨端一致性验证。")
        print("  · 现有输入构造: 54 维编码 / Z 44 行 / X 18 维 / 历史 32 手 (上一代)")
        print("  · 当前模型    : 120 维编码 / Z 41 行 / X 52|56 维 / 历史 35 手")
        print("  迁移方式见本文件头部说明 (改为复用 douzero.env.env.get_obs)。")
        print("=" * 68)
        sys.exit(2)

    args = parse_args()
    CHECKPOINT_DIR = args.checkpoint_dir
    OUTPUT_DIR = args.output_dir
    TRANSFORMER_KWARGS = dict(d_model=args.d_model, nhead=args.nhead,
                              num_encoder_layers=args.num_encoder_layers,
                              dim_feedforward=args.dim_feedforward)

    if args.suffix:
        SUFFIX = str(args.suffix)
    else:
        SUFFIX = detect_suffix(CHECKPOINT_DIR)
        if SUFFIX is None:
            print(f"[FAIL] 在 {CHECKPOINT_DIR} 中找不到 6 个位置共有的 checkpoint 后缀")
            sys.exit(2)
        SUFFIX = str(SUFFIX)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"PyTorch version: {torch.__version__}")
    print(f"checkpoint 目录: {CHECKPOINT_DIR}   (后缀 {SUFFIX})")
    print(f"输出目录: {OUTPUT_DIR}")

    all_ok = True

    # 验证叫分模型
    for name in ['first', 'second', 'third']:
        ckpt = os.path.join(CHECKPOINT_DIR, f'{name}_{SUFFIX}.ckpt')
        if os.path.exists(ckpt):
            try:
                validate_bid_model(name, ckpt)
            except Exception as e:
                print(f"  [FAIL] 失败: {e}")
                all_ok = False
        else:
            print(f"  [WARN] 跳过 {name}: checkpoint 不存在")
            all_ok = False

    # 验证出牌模型
    for name in ['landlord', 'landlord_up', 'landlord_down']:
        ckpt = os.path.join(CHECKPOINT_DIR, f'{name}_{SUFFIX}.ckpt')
        if os.path.exists(ckpt):
            try:
                validate_play_model(name, ckpt)
            except Exception as e:
                print(f"  [FAIL] 失败: {e}")
                all_ok = False
        else:
            print(f"  [WARN] 跳过 {name}: checkpoint 不存在")
            all_ok = False

    print(f"\n{'='*60}")
    if all_ok:
        print("全部验证通过！张量已保存到:", OUTPUT_DIR)
        print("\nC++ 端使用方式:")
        print("  1. 读取 .bin 文件 (维度信息 + float32 数据)")
        print("  2. 创建 torch::from_blob() 张量")
        print("  3. 调用 model.forward()")
        print("  4. 与 _out.bin 中的输出逐元素对比")
    else:
        print("部分验证失败，请检查 checkpoint 文件")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
