import os
import argparse
from douzero.evaluation.simulation import evaluate
import random
import numpy as np
import torch

# 设置 random 模块的随机数种子
random.seed(78419)
# 设置 NumPy 的随机数种子
np.random.seed(78419)
# 设置 PyTorch 的随机数种子
torch.manual_seed(78419)
# 如果你的代码也将在CUDA设备上运行，还需要为所有GPU设置随机数种子
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(78419)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dou Dizhu Evaluation')
    parser.add_argument('--player_1_bid', type=str, default='random',
                        help='First bidder: random, Supervised, or path to model')
    parser.add_argument('--player_2_bid', type=str, default='random')
    parser.add_argument('--player_3_bid', type=str, default='random')
    parser.add_argument('--player_1_playcard', type=str, required=True,
                        help='Landlord play model path (e.g. douzero_checkpoints/douzero/landlord_100000.ckpt)')
    parser.add_argument('--player_2_playcard', type=str, required=True,
                        help='Landlord_down play model path')
    parser.add_argument('--player_3_playcard', type=str, required=True,
                        help='Landlord_up play model path')
    parser.add_argument('--eval_data', type=str, default='eval_data.pkl')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--gpu_device', type=str, default='0')
    args = parser.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    evaluate(args.player_1_bid,
             args.player_2_bid,
             args.player_3_bid,
             args.player_1_playcard,
             args.player_2_playcard,
             args.player_3_playcard,
             args.eval_data,
             args.num_workers)
