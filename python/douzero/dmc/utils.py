import os
import typing
import logging
import traceback
import numpy as np
from collections import Counter
import time
from douzero.radam.radam import RAdam
import torch
from torch import multiprocessing as mp
from .env_utils import Environment
from douzero.env import Env
from douzero.env.opponent_modeling import get_x_dim

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

# v2 8×15 encoding constants
CARD_MATRIX_DIM = 120
NumOnes2ArrayV2 = {
    0: np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    1: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    2: np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
    3: np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=np.int8),
    4: np.array([1, 1, 1, 1, 0, 0, 0, 0], dtype=np.int8),
}
MIN_SINGLE_CHAIN = 5
MIN_PAIR_CHAIN = 3
MIN_TRIPLE_CHAIN = 2
MAX_CHAIN_RANK = 12    # 链类牌型可用 rank 数: 0-11 (3-A), 排除 2/joker

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# 缓冲区用于在 actor 进程和 learner 进程之间传输数据。
# 它们是 GPU 上的共享张量
Buffers = typing.Dict[str, typing.List[torch.Tensor]]


def create_env(flags):
    """创建 DMC 训练环境."""
    return Env(flags.objective)


def get_batch(b_queue, flags, stop_event=None):
    """从队列收集一个 batch 的训练样本 (在 unroll 维度 stack)。

    注意: 当前 learner (dmc.batch_and_learn) 已改为直接轮询各位置队列并自行拼批
    (用 torch.cat), 本函数不再被调用, 仅作为独立工具保留。
    """

    buffer = []
    while len(buffer) < flags.batch_size:
        if stop_event is not None and stop_event.is_set():
            raise StopIteration()
        try:
            buffer.append(b_queue.get(timeout=0.5))
        except Exception:
            if stop_event is not None and stop_event.is_set():
                raise StopIteration()
    batch = {
        key: torch.stack([m[key] for m in buffer], dim=1)
        for key in ["done", "episode_return", "target_adp", "target_wp",
                     "target_lor", "obs_z", "obs_x_batch"]
    }
    del buffer
    return batch


def create_optimizers(flags, learner_model):
    """为 6 个位置各创建一个 RAdam 优化器."""
    positions = ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']
    optimizers = {}
    for position in positions:
        optimizer = RAdam(
            learner_model.parameters(position),
            lr=flags.learning_rate,
            eps=flags.epsilon)
        optimizers[position] = optimizer
    return optimizers


def _decayed_epsilon(start_val, end_val, current_game, total_games):
    """探索率线性衰减: start_val → end_val over total_games 局."""
    if total_games <= 0 or current_game >= total_games:
        return end_val
    t = current_game / total_games
    return start_val + (end_val - start_val) * t


def act(i, device, batch_queues, model, flags, stop_event=None):
    """Actor 进程主循环 — 生成对局数据并通过队列发送给 Learner."""
    positions = ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']
    decay_games = getattr(flags, 'epsilon_decay_games', 0)
    # 探索率衰减的终点值必须在循环外固定下来:
    # 循环内每次都用衰减结果覆写 flags.exp_epsilon / flags.bid_exp_epsilon (模型 ε-greedy 读的就是它们),
    # 若把它们本身当作 end_val 传回 _decayed_epsilon, 首次调用后 start_val == end_val,
    # 衰减立即退化为不动点 (探索率被钉死在起始值).
    eps_end = flags.exp_epsilon
    bid_eps_end = flags.bid_exp_epsilon
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started.', str(device), i)

        # 预热：预分配 CUDA 上下文和 cuDNN 工作空间
        _x_dim = get_x_dim()
        if device != 'cpu':
            dev = torch.device('cuda:' + str(device))
            for pos in ['landlord', 'landlord_up', 'landlord_down']:
                if model.use_v2:
                    # V2: 拆分输入 (state_z, action_z, history_z, x)
                    dummy_state_z = torch.empty(1, 5, 120, device=dev)
                    dummy_action_z = torch.empty(2, 1, 120, device=dev)
                    dummy_history_z = torch.empty(1, 35, 120, device=dev)
                    dummy_x = torch.empty(2, _x_dim, device=dev)
                    with torch.no_grad():
                        model.get_model(pos).forward(dummy_state_z, dummy_action_z,
                                                      dummy_history_z, dummy_x,
                                                      return_value=True)
                else:
                    dummy_z = torch.empty(1, 41, 120, device=dev)
                    dummy_x = torch.empty(1, _x_dim, device=dev)
                    with torch.no_grad():
                        model.get_model(pos).forward(dummy_z, dummy_x, return_value=True)
            for pos in ['first', 'second', 'third']:
                dummy_z = torch.empty(1, 2, 120, device=dev)
                dummy_x = torch.empty(1, 3, device=dev)
                with torch.no_grad():
                    model.get_model(pos).forward(dummy_z, dummy_x, return_value=True)
            del dummy_z, dummy_x
            if model.use_v2:
                del dummy_state_z, dummy_action_z, dummy_history_z
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        model.eval()  # Actor 始终运行在 eval 模式（模型与 learner 共享）

        env = create_env(flags)
        env = Environment(env, device)

        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_adp_buf = {p: [] for p in positions}
        target_wp_buf = {p: [] for p in positions}
        target_lor_buf = {p: [] for p in positions}
        obs_z_buf = {p: [] for p in positions}
        size = {p: 0 for p in positions}
        obs_x_batch_buf = {p: [] for p in positions}

        position, obs, env_output = env.initial(model, device, flags=flags)

        step = 0
        game_count = 0
        t_game_start = time.time()
        while True:
            if stop_event is not None and stop_event.is_set():
                break
            while True:

                if len(obs['legal_actions']) > 1:
                    # 探索率线性衰减 (per-actor game_count 近似全局进度)
                    if decay_games > 0:
                        flags.exp_epsilon = _decayed_epsilon(
                            flags.exp_epsilon_start, eps_end, game_count, decay_games)
                        flags.bid_exp_epsilon = _decayed_epsilon(
                            flags.bid_exp_epsilon_start, bid_eps_end, game_count, decay_games)
                    with torch.inference_mode():
                        agent_output = model.forward(
                            position, obs['z_batch'], obs['x_batch'], flags=flags,
                            state_z=obs.get('state_z'),
                            action_z=obs.get('action_z'),
                            history_z=obs.get('history_z'))
                    _action_idx = int(agent_output['action'].cpu().detach().numpy())
                    action = obs['legal_actions'][_action_idx]
                    del agent_output
                    # 释放推理用 GPU 张量，降低 actor CUDA allocator 碎片压力
                    del obs['z_batch'], obs['x_batch']
                    if obs.get('state_z') is not None:
                        del obs['state_z'], obs['action_z'], obs['history_z']
                else:
                    _action_idx = 0
                    action = obs['legal_actions'][0]

                # 叫分课程: 训练初期 P0 偏向叫 0 分, 给 P1/P2 留出叫分空间
                # 概率从 1.0 线性衰减到 0.0, 超出 bid_curriculum_games 局后关闭
                if (flags.bid_curriculum_games > 0
                        and position == 'first'
                        and game_count < flags.bid_curriculum_games):
                    p_zero = 1.0 - game_count / flags.bid_curriculum_games
                    if np.random.rand() < p_zero:
                        zero_actions = [a for a in obs['legal_actions'] if a[0] == 0]
                        if zero_actions:
                            action = zero_actions[0]

                if position in ['first', 'second', 'third']:
                    obs_z_buf[position].append(
                        torch.vstack((torch.full((1, CARD_MATRIX_DIM), action[0], dtype=torch.int8), env_output['obs_z'])))
                else:
                    obs_z_buf[position].append(
                        torch.vstack((_cards2tensor_v2(action).unsqueeze(0), env_output['obs_z'])))

                x_batch = env_output['obs_x_no_action'].float()
                # 逐动作神谕: 将实际执行动作的神谕特征注入 x_no_action 的 oracle 偏移处
                # (X 布局: bid(3)+cards(3)+spring(1)+bomb(15)+hand_stats(7)=29 + oracle(5))
                if position not in ('first', 'second', 'third'):
                    oracle_feat_batch = env_output.get('oracle_feat_batch', None)
                    if oracle_feat_batch is not None:
                        safe_idx = min(_action_idx, oracle_feat_batch.shape[0] - 1)
                        x_batch[29:34] = oracle_feat_batch[safe_idx]
                obs_x_batch_buf[position].append(x_batch)
                # 存储神谕监督标签 (仅出牌阶段有效, 叫分阶段使用 dummy 零值)
                # oracle_labels 现在是 (B, 3) 逐动作标签, 取实际执行动作的标签
                oracle_labels = env_output.get('oracle_labels', torch.zeros(1, 3))
                safe_idx = min(_action_idx, oracle_labels.shape[0] - 1)
                target_lor_buf[position].append(oracle_labels[safe_idx])
                size[position] += 1
                step += 1

                position, obs, env_output = env.step(action, model, device, flags=flags)

                if stop_event is not None and stop_event.is_set():
                    break

                if env_output['done'] or env_output['draw']:
                    game_count += 1
                    # 每 50 局释放一次 actor CUDA allocator 缓存碎片
                    # 12 个 actor 进程各自持有独立 allocator, 碎片随时间累积
                    if device != 'cpu' and game_count % 50 == 0:
                        torch.cuda.empty_cache()
                    dt_game = time.time() - t_game_start
                    if flags.log_game_steps:
                        log.info('Game %d finished: %d steps in %.1fs (%.0f step/s)',
                                 game_count, step, dt_game,
                                 step / dt_game if dt_game > 0 else 0)
                    t_game_start = time.time()
                    for p in positions:
                        diff = size[p] - len(target_adp_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff - 1)])
                            done_buf[p].append(True)
                            if env_output['draw']:
                                episode_return = 0.
                                wp_return = 0.
                            else:
                                episode_return = env_output['episode_return']["play"][p]
                                wp_return = 1. if episode_return > 0. else -1.
                            episode_return_buf[p].extend([0.0 for _ in range(diff - 1)])
                            episode_return_buf[p].append(episode_return)
                            target_adp_buf[p].extend([episode_return * flags.decay ** (diff - n) for n in range(diff)])
                            target_wp_buf[p].extend([wp_return * flags.decay ** (diff - n) for n in range(diff)])
                    break
            if stop_event is not None and stop_event.is_set():
                continue
            for p in positions:
                if size[p] > T:
                    if stop_event is not None and stop_event.is_set():
                        break
                    data = {
                        "done": torch.stack([torch.tensor(ndarr, device="cpu") for ndarr in done_buf[p][:T]]),
                        "episode_return": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in episode_return_buf[p][:T]]),
                        "target_adp": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in target_adp_buf[p][:T]]),
                        "target_wp": torch.stack(
                            [torch.tensor(ndarr, device="cpu") for ndarr in target_wp_buf[p][:T]]),
                        "target_lor": torch.stack(
                            [ndarr.detach() for ndarr in target_lor_buf[p][:T]]),
                        "obs_z": torch.stack([ndarr.detach() for ndarr in obs_z_buf[p][:T]]),
                        "obs_x_batch": torch.stack(
                            [ndarr.detach() for ndarr in obs_x_batch_buf[p][:T]]),
                    }
                    while True:
                        if stop_event is not None and stop_event.is_set():
                            break
                        try:
                            batch_queues[p].put(data, timeout=0.5)
                            break
                        except Exception:
                            if stop_event is not None and stop_event.is_set():
                                break
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_adp_buf[p] = target_adp_buf[p][T:]
                    target_wp_buf[p] = target_wp_buf[p][T:]
                    target_lor_buf[p] = target_lor_buf[p][T:]
                    obs_x_batch_buf[p] = obs_x_batch_buf[p][T:]
                    obs_z_buf[p] = obs_z_buf[p][T:]
                    size[p] -= T

            # 每 50 局执行一次 empty_cache() 以释放 actor CUDA allocator 碎片.
            # 碎片主要来自变长推理 (legal_actions 数量变化导致 action_z 维度抖动).
            # 过于频繁的 empty_cache 会强制重新分配, 50 局是平衡点.

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
    finally:
        # 释放 actor 本地的 CUDA 张量引用，确保进程退出时 IPC 句柄被正确清理。
        for buf in [done_buf, episode_return_buf, target_adp_buf,
                     target_wp_buf, target_lor_buf, obs_z_buf, obs_x_batch_buf]:
            for k in list(buf.keys()):
                buf[k].clear()
        # 防止 mp.Queue 后台 feeder 线程阻塞进程退出
        #（Windows 上 multiprocessing 的已知问题）。
        for q in batch_queues.values():
            q.cancel_join_thread()
        # 释放共享模型引用以防止 CUDA IPC 警告
        del model
        del env


def _cards2tensor_v2(list_cards):
    """DouEnMH 8×15 矩阵编码 → 120-dim tensor."""
    if len(list_cards) == 0:
        return torch.zeros(CARD_MATRIX_DIM, dtype=torch.int8)

    matrix = np.zeros([8, 15], dtype=np.int8)
    counter = Counter(list_cards)

    # Row 0-3: 计数行
    for card, n in counter.items():
        if card < 20:
            col = Card2Column[card]
        elif card == 20:
            col = 13
        else:
            col = 14
        matrix[0:4, col] = NumOnes2ArrayV2[min(n, 4)][0:4]

    # Row 4: 断点标记
    ranks_present = set(Card2Column[c] for c in list_cards if c < 20)
    for i in range(14):
        if i in ranks_present and (i + 1) not in ranks_present:
            matrix[4, i] = 1

    # Row 5: 单顺起点 (≥5 连, 0..7, 链不入 rank 12+)
    for i in range(MAX_CHAIN_RANK - MIN_SINGLE_CHAIN + 1):
        if all(matrix[0, i + k] == 1 for k in range(MIN_SINGLE_CHAIN)):
            matrix[5, i] = 1

    # Row 6: 连对起点 (≥3 连对, 0..9, 链不入 rank 12+)
    for i in range(MAX_CHAIN_RANK - MIN_PAIR_CHAIN + 1):
        if all(matrix[1, i + k] == 1 for k in range(MIN_PAIR_CHAIN)):
            matrix[6, i] = 1

    # Row 7: 飞机起点 (≥2 连三条, 0..10, 链不入 rank 12+)
    for i in range(MAX_CHAIN_RANK - MIN_TRIPLE_CHAIN + 1):
        if all(matrix[2, i + k] == 1 for k in range(MIN_TRIPLE_CHAIN)):
            matrix[7, i] = 1

    return torch.from_numpy(matrix.flatten('F'))


