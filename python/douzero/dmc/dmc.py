import os
import threading
import time
import timeit
import pprint
import gc
from collections import deque
import numpy as np

import torch
from torch import multiprocessing as mp
mp.set_sharing_strategy('file_system')
from torch import nn

import douzero.dmc.models
from douzero.dmc.models import GeneralModelTransformerV2
import douzero.env.env
import douzero.env.numba_utils  # Numba JIT 热点函数加速 (自动 patch)
from .file_writer import FileWriter
from .models import Model

from .utils import log, create_env, create_optimizers, act

mean_episode_return_buf = {p: deque(maxlen=50) for p in
                           ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']}


def compute_loss(logits, targets):
    loss = ((logits.squeeze(-1) - targets) ** 2).mean()
    return loss



def _get_oracle_lambda(lambda_init, global_step, T_decay, lambda_min=0.03):
    """LOR loss weight 线性衰减调度。

    设计依据 oracle_heads_design.md §2.3 与 oracle_post_action_design.md:
      λ(t) = max(λ_min, λ_init · (1 − t / T_decay))

    T_decay = total_frames * oracle_decay_ratio（--oracle_decay_ratio 默认 0.2，
    即前 20% 训练帧内线性衰减到 lambda_min）。lambda_min 默认 0.03，与
    arguments.py 的 --oracle_lambda_min 默认值一致（调用方总是显式传入 flags 值）。
    """
    if global_step >= T_decay:
        return lambda_min
    return max(lambda_min, lambda_init * (1.0 - global_step / T_decay))


def learn(position, model, batch, optimizer, flags, lock, accum_step=0, accum_total=1,
          global_step=None):
    """执行一个学习（优化）步骤。model 与 actor 进程共享。

    当 accum_total > 1 时，梯度会在多个批次上累积
    再执行一次优化器更新——相当于在不增加 VRAM 开销的情况下使用更大的有效批大小。
    只有最后一个累积步骤会裁剪梯度并更新权重；中间步骤仅执行前向+反向传播。"""
    if flags.training_device != "cpu":
        device = torch.device('cuda:' + str(flags.training_device))
    else:
        device = torch.device('cpu')
    obs_x = batch["obs_x_batch"].to(device)
    obs_z = batch['obs_z'].to(device).float()
    target_adp = batch['target_adp'].to(device)
    target_wp = batch['target_wp'].to(device)
    target_lor = batch['target_lor'].to(device)  # (T*B, 3)
    episode_returns = batch['episode_return'][batch['done']]
    if len(episode_returns) > 0:
        mean_episode_return_buf[position].append(torch.mean(episode_returns).item())
    with lock:
        model.train()
        # V2 Cross-Attention: model 是单个 nn.Module (由 Model.get_model 返回),
        # 需区分 V1/V2 调用签名
        is_v2 = isinstance(model, GeneralModelTransformerV2)
        if is_v2:
            # obs_z shape: (T*B, 41, 120) = action(1) + static(5) + history(35)
            state_z = obs_z[:, 1:6, :]     # (T*B, 5, 120) — only 5 static state rows
            action_z = obs_z[:, 0:1, :]    # (T*B, 1, 120)
            history_z = obs_z[:, 6:, :]    # (T*B, 35, 120)
            result = model.forward(state_z, action_z, history_z, obs_x, return_value=True)
        else:
            result = model.forward(obs_z, obs_x, return_value=True)
        win_rate, win, lose = result['values']
        # LWR: win rate prediction (Eq. 2)
        loss1 = compute_loss(win_rate, target_wp)
        # Soft weight based on predicted win rate (Eq. 3 & 4)
        _wr = (win_rate + 1.0) / 2.0
        # LPW: scaled win head predicts raw ADP (Eq. 3)
        loss2 = compute_loss(_wr * win, target_adp)
        # LPL: scaled lose head predicts raw ADP (Eq. 4)
        loss3 = compute_loss((1.0 - _wr) * lose, target_adp)
        loss = loss1 + loss2 + loss3
        # LOR: oracle auxiliary heads — only for play positions (bidding models lack oracle heads)
        lor_a = lor_p = lor_d = None
        loss_lor_a = loss_lor_p = loss_lor_d = None
        if 'oracle' in result:
            lor_a, lor_p, lor_d = result['oracle']
            # Oracle λ 线性衰减: 前 oracle_decay_ratio (默认 20%) 训练帧数内
            # n_active: 1.0 → 0.03,  n_passive/ΔN: 0.6 → 0.03
            step = global_step if global_step is not None else 0
            T_decay = int(flags.total_frames * getattr(flags, 'oracle_decay_ratio', 0.2))
            lambda_min = getattr(flags, 'oracle_lambda_min', 0.03)
            la = _get_oracle_lambda(
                getattr(flags, 'oracle_lambda_landlord', 1.0), step, T_decay, lambda_min)
            lp = _get_oracle_lambda(
                getattr(flags, 'oracle_lambda_peasant', 0.6), step, T_decay, lambda_min)
            ld = _get_oracle_lambda(
                getattr(flags, 'oracle_lambda_peasant', 0.6), step, T_decay, lambda_min)
            loss_lor_a = compute_loss(lor_a.squeeze(-1), target_lor[:, 0]) * la
            loss_lor_p = compute_loss(lor_p.squeeze(-1), target_lor[:, 1]) * lp
            loss_lor_d = compute_loss(lor_d.squeeze(-1), target_lor[:, 2]) * ld
            loss = loss + loss_lor_a + loss_lor_p + loss_lor_d
        loss = loss / accum_total

        buf = mean_episode_return_buf[position]
        mean_return = sum(buf) / len(buf) if len(buf) > 0 else 0.0
        stats = {
            'mean_episode_return_' + position: mean_return,
            'loss_' + position: loss.item() * accum_total,  # report unscaled loss
        }
        loss.backward()

        is_last = (accum_step == accum_total - 1)
        if is_last:
            nn.utils.clip_grad_norm_(model.parameters(), flags.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    # Explicitly release GPU tensors to prevent memory accumulation
    del obs_x, obs_z, target_adp, target_wp, target_lor, episode_returns
    del win_rate, win, lose, _wr, loss1, loss2, loss3, loss
    del result, lor_a, lor_p, lor_d, loss_lor_a, loss_lor_p, loss_lor_d
    if is_v2:
        del state_z, action_z, history_z
    return stats, is_last


def train(flags):
    """
    训练主函数。首先初始化所有组件（缓冲区、优化器等），
    然后启动子进程作为 actor，最后调用学习函数进行多线程训练。
    """
    if not flags.actor_device_cpu or flags.training_device != 'cpu':
        if not torch.cuda.is_available():
            raise AssertionError(
                "CUDA not available. If you have GPUs, please specify the ID after `--gpu_devices`. Otherwise, please train with CPU with `python3 train.py --actor_device_cpu --training_device cpu`")
    plogger = FileWriter(
        xpid=flags.xpid,
        xp_args=flags.__dict__,
        rootdir=flags.savedir,
    )
    checkpointpath = os.path.expandvars(
        os.path.expanduser('%s/%s/%s' % (flags.savedir, flags.xpid, 'model.tar')))

    T = flags.unroll_length
    B = flags.batch_size

    if flags.actor_device_cpu:
        device_iterator = ['cpu']
    else:
        device_iterator = range(flags.num_actor_devices)
        assert flags.num_actor_devices <= len(
            flags.gpu_devices.split(',')), 'The number of actor devices can not exceed the number of available devices'

    # Architecture args passed to all models
    model_kwargs = dict(
        d_model=flags.d_model,
        nhead=flags.nhead,
        num_encoder_layers=flags.num_encoder_layers,
        dim_feedforward=flags.dim_feedforward,
        use_v2=getattr(flags, 'use_v2', False),
    )

    def _clear_gpu():
        """释放 GPU 缓存并执行垃圾回收。"""
        if flags.training_device != 'cpu':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        gc.collect()

    # --- 阶段 1：在 GPU 上创建共享模型。
    #     Actor 和 Learner 共享同一个模型。
    models = {}
    for device in device_iterator:
        model = Model(device=device, **model_kwargs)
        model.share_memory()
        model.eval()
        models[device] = model
    _clear_gpu()

    optimizers = create_optimizers(flags, models[list(device_iterator)[0]])

    # --- Phase 4: Initialize queues ---
    actor_processes = []
    ctx = mp.get_context('spawn')
    stop_event = ctx.Event()
    batch_queues = {}
    for device in device_iterator:
        batch_queue = {"first": ctx.Queue(), "second": ctx.Queue(), "third": ctx.Queue(),
                       "landlord": ctx.Queue(), "landlord_up": ctx.Queue(),
                       "landlord_down": ctx.Queue()}
        batch_queues[device] = batch_queue

    # Stat Keys
    stat_keys = [
        'mean_episode_return_first',
        'loss_first',
        'mean_episode_return_second',
        'loss_second',
        'mean_episode_return_third',
        'loss_third',
        'mean_episode_return_landlord',
        'loss_landlord',
        'mean_episode_return_landlord_up',
        'loss_landlord_up',
        'mean_episode_return_landlord_down',
        'loss_landlord_down',
    ]
    frames, stats = 0, {k: 0 for k in stat_keys}
    position_frames = {'first': 0, 'second': 0, 'third': 0, 'landlord': 0, 'landlord_up': 0, 'landlord_down': 0}

    # Load models if any
    if flags.load_model and os.path.exists(checkpointpath):
        checkpoint_states = torch.load(
            checkpointpath,
            map_location='cpu'
        )

        # 6 个位置的权重逐一加载; 任一位置与当前模型结构不匹配 → 默认**硬报错**。
        # 静默跳过会让该位置从头训练, 在高成本训练中不易察觉 (且原实现因 and/or
        # 优先级问题, 连叫分位置也会被静默跳过)。
        # 确需在架构变更后继续训练时, 显式加 --allow_partial_load 豁免这些位置。
        bid_positions = ['first', 'second', 'third']
        play_positions = ['landlord', 'landlord_up', 'landlord_down']

        allow_partial = getattr(flags, 'allow_partial_load', False)
        reset_positions = []
        for k in bid_positions + play_positions:
            try:
                models[list(device_iterator)[0]].get_model(k).load_state_dict(checkpoint_states["model_state_dict"][k])
                optimizers[k].load_state_dict(checkpoint_states["optimizer_state_dict"][k])
            except RuntimeError as e:
                msg = str(e)
                arch_mismatch = ('missing keys' in msg) or ('unexpected keys' in msg)
                if arch_mismatch and allow_partial:
                    reset_positions.append(k)
                    log.warning('Architecture mismatch for %s — this position restarts from scratch '
                                '(--allow_partial_load): %s', k, msg.replace('\n', ' ')[:300])
                else:
                    raise RuntimeError(
                        'Checkpoint architecture mismatch for position "%s".\n%s\n'
                        '该位置的权重与当前模型结构不一致 (常见原因: --use_v2 切换、'
                        '--d_model / --nhead / --num_encoder_layers / --dim_feedforward 改动、'
                        'DOUENMH_ENABLE_C5 与训练时不同)。\n'
                        '若确认要让该位置从头训练, 请显式加 --allow_partial_load。' % (k, msg)
                    ) from e
        if reset_positions:
            log.warning('Positions restarting from scratch: %s', ', '.join(reset_positions))

        stats = checkpoint_states["stats"]
        frames = checkpoint_states["frames"]
        position_frames = checkpoint_states["position_frames"]
        log.info(f"Resuming preempted job, current stats:\n{stats}")
        del checkpoint_states
        _clear_gpu()

    def batch_and_learn():
        """Learner 单线程：非阻塞轮询所有 6 个位置队列。

        排空所有队列，然后对所有有数据的位置各训练最多 K=gradient_accumulation_steps 个批次。
        连续的前向/反向传播让 GPU 在优化器步骤之间保持忙碌，而不是每批都切换位置。
        叫分位置使用较小的 bid_batch_size，避免数据积累等待过久。"""
        nonlocal frames, position_frames, stats
        positions = ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']
        bid_positions = {'first', 'second', 'third'}
        device = list(device_iterator)[0]
        K = flags.gradient_accumulation_steps
        play_bs = flags.batch_size
        bid_bs = getattr(flags, 'bid_batch_size', max(4, play_bs // 4))

        buffers = {p: [] for p in positions}

        while frames < flags.total_frames:
            # 阶段 1：将所有队列排空到暂存缓冲区（非阻塞）
            for position in positions:
                while True:
                    if stop_event.is_set():
                        return
                    try:
                        buffers[position].append(
                            batch_queues[device][position].get(timeout=0.01))
                    except Exception:
                        break

            # 阶段 2：对所有有数据的位置各训练最多 K 个批次
            trained = False
            for position in sorted(positions, key=lambda p: len(buffers[p]), reverse=True):
                bs = bid_bs if position in bid_positions else play_bs
                available = len(buffers[position]) // bs
                if available == 0:
                    continue

                n_batches = min(available, K)
                model_p = models[device].get_model(position)
                opt = optimizers[position]

                for step_idx in range(n_batches):
                    start = step_idx * bs
                    end = start + bs
                    batch = {
                        key: torch.cat([m[key] for m in buffers[position][start:end]])
                        for key in ["done", "episode_return", "target_adp", "target_wp",
                                     "target_lor", "obs_z", "obs_x_batch"]
                    }
                    _stats, _ = learn(position, model_p, batch, opt, flags, learn_lock,
                                      step_idx, n_batches, global_step=frames)
                    del batch

                buffers[position] = buffers[position][n_batches * bs:]

                for k in _stats:
                    stats[k] = _stats[k]
                to_log = dict(frames=frames)
                to_log.update({k: stats[k] for k in stat_keys})
                plogger.log(to_log)
                frames += T * bs * n_batches
                position_frames[position] += T * bs * n_batches
                trained = True

            if not trained:
                time.sleep(0.01)

    learn_lock = threading.Lock()
    learner_thread = threading.Thread(target=batch_and_learn, name='learner')
    learner_thread.start()

    # 启动 actor 进程（错开启动以避免 CUDA 上下文初始化并发峰值）
    for device in device_iterator:
        for i in range(flags.num_actors):
            actor = ctx.Process(
                target=act,
                args=(i, device, batch_queues[device], models[device], flags, stop_event))
            actor.start()
            actor_processes.append(actor)
            time.sleep(0.5)  # Let each actor's CUDA context settle before next one
    _clear_gpu()

    def _cpu_sd(obj):
        """递归地将嵌套字典/列表中的所有张量移动到 CPU。"""
        if isinstance(obj, torch.Tensor):
            return obj.cpu()
        if isinstance(obj, dict):
            return {k: _cpu_sd(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_cpu_sd(v) for v in obj]
        return obj

    def checkpoint(frames, final=False):
        if flags.disable_checkpoint:
            return
        log.info('Saving checkpoint to %s', checkpointpath)

        # 对于定期检查点，暂停 learner 线程以防止
        # 并发前向/反向传播和保存导致的 GPU 内存尖峰。
        # 对于最终检查点，线程已在停止——无需加锁。
        if not final:
            if not learn_lock.acquire(timeout=30):
                log.warning('Could not pause learner within 30s, saving anyway')

        try:
            shared = models[list(device_iterator)[0]]
            _models = shared.get_models()

            # 将 state_dict 张量移到 CPU，这样 torch.save() 就不会分配
            # GPU 暂存缓冲区——避免约 7GB 的检查点内存尖峰。
            cpu_model_sd = _cpu_sd({k: v.state_dict() for k, v in _models.items()})
            cpu_optim_sd = _cpu_sd({k: v.state_dict() for k, v in optimizers.items()})

            torch.save({
                'model_state_dict': cpu_model_sd,
                'optimizer_state_dict': cpu_optim_sd,
                "stats": stats,
                'flags': vars(flags),
                'frames': frames,
                'position_frames': position_frames
            }, checkpointpath, _use_new_zipfile_serialization=False)

            # 保存每个位置的权重——重用已复制到 CPU 的张量
            for position in ['first', 'second', 'third', 'landlord', 'landlord_up', 'landlord_down']:
                model_weights_dir = os.path.expandvars(os.path.expanduser(
                    '%s/%s/%s' % (flags.savedir, flags.xpid, position + '_' + str(frames) + '.ckpt')))
                torch.save(cpu_model_sd[position], model_weights_dir, _use_new_zipfile_serialization=False)
            del cpu_model_sd, cpu_optim_sd
        finally:
            if not final:
                learn_lock.release()

        if flags.training_device != 'cpu':
            torch.cuda.empty_cache()
        gc.collect()

    fps_log = []
    timer = timeit.default_timer
    try:
        last_checkpoint_time = timer()  # First checkpoint after save_interval, not immediately
        while frames < flags.total_frames:
            start_frames = frames
            position_start_frames = {k: position_frames[k] for k in position_frames}
            start_time = timer()
            time.sleep(5)

            if timer() - last_checkpoint_time > flags.save_interval * 60:
                checkpoint(frames)
                last_checkpoint_time = timer()

            # 仅在真正接近 OOM（分配 >90%）时紧急清理缓存。
            # 定期调用 empty_cache() 适得其反——它会丢弃刚被重新分配的缓存块，导致分配器抖动。
            if flags.training_device != 'cpu':
                device_idx = int(flags.training_device)
                total_mem = torch.cuda.get_device_properties(device_idx).total_memory
                allocated_ratio = torch.cuda.memory_allocated(device_idx) / total_mem
                if allocated_ratio > 0.90:
                    log.warning('GPU allocated memory at %.1f%%, emergency cache clear', allocated_ratio * 100)
                    torch.cuda.empty_cache()

            end_time = timer()

            fps = (frames - start_frames) / (end_time - start_time)
            fps_log.append(fps)
            if len(fps_log) > 24:
                fps_log = fps_log[1:]
            fps_avg = np.mean(fps_log)

            position_fps = {k: (position_frames[k] - position_start_frames[k]) / (end_time - start_time) for k in
                            position_frames}
            log.info('After %i (L:%i U:%i D:%i) frames: @ %.1f fps (avg@ %.1f fps) (L:%.1f U:%.1f D:%.1f) Stats:\n%s',
                     frames,
                     position_frames['landlord'],
                     position_frames['landlord_up'],
                     position_frames['landlord_down'],
                     fps,
                     fps_avg,
                     position_fps['landlord'],
                     position_fps['landlord_up'],
                     position_fps['landlord_down'],
                     pprint.pformat(stats))

    except KeyboardInterrupt:
        pass
    else:
        log.info('Learning finished after %d frames.', frames)

    stop_event.set()

    # Learner 线程快速退出：batch_and_learn 在每轮轮询中直接检查 stop_event。
    learner_thread.join(timeout=10)

    # 最终检查点：learner 已停止，无锁竞争。
    checkpoint(frames, final=True)

    # Actor 清理：actor 现在应快速退出，因为队列已在 finally 块中调用了 cancel_join_thread()。
    for actor in actor_processes:
        actor.join(timeout=5)
        if actor.is_alive():
            log.warning('Actor did not exit within 5s, terminating')
            actor.terminate()
    if flags.training_device != 'cpu':
        torch.cuda.ipc_collect()
    plogger.close()
