"""
这里封装了原始环境以方便使用。当一局游戏结束时，
会自动重置环境，无需手动操作。
"""
import numpy as np
import torch

def _format_observation(obs, device):
    """
    处理观测数据并将其移动到 CUDA 的辅助函数。
    """
    position = obs['position']
    if not device == "cpu":
        device = 'cuda:' + str(device)
    device = torch.device(device)
    x_batch = torch.from_numpy(obs['x_batch']).to(device)
    z_batch = torch.from_numpy(obs['z_batch']).to(device)
    x_no_action = torch.from_numpy(obs['x_no_action'])
    z = torch.from_numpy(obs['z'])
    # oracle_labels: (B, 3) 逐动作标签 (B=候选动作数); 叫分阶段/异常回退为 (1, 3) dummy
    oracle_labels = torch.from_numpy(
        obs.get('oracle_labels', np.zeros((1, 3), dtype=np.float32)))
    # oracle_feat_batch: (B, 5) 逐动作神谕特征 (X 向量 C2), 叫分阶段为 None
    oracle_feat_batch = obs.get('oracle_feat_batch', None)
    if oracle_feat_batch is not None:
        oracle_feat_batch = torch.from_numpy(oracle_feat_batch)
    obs = {'x_batch': x_batch,
           'z_batch': z_batch,
           'legal_actions': obs['legal_actions'],
           # V2 Cross-Attention 拆分数据 (若存在)
           'state_z': torch.from_numpy(obs['state_z']).to(device) if 'state_z' in obs else None,
           'action_z': torch.from_numpy(obs['action_z']).to(device) if 'action_z' in obs else None,
           'history_z': torch.from_numpy(obs['history_z']).to(device) if 'history_z' in obs else None,
           }
    return position, obs, x_no_action, z, oracle_labels, oracle_feat_batch

class Environment:
    def __init__(self, env, device):
        """初始化此环境封装器。"""
        self.env = env
        self.device = device
        self.episode_return = None

    def initial(self, model, device, flags=None):
        obs = self.env.reset(model, device, flags=flags)
        initial_position, initial_obs, x_no_action, z, oracle_labels, oracle_feat_batch = \
            _format_observation(obs, self.device)
        self.episode_return = torch.zeros(1, 1)
        initial_done = torch.ones(1, 1, dtype=torch.bool)
        return initial_position, initial_obs, dict(
            done=initial_done,
            episode_return=self.episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            oracle_labels=oracle_labels,
            oracle_feat_batch=oracle_feat_batch,
        )

    def step(self, action, model, device, flags=None):
        obs, reward, done, draw, _ = self.env.step(action)

        self.episode_return = reward
        episode_return = self.episode_return
        if done:
            obs = self.env.reset(model, device, flags=flags)
            self.episode_return = torch.zeros(1, 1)
        if draw:
            obs = self.env.reset(model, device, flags=flags)
            self.episode_return = torch.zeros(1, 1)
        position, obs, x_no_action, z, oracle_labels, oracle_feat_batch = \
            _format_observation(obs, self.device)
        done = torch.tensor(done).view(1, 1)
        draw = torch.tensor(draw).view(1, 1)
        return position, obs, dict(
            draw=draw,
            done=done,
            episode_return=episode_return,
            obs_x_no_action=x_no_action,
            obs_z=z,
            oracle_labels=oracle_labels,
            oracle_feat_batch=oracle_feat_batch,
        )
