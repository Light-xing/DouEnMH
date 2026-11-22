"""
此文件包含 PyTorch 模型。将叫牌与出牌模型
封装到一个类中以方便使用。
"""
import math
import gc
import numpy as np
import torch.nn.functional as F
import torch
from torch import nn
from douzero.env.opponent_modeling import get_x_dim


class ChannelAttention(nn.Module):
    """SE 通道注意力 — 1D 自适应平均池化 + 两层 FC + Sigmoid 门控."""
    def __init__(self, channel, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class BasicBlockM(nn.Module):
    """ResNet 基础块 — Mish 激活 + SE 通道注意力, 用于叫分模型."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockM, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.mish = nn.Mish(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = ChannelAttention(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.mish(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += self.shortcut(x)
        out = self.mish(out)
        return out


class GeneralModelBid(nn.Module):
    """叫分模型 — ResNet 风格, 3 层下采样 + MLP, 输出 LWR/LPW/LPL 三头."""
    def __init__(self):
        super().__init__()
        self.in_planes = 2
        # input (B, 2, 120) → layer1 (B, 4, 60) → layer2 (B, 8, 30) → layer3 (B, 16, 15)
        self.layer1 = self._make_layer(BasicBlockM, 4, 3, stride=2)
        self.layer2 = self._make_layer(BasicBlockM, 8, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlockM, 16, 3, stride=2)
        conv_out_dim = 16 * BasicBlockM.expansion * 15  # 240
        x_dim = 3  # bid_info
        self.linear1 = nn.Linear(conv_out_dim + x_dim, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 3)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, z, x, return_value=False, flags=None):
        out = self.layer1(z)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.flatten(1, 2)
        out = torch.cat([out, x], dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = self.linear4(out)
        win_rate, win, lose = torch.split(out, (1, 1, 1), dim=-1)
        win_rate = torch.tanh(win_rate)
        _win_rate = (win_rate + 1) / 2
        out = _win_rate * win + (1. - _win_rate) * lose

        if return_value:
            return dict(values=(win_rate, win, lose))
        else:
            eps = flags.bid_exp_epsilon if flags is not None else 0.01
            if flags is not None and eps > 0 and np.random.rand() < eps:
                action = torch.randint(out.shape[0], (1,))[0]
            else:
                action = torch.argmax(out, dim=0)[0]
            return dict(action=action, max_value=torch.max(out), values=out)


class PositionalEncoding(nn.Module):
    """正弦位置编码 — 用于出牌历史序列 (35 手时序信息)."""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class GeneralModelTransformer(nn.Module):
    """DouEnMH V1 出牌模型 — 双塔 Transformer (State + History) + MLP 6 头.

    State 编码器: 5 行静态卡牌状态 (无位置编码, 无序).
    History 编码器: 35 手出牌历史 (带位置编码, 有序).
    """
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=1024):
        super(GeneralModelTransformer, self).__init__()

        # PointWiseConv2D：1×1 卷积在 8×15 牌面矩阵的 120 维展平向量上逐点投影
        self.state_embed = nn.Conv1d(120, d_model, kernel_size=1)
        self.action_embed = nn.Conv1d(120, d_model, kernel_size=1)

        # 位置编码仅用于动作历史（有序序列），长度 35 与 card_encoding.h 的 HISTORY_LENGTH 一致
        self.pos_encoder = PositionalEncoding(d_model, max_len=35)

        # 状态 Transformer 编码器 — 无位置编码（牌面特征无序）
        self.state_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_encoder_layers
        )

        # 动作 Transformer 编码器 — 带位置编码（历史有序）
        self.action_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_feedforward, batch_first=True),
            num_layers=num_encoder_layers
        )

        # 全连接头：状态(d_model) + 动作(d_model) + 辅助特征 x_batch
        # X: bid_info(3) + num_cards(3) + spring(1) + bomb_num(15)
        #    + hand_stats(7) + oracle_feat(5) + opponent_feat(4) + ctp_feat(14)
        #    + c5_feat(4) [if ENABLE_C5]
        # 使用 get_x_dim() 获取当前配置下的 X 维度 (52 或 56)
        fc_input_dim = d_model * 2 + get_x_dim()
        self.linear1 = nn.Linear(fc_input_dim, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        # 6 头: LWR(1) + LPW(1) + LPL(1) + LOR_a(1) + LOR_p(1) + LOR_d(1)
        self.linear4 = nn.Linear(128, 6)

    def forward(self, src1, src2, return_value=False, flags=None):
        # src1: (B, 41, 120) — 1 动作行 + 5 静态卡牌行 + 35 动作历史行
        # src2: (B, 52/56) — bid_info(3)+num_cards(3)+spring(1)+bomb_num(15)+hand_stats(7)+oracle_feat(5)+opponent_feat(4)+ctp_feat(14)+[c5_feat(4)]

        # 状态路径 (rows 0-5 = action + 5 card matrix rows)
        state_input = src1[:, :6].transpose(1, 2)    # (B, 120, 6)
        state_emb = self.state_embed(state_input)      # (B, d_model, 6)
        state_emb = state_emb.transpose(1, 2)           # (B, 6, d_model)
        state_out = self.state_encoder(state_emb)       # (B, 6, d_model)
        state_out = state_out.mean(dim=1)               # (B, d_model)

        # 动作历史路径 (rows 6-40 = 35 history rows)
        action_input = src1[:, -35:].transpose(1, 2)   # (B, 120, 35)
        action_emb = self.action_embed(action_input)     # (B, d_model, 35)
        action_emb = action_emb.transpose(1, 2)          # (B, 35, d_model)
        action_emb = self.pos_encoder(action_emb)         # (B, 35, d_model)
        action_out = self.action_encoder(action_emb)      # (B, 35, d_model)
        action_out = action_out.mean(dim=1)               # (B, d_model)

        # 拼接：状态特征 + 动作历史特征 + 辅助特征
        out = torch.cat([state_out, action_out, src2], dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = self.linear4(out)

        win_rate, win, lose, lor_a, lor_p, lor_d = torch.split(
            out, (1, 1, 1, 1, 1, 1), dim=-1)
        win_rate = torch.tanh(win_rate)
        _win_rate = (win_rate + 1) / 2
        out = _win_rate * win + (1. - _win_rate) * lose

        if return_value:
            return dict(values=(win_rate, win, lose),
                        oracle=(lor_a, lor_p, lor_d))
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(out.shape[0], (1,))[0]
            else:
                action = torch.argmax(out, dim=0)[0]
            return dict(action=action, max_value=torch.max(out), values=out)


class GeneralModelTransformerV2(nn.Module):
    """DouEnMH Cross-Attention 架构 — DouMH + Action Encoder + Cross-Attention.

    核心改进:
      - State Encoder:  5 行静态卡牌状态编码一次 (非 N 次重复)
      - Action Encoder: N 个候选动作独立投影 + 2 层 Transformer (1-token, 等效 FFN)
      - Cross-Attention: 单层 MultiheadAttention, Q=Action, KV=State
      - History Encoder: 35 行出牌历史 + 位置编码, 编码一次
      - 训练加速: 15-21× (消除 State/History 的 N 次冗余编码)
    """

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048):
        super().__init__()
        self.d_model = d_model

        # ── Projection 层 (120-dim 8×15 矩阵 → d_model) ──
        self.state_proj = nn.Conv1d(120, d_model, kernel_size=1)
        self.action_proj = nn.Conv1d(120, d_model, kernel_size=1)
        self.history_proj = nn.Conv1d(120, d_model, kernel_size=1)

        # ── 位置编码 (仅历史序列) ──
        self.pos_encoder = PositionalEncoding(d_model, max_len=35)

        # ── Transformer Encoders ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Action Encoder: 2 层轻量 Transformer (seq_len=1 时 self-attn 恒等, 等效 FFN)
        self.action_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # ── Cross-Attention (核心新增) ──
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)

        # ── MLP 头 ──
        fc_input_dim = d_model * 2 + get_x_dim()
        self.linear1 = nn.Linear(fc_input_dim, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 6)  # LWR/LPW/LPL/LOR_a/LOR_p/LOR_d

    def forward(self, state_z, action_z, history_z, x, return_value=False, flags=None):
        """
        state_z:   (1, 5, 120)    — batch=1, 5 行静态卡牌状态
        action_z:  (N, 1, 120)    — N 个候选动作, 每个 8×15 矩阵
        history_z: (1, 35, 120)   — 35 行出牌历史, batch=1
        x:         (N, X_dim)     — 每个候选动作的辅助特征
        """
        N = action_z.shape[0]
        d = self.d_model

        # ── 1. State Encoder (一次, 5 行静态状态) ──
        S = state_z.shape[1]                           # 5 (static state rows)
        state_in = state_z.transpose(1, 2)            # (1, 120, S)
        state_emb = self.state_proj(state_in)          # (1, d, S)
        state_emb = state_emb.transpose(1, 2)          # (1, S, d)
        state_out = self.state_encoder(state_emb)       # (1, S, d)
        state_out = state_out.expand(N, -1, -1)         # (N, S, d)

        # ── 2. Action Encoder (N 个候选动作, 并行) ──
        action_in = action_z.squeeze(1)               # (N, 120)
        action_in = action_in.unsqueeze(-1)            # (N, 120, 1)
        action_emb = self.action_proj(action_in)       # (N, d, 1)
        action_emb = action_emb.transpose(1, 2)        # (N, 1, d)
        action_emb = self.action_encoder(action_emb)    # (N, 1, d)

        # ── 3. Cross-Attention: Action attends State ──
        action_aware, _ = self.cross_attn(
            query=action_emb,                          # (N, 1, d)
            key=state_out,                             # (N, 5, d)
            value=state_out                            # (N, 5, d)
        )
        action_aware = action_aware.squeeze(1)          # (N, d)

        # ── 4. History Encoder (一次, 35 行 + 位置编码) ──
        hist_in = history_z.transpose(1, 2)             # (1, 120, 35)
        hist_emb = self.history_proj(hist_in)            # (1, d, 35)
        hist_emb = hist_emb.transpose(1, 2)              # (1, 35, d)
        hist_emb = self.pos_encoder(hist_emb)             # (1, 35, d)  + PE
        hist_out = self.history_encoder(hist_emb)         # (1, 35, d)
        hist_out = hist_out.mean(dim=1)                   # (1, d)
        hist_out = hist_out.expand(N, -1)                 # (N, d)

        # ── 5. Concat + MLP → 6 头 ──
        out = torch.cat([action_aware, hist_out, x], dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = self.linear4(out)

        win_rate, win, lose, lor_a, lor_p, lor_d = torch.split(
            out, (1, 1, 1, 1, 1, 1), dim=-1)
        win_rate = torch.tanh(win_rate)
        _win_rate = (win_rate + 1) / 2
        out = _win_rate * win + (1. - _win_rate) * lose

        if return_value:
            return dict(values=(win_rate, win, lose),
                        oracle=(lor_a, lor_p, lor_d))
        else:
            if flags is not None and flags.exp_epsilon > 0 and np.random.rand() < flags.exp_epsilon:
                action = torch.randint(out.shape[0], (1,))[0]
            else:
                action = torch.argmax(out, dim=0)[0]
            return dict(action=action, max_value=torch.max(out), values=out)


model_dict = {
    "first": GeneralModelBid,
    "second": GeneralModelBid,
    'third': GeneralModelBid,
    "landlord": GeneralModelTransformer,
    "landlord_down": GeneralModelTransformer,
    "landlord_up": GeneralModelTransformer,
}

model_dict_v2 = {
    "first": GeneralModelBid,
    "second": GeneralModelBid,
    'third': GeneralModelBid,
    "landlord": GeneralModelTransformerV2,
    "landlord_down": GeneralModelTransformerV2,
    "landlord_up": GeneralModelTransformerV2,
}


class Model:
    """
    六模型封装 (3 叫分 + 3 出牌) — share_memory / eval / 前向分发.

    通过 use_v2 参数在 V1 (双塔 Transformer) 和 V2 (Cross-Attention) 间切换.
    V2 模式接受拆分后的 (state_z, action_z, history_z, x) 输入.
    """

    def __init__(self, device=0, d_model=256, nhead=8, num_encoder_layers=6,
                 dim_feedforward=1024, use_v2=False):
        if not device == "cpu":
            device = 'cuda:' + str(device)
        self.use_v2 = use_v2

        transformer_args = dict(d_model=d_model, nhead=nhead,
                                num_encoder_layers=num_encoder_layers,
                                dim_feedforward=dim_feedforward)

        # 逐个构建模型，避免 6 个模型同时进行 CPU+GPU 拷贝。
        # 每次 .to(device) 创建 GPU 张量，而 CPU 原始数据在 GC 之前不会释放 —
        # 在模型之间清理缓存可防止峰值内存翻倍。
        self.models = {}
        gpu = torch.device(device)

        TransformerCls = GeneralModelTransformerV2 if use_v2 else GeneralModelTransformer

        for name, cls in [('first', GeneralModelBid),
                          ('second', GeneralModelBid),
                          ('third', GeneralModelBid),
                          ('landlord', TransformerCls),
                          ('landlord_down', TransformerCls),
                          ('landlord_up', TransformerCls)]:
            if cls is GeneralModelTransformer or cls is GeneralModelTransformerV2:
                m = cls(**transformer_args).to(gpu)
            else:
                m = cls().to(gpu)
            self.models[name] = m
            if device != 'cpu':
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                gc.collect()

    def forward(self, position, z, x, training=False, flags=None,
                state_z=None, action_z=None, history_z=None):
        """统一前向接口。

        V1 模式: 使用 (z, x), 忽略 state_z/action_z/history_z。
        V2 模式: 使用拆分后的 (state_z, action_z, history_z, x)。
                 叫分模型 (first/second/third) 在 V2 模式下仍使用 (z, x)。
        """
        model = self.models[position]
        if self.use_v2 and position not in ('first', 'second', 'third'):
            return model.forward(state_z, action_z, history_z, x,
                                 return_value=training, flags=flags)
        else:
            return model.forward(z, x, return_value=training, flags=flags)

    def share_memory(self):
        self.models['first'].share_memory()
        self.models['second'].share_memory()
        self.models['third'].share_memory()
        self.models['landlord'].share_memory()
        self.models['landlord_down'].share_memory()
        self.models['landlord_up'].share_memory()

    def eval(self):
        self.models['first'].eval()
        self.models['second'].eval()
        self.models['third'].eval()
        self.models['landlord'].eval()
        self.models['landlord_down'].eval()
        self.models['landlord_up'].eval()

    def parameters(self, position):
        return self.models[position].parameters()

    def get_model(self, position):
        return self.models[position]

    def get_models(self):
        return self.models

    def to_gpu(self, device):
        """将所有子模型逐个移动到 GPU，并在每次移动后清理缓存。"""
        gpu = torch.device('cuda:' + str(device))
        for name, m in self.models.items():
            self.models[name] = m.to(gpu)
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            gc.collect()



