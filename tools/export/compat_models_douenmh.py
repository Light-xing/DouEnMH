"""
DouEnMH PyTorch 1.5.0 兼容模型定义（V1 出牌模型 + 叫分模型）。

与 douzero/dmc/models.py 的差异:
1. nn.Mish → 自定义 mish(x) = x * tanh(softplus(x))
2. batch_first=True → 手动 transpose (S,B,E) ↔ (B,S,E)
3. forward 只保留推理分支并直接返回 Q 张量；训练端 forward 还有
   return_value / flags 参数与 dict 返回值，导出用不到，脚本化也会更复杂
4. 无其他结构变化 — state_dict key 名完全一致, 可直接 load_state_dict

不包含 GeneralModelTransformerV2（Cross-Attention 架构），其 state_dict 使用
state_proj / action_proj / history_proj / cross_attn / history_encoder 等键，
与这里的 state_embed / action_embed 不同，不能加载进本文件的模型。

用法:
    from compat_models_douenmh import GeneralModelBid, GeneralModelTransformer
    sd = torch.load('landlord_620000000.ckpt', map_location='cpu')
    arch = detect_play_architecture(sd)
    model = GeneralModelTransformer(**arch)
    model.load_state_dict(sd)
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

# C5 feature dimension (4 when DOUENMH_ENABLE_C5=1, default enabled)
_C5_DIM = 4


def get_x_dim(enable_c5=True):
    return 52 + (_C5_DIM if enable_c5 else 0)


def detect_play_architecture(state_dict, nhead=8):
    """从 checkpoint state_dict 自动检测 DouEnMH V1 出牌模型架构参数。

    检测规则:
      d_model:       state_embed.weight.shape[0]  (Conv1d out_channels)
      num_layers:    计数 state_encoder.layers.{i}.self_attn.in_proj_weight
      dim_feedforward: state_encoder.layers.0.linear1.weight.shape[0]
      nhead:         无法从权重可靠推断, 需与训练配置一致, 默认 8
      x_dim:         linear1.weight.shape[1] - d_model*2

    参数:
      state_dict: checkpoint 的 state_dict
      nhead: 注意力头数 (无法从权重推断, 需匹配训练配置)
    """
    sd = state_dict
    d_model = sd['state_embed.weight'].shape[0]

    num_layers = 0
    while f'state_encoder.layers.{num_layers}.self_attn.in_proj_weight' in sd:
        num_layers += 1

    dim_ff = sd['state_encoder.layers.0.linear1.weight'].shape[0]

    # x_dim: 从 MLP 头推断 (fc_input_dim = d_model*2 + x_dim)
    fc_input_dim = sd['linear1.weight'].shape[1]
    x_dim = fc_input_dim - d_model * 2

    return dict(d_model=d_model, nhead=nhead, num_encoder_layers=num_layers,
                dim_feedforward=dim_ff, x_dim=x_dim)


# ---------------------------------------------------------------------------
# 自定义 Mish (PyTorch 1.9+ 才有 nn.Mish)
# ---------------------------------------------------------------------------
def mish(x):
    return x * torch.tanh(F.softplus(x))


# ---------------------------------------------------------------------------
# ChannelAttention — 无需修改 (1.5 支持)
# ---------------------------------------------------------------------------
class ChannelAttention(nn.Module):
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


# ---------------------------------------------------------------------------
# BasicBlockM — 将 nn.Mish → mish()
# ---------------------------------------------------------------------------
class BasicBlockM(nn.Module):
    """与训练模型结构一致, 仅 nn.Mish → mish() 函数."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockM, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
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
        out = mish(self.bn1(self.conv1(x)))
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += self.shortcut(x)
        out = mish(out)
        return out


# ---------------------------------------------------------------------------
# GeneralModelBid — 叫分模型 (first / second / third)
# ---------------------------------------------------------------------------
class GeneralModelBid(nn.Module):
    """DouEnMH 叫分模型 — ResNet + MLP, 输出 LWR/LPW/LPL 三头.

    state_dict 与训练端 douzero/dmc/models.py:GeneralModelBid 完全兼容.
    """

    def __init__(self):
        super().__init__()
        self.in_planes = 2
        # input (B, 2, 120) → layer1 (B, 4, 60) → layer2 (B, 8, 30) → layer3 (B, 16, 15)
        self.layer1 = self._make_layer(BasicBlockM, 4, 3, stride=2)
        self.layer2 = self._make_layer(BasicBlockM, 8, 3, stride=2)
        self.layer3 = self._make_layer(BasicBlockM, 16, 3, stride=2)
        conv_out_dim = 16 * BasicBlockM.expansion * 15  # 240
        x_dim = 3  # bid_info only, hardcoded in training model
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

    def forward(self, z, x):
        """
        z: (B, 2, 120)  — [action_encoding(1), my_handcards(1)]
        x: (B, 3)       — bid_info
        返回: (B, 1)     — Q-value (ŵ·LPW + (1-ŵ)·LPL)
        """
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
        return out


# ---------------------------------------------------------------------------
# PositionalEncoding — 无需修改 (纯数学运算)
# ---------------------------------------------------------------------------
class PositionalEncoding(nn.Module):
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


# ---------------------------------------------------------------------------
# GeneralModelTransformer — DouEnMH V1 出牌模型
#
# 关键修改 vs 训练端:
#   1. batch_first=True → 手动 transpose (PyTorch 1.5.0 不支持 batch_first)
#   2. nn.Mish → mish() (在 BasicBlockM 中已处理, 此处模型不使用 Mish)
# ---------------------------------------------------------------------------
class GeneralModelTransformer(nn.Module):
    """DouEnMH V1 出牌模型 — 双塔 Transformer (State + History) + MLP 6 头.

    state_dict 与训练端 douzero/dmc/models.py:GeneralModelTransformer 完全兼容.
    """

    def __init__(self, d_model=384, nhead=8, num_encoder_layers=6,
                 dim_feedforward=256, x_dim=56):
        super(GeneralModelTransformer, self).__init__()

        # PointWiseConv2D：1×1 卷积投影 120-dim → d_model
        self.state_embed = nn.Conv1d(120, d_model, kernel_size=1)
        self.action_embed = nn.Conv1d(120, d_model, kernel_size=1)

        # 位置编码仅用于动作历史（有序序列）
        self.pos_encoder = PositionalEncoding(d_model, max_len=35)

        # Transformer 编码器 — batch_first=False (PyTorch 1.5 兼容)
        self.state_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )

        self.action_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                       dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )

        # MLP 头: state(d_model) + action(d_model) + X(x_dim) → 6 头
        fc_input_dim = d_model * 2 + x_dim
        self.linear1 = nn.Linear(fc_input_dim, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(512, 128)
        self.linear4 = nn.Linear(128, 6)

    def forward(self, src1, src2):
        """
        src1: (B, 41, 120) — 1 action + 5 static state + 35 history rows
        src2: (B, X_dim)   — auxiliary features (52 or 56 dim)

        返回: (B, 1) — Q-value (ŵ·LPW + (1-ŵ)·LPL)
        """
        # ── 状态路径 (rows 0-5 = 1 action + 5 card matrix rows) ──
        state_input = src1[:, :6].transpose(1, 2)     # (B, 120, 6)
        state_emb = self.state_embed(state_input)       # (B, d_model, 6)
        state_emb = state_emb.transpose(1, 2)           # (B, 6, d_model)

        # PyTorch 1.5.0: TransformerEncoder 需要 (S, B, E) 格式
        state_emb = state_emb.transpose(0, 1)           # (6, B, d_model)
        state_out = self.state_encoder(state_emb)        # (6, B, d_model)
        state_out = state_out.transpose(0, 1)            # (B, 6, d_model)
        state_out = state_out.mean(dim=1)                # (B, d_model)

        # ── 动作历史路径 (rows 6-40 = 35 history rows) ──
        action_input = src1[:, -35:].transpose(1, 2)   # (B, 120, 35)
        action_emb = self.action_embed(action_input)     # (B, d_model, 35)
        action_emb = action_emb.transpose(1, 2)          # (B, 35, d_model)
        action_emb = self.pos_encoder(action_emb)         # (B, 35, d_model)

        action_emb = action_emb.transpose(0, 1)          # (35, B, d_model)
        action_out = self.action_encoder(action_emb)      # (35, B, d_model)
        action_out = action_out.transpose(0, 1)           # (B, 35, d_model)
        action_out = action_out.mean(dim=1)               # (B, d_model)

        # ── 拼接 → MLP → 6 头 ──
        out = torch.cat([state_out, action_out, src2], dim=-1)
        out = F.leaky_relu_(self.linear1(out))
        out = F.leaky_relu_(self.linear2(out))
        out = F.leaky_relu_(self.linear3(out))
        out = self.linear4(out)

        win_rate, win, lose, _lor_a, _lor_p, _lor_d = torch.split(
            out, (1, 1, 1, 1, 1, 1), dim=-1)
        win_rate = torch.tanh(win_rate)
        _win_rate = (win_rate + 1) / 2
        out = _win_rate * win + (1. - _win_rate) * lose
        return out


# ---------------------------------------------------------------------------
# 模型字典 — 与训练端一致
# ---------------------------------------------------------------------------
BID_POSITIONS = ['first', 'second', 'third']
PLAY_POSITIONS = ['landlord', 'landlord_up', 'landlord_down']

model_dict = {
    "first": GeneralModelBid,
    "second": GeneralModelBid,
    'third': GeneralModelBid,
    "landlord": GeneralModelTransformer,
    "landlord_down": GeneralModelTransformer,
    "landlord_up": GeneralModelTransformer,
}
