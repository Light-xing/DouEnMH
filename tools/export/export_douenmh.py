"""
DouEnMH 模型导出为 TorchScript (.pt) 供 Botzone LibTorch 1.5.0 加载。

特性:
  - 自动从 checkpoint 检测架构参数 (d_model/nhead/layers/dim_ff/x_dim)
  - Bid:  z=(B, 2, 120), x=(B, 3) → 3 头输出
  - Play: src1=(B, 41, 120), src2=(B, X_dim) → 6 头输出  [V1 双塔]
  - 支持 FP32 (torch.jit.trace) 和 FP16 (torch.jit.script)

用法 (在 bridge_python 的 PyTorch 1.5.0+ 环境中运行):
    h:/bot/environments/bridge_python/python38_full/python.exe \
      h:/bot/DouEnMH-design/bridge/export_douenmh.py \
      --checkpoint-dir h:/bot/DouEnMH-design/DouEnMH/DouEnMH-new/douzero_checkpoints/DouEnMH_v1.2 \
      --output-dir h:/bot/DouEnMH-design/bridge/export_model \
      --suffix 4093600

    # Auto-detect latest suffix:
    h:/bot/environments/bridge_python/python38_full/python.exe \
      h:/bot/DouEnMH-design/bridge/export_douenmh.py
"""
import os
import sys
import argparse
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from compat_models_douenmh import (
    GeneralModelBid, GeneralModelTransformer,
    detect_play_architecture,
    BID_POSITIONS, PLAY_POSITIONS,
)

# 默认路径 (可通过 CLI 覆盖)
CHECKPOINT_DIR = r'h:\bot\DouEnMH-design\DouEnMH\DouEnMH-new\douzero_checkpoints\DouEnMH_v1.2'
OUTPUT_DIR = r'h:\bot\DouEnMH-design\bridge\export_model'
SUFFIX = '4093600'

# Fixed dimensions (encoding level, not model architecture)
Z_DIM = 120           # cards2array_v2 维度
BID_Z_ROWS = 2        # action broadcast + hand
PLAY_Z_ROWS = 41      # 1 action + 5 static + 35 history
BID_X_DIM = 3         # bid_info only


def _find_latest_suffix(checkpoint_dir):
    """在 checkpoint 目录中找到最新(最大步数)的 suffix."""
    best = 0
    seen = set()
    for f in os.listdir(checkpoint_dir):
        if not f.endswith('.ckpt'):
            continue
        # 格式: {position}_{step}.ckpt
        base = f[:-5]  # remove .ckpt
        parts = base.rsplit('_', 1)
        if len(parts) == 2:
            try:
                step = int(parts[1])
            except ValueError:
                continue
            if step > best:
                best = step
            seen.add(base)
    return str(best) if best > 0 else None


# ---------------------------------------------------------------------------
# 导出: 叫分模型 FP32
# ---------------------------------------------------------------------------
def export_bid_fp32(name, checkpoint_path, output_dir, verify=True):
    print(f'\n{"="*60}')
    print(f'DouEnMH Bid [FP32]: {name}')
    print(f'  checkpoint: {checkpoint_path}')

    model = GeneralModelBid()
    model.eval()
    sd = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(sd)
    print(f'  weights loaded ({len(sd)} keys)')

    example_z = torch.randn(1, BID_Z_ROWS, Z_DIM)
    example_x = torch.randn(1, BID_X_DIM)
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_z, example_x))

    if verify:
        _verify_bid(model, traced)

    out_path = os.path.join(output_dir, f'{name}_jit.pt')
    traced.save(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'  -> {out_path} ({size_mb:.1f} MB)')
    return out_path


# ---------------------------------------------------------------------------
# 导出: 叫分模型 FP16
# ---------------------------------------------------------------------------
def export_bid_fp16(name, checkpoint_path, output_dir, verify=True):
    print(f'\n{"="*60}')
    print(f'DouEnMH Bid [FP16]: {name}')
    print(f'  checkpoint: {checkpoint_path}')

    model = GeneralModelBid()
    model.eval()
    sd = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(sd)
    model = model.half()

    with torch.no_grad():
        scripted = torch.jit.script(model)

    if verify:
        _verify_bid_fp16(scripted, output_dir, name)

    out_path = os.path.join(output_dir, f'{name}_jit_fp16.pt')
    scripted.save(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'  -> {out_path} ({size_mb:.1f} MB)')
    return out_path


# ---------------------------------------------------------------------------
# 导出: 出牌模型 FP32
# ---------------------------------------------------------------------------
def export_play_fp32(name, checkpoint_path, output_dir, nhead=8, verify=True):
    sd = torch.load(checkpoint_path, map_location='cpu')
    arch = detect_play_architecture(sd, nhead=nhead)

    print(f'\n{"="*60}')
    print(f'DouEnMH Play [FP32]: {name}')
    print(f'  checkpoint: {checkpoint_path}')
    print(f'  detected: d_model={arch["d_model"]} nhead={arch["nhead"]} '
          f'layers={arch["num_encoder_layers"]} dim_ff={arch["dim_feedforward"]} '
          f'x_dim={arch["x_dim"]}')

    model = GeneralModelTransformer(**arch)
    model.eval()
    model.load_state_dict(sd)
    print(f'  weights loaded ({len(sd)} keys)')

    example_src1 = torch.randn(1, PLAY_Z_ROWS, Z_DIM)
    example_src2 = torch.randn(1, arch['x_dim'])
    with torch.no_grad():
        traced = torch.jit.trace(model, (example_src1, example_src2))

    if verify:
        _verify_play(model, traced, arch['x_dim'])

    out_path = os.path.join(output_dir, f'{name}_jit.pt')
    traced.save(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'  -> {out_path} ({size_mb:.1f} MB)')
    return out_path


# ---------------------------------------------------------------------------
# 导出: 出牌模型 FP16
# ---------------------------------------------------------------------------
def export_play_fp16(name, checkpoint_path, output_dir, nhead=8, verify=True):
    sd = torch.load(checkpoint_path, map_location='cpu')
    arch = detect_play_architecture(sd, nhead=nhead)

    print(f'\n{"="*60}')
    print(f'DouEnMH Play [FP16]: {name}')
    print(f'  checkpoint: {checkpoint_path}')
    print(f'  detected: d_model={arch["d_model"]} nhead={arch["nhead"]} '
          f'layers={arch["num_encoder_layers"]} dim_ff={arch["dim_feedforward"]} '
          f'x_dim={arch["x_dim"]}')

    model = GeneralModelTransformer(**arch)
    model.eval()
    model.load_state_dict(sd)
    model = model.half()  # register_buffer('pe') 会自动跟随转换

    with torch.no_grad():
        scripted = torch.jit.script(model)

    if verify:
        _verify_play_fp16(scripted, output_dir, name, arch['x_dim'])

    out_path = os.path.join(output_dir, f'{name}_jit_fp16.pt')
    scripted.save(out_path)
    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f'  -> {out_path} ({size_mb:.1f} MB)')
    return out_path


# ---------------------------------------------------------------------------
# 验证辅助
# ---------------------------------------------------------------------------
def _verify_bid(model, traced):
    for bs in [1, 3, 10]:
        z = torch.randn(bs, BID_Z_ROWS, Z_DIM)
        x = torch.randn(bs, BID_X_DIM)
        with torch.no_grad():
            o1 = model(z, x)
            o2 = traced(z, x)
        diff = (o1 - o2).abs().max().item()
        status = "OK" if diff < 1e-5 else "MISMATCH"
        print(f'  batch={bs:3d} orig vs traced max diff: {diff:.2e}  [{status}]')


def _verify_play(model, traced, x_dim):
    for bs in [1, 3, 10, 50]:
        s1 = torch.randn(bs, PLAY_Z_ROWS, Z_DIM)
        s2 = torch.randn(bs, x_dim)
        with torch.no_grad():
            o1 = model(s1, s2)
            o2 = traced(s1, s2)
        diff = (o1 - o2).abs().max().item()
        status = "OK" if diff < 1e-5 else "MISMATCH"
        print(f'  batch={bs:3d} orig vs traced max diff: {diff:.2e}  [{status}]')


def _verify_bid_fp16(scripted, output_dir, name):
    fp32_path = os.path.join(output_dir, f'{name}_jit.pt')
    if not os.path.exists(fp32_path):
        print(f'  (skip cross-precision verify: FP32 model not found)')
        return

    fp32_model = torch.jit.load(fp32_path)
    fp16_as_fp32 = scripted.float()

    for bs in [1, 3, 10]:
        z = torch.randn(bs, BID_Z_ROWS, Z_DIM)
        x = torch.randn(bs, BID_X_DIM)
        with torch.no_grad():
            out_fp32 = fp32_model(z, x)
            out_fp16 = fp16_as_fp32(z, x)
        diff = (out_fp32 - out_fp16).abs().max().item()
        status = "OK" if diff < 1e-3 else "LARGE"
        print(f'  batch={bs:3d} FP32-traced vs FP16-scripted(->FP32) diff: {diff:.2e}  [{status}]')


def _verify_play_fp16(scripted, output_dir, name, x_dim):
    fp32_path = os.path.join(output_dir, f'{name}_jit.pt')
    if not os.path.exists(fp32_path):
        print(f'  (skip cross-precision verify: FP32 model not found)')
        return

    fp32_model = torch.jit.load(fp32_path)
    fp16_as_fp32 = scripted.float()

    for bs in [1, 3, 10, 50]:
        s1 = torch.randn(bs, PLAY_Z_ROWS, Z_DIM)
        s2 = torch.randn(bs, x_dim)
        with torch.no_grad():
            out_fp32 = fp32_model(s1, s2)
            out_fp16 = fp16_as_fp32(s1, s2)
        diff = (out_fp32 - out_fp16).abs().max().item()
        status = "OK" if diff < 1e-3 else "LARGE"
        print(f'  batch={bs:3d} FP32-traced vs FP16-scripted(->FP32) diff: {diff:.2e}  [{status}]')


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='DouEnMH -> TorchScript 导出 (Botzone LibTorch 1.5.0)')
    parser.add_argument('--checkpoint-dir', default=CHECKPOINT_DIR,
                        help=f'checkpoint 目录 (默认: {CHECKPOINT_DIR})')
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help=f'输出目录 (默认: {OUTPUT_DIR})')
    parser.add_argument('--suffix', default=None,
                        help='checkpoint frame 编号 (默认: 自动选择最新)')
    parser.add_argument('--positions', nargs='+',
                        default=['first', 'second', 'third',
                                 'landlord', 'landlord_up', 'landlord_down'],
                        help='要导出的模型位置 (默认: 全部 6 个)')
    parser.add_argument('--skip-verify', action='store_true',
                        help='跳过导出后验证')
    parser.add_argument('--fp16', action='store_true',
                        help='额外导出 FP16 量化版本')
    parser.add_argument('--fp16-only', action='store_true',
                        help='仅导出 FP16 版本')
    parser.add_argument('--nhead', type=int, default=8,
                        help='注意力头数 (无法从权重推断, 需匹配训练配置, 默认: 8)')
    args = parser.parse_args()

    # 自动选择最新 suffix
    suffix = args.suffix
    if suffix is None:
        suffix = _find_latest_suffix(args.checkpoint_dir)
        if suffix is None:
            print(f'ERROR: 未在 {args.checkpoint_dir} 中找到任何 .ckpt 文件')
            print('请指定 --suffix')
            sys.exit(1)
        print(f'自动选择最新 suffix: {suffix}')

    output_dir = os.path.join(args.output_dir, suffix)

    print(f'PyTorch 版本: {torch.__version__}')
    print(f'Checkpoint 目录: {args.checkpoint_dir}')
    print(f'Suffix: {suffix}')
    print(f'输出目录: {output_dir}')
    print(f'位置: {args.positions}')
    print()

    if args.fp16 or args.fp16_only:
        print('⚠ FP16 使用 torch.jit.script 导出 (非 trace)。')
        print('  LibTorch 1.5.0 CPU 不支持 FP16 内核, C++ 端会自动 .to(kFloat32) 转 FP32 推理。')
        print()

    os.makedirs(output_dir, exist_ok=True)
    verify = not args.skip_verify
    paths = []

    bid_pos = [p for p in args.positions if p in BID_POSITIONS]
    play_pos = [p for p in args.positions if p in PLAY_POSITIONS]

    # --- FP32 ---
    if not args.fp16_only:
        for name in bid_pos:
            ckpt = os.path.join(args.checkpoint_dir, f'{name}_{suffix}.ckpt')
            if not os.path.exists(ckpt):
                print(f'  [SKIP] not found: {ckpt}')
                continue
            p = export_bid_fp32(name, ckpt, output_dir, verify)
            paths.append(p)

        for name in play_pos:
            ckpt = os.path.join(args.checkpoint_dir, f'{name}_{suffix}.ckpt')
            if not os.path.exists(ckpt):
                print(f'  [SKIP] not found: {ckpt}')
                continue
            p = export_play_fp32(name, ckpt, output_dir, args.nhead, verify)
            paths.append(p)

    # --- FP16 ---
    if args.fp16 or args.fp16_only:
        for name in bid_pos:
            ckpt = os.path.join(args.checkpoint_dir, f'{name}_{suffix}.ckpt')
            if not os.path.exists(ckpt):
                continue
            p = export_bid_fp16(name, ckpt, output_dir, verify)
            paths.append(p)

        for name in play_pos:
            ckpt = os.path.join(args.checkpoint_dir, f'{name}_{suffix}.ckpt')
            if not os.path.exists(ckpt):
                continue
            p = export_play_fp16(name, ckpt, output_dir, args.nhead, verify)
            paths.append(p)

    # --- 汇总 ---
    if not paths:
        print('\nERROR: 没有成功导出任何模型!')
        print(f'请检查 checkpoints 是否存在: {args.checkpoint_dir}/*_{suffix}.ckpt')
        sys.exit(1)

    print(f'\n{"="*60}')
    print(f'导出完成! 共 {len(paths)} 个模型文件:')
    total_mb = 0
    for p in paths:
        mb = os.path.getsize(p) / (1024 * 1024)
        total_mb += mb
        print(f'  {os.path.basename(p):50s} {mb:7.1f} MB')
    print(f'  总计: {total_mb:.1f} MB')

    if args.fp16 or args.fp16_only:
        print(f'\n--- FP32 vs FP16 文件大小对比 ---')
        for name in bid_pos + play_pos:
            fp32 = os.path.join(output_dir, f'{name}_jit.pt')
            fp16 = os.path.join(output_dir, f'{name}_jit_fp16.pt')
            if os.path.exists(fp32) and os.path.exists(fp16):
                mb32 = os.path.getsize(fp32) / (1024 * 1024)
                mb16 = os.path.getsize(fp16) / (1024 * 1024)
                pct = (1 - mb16 / mb32) * 100
                print(f'  {name:20s}  FP32={mb32:.1f}MB  ->  FP16={mb16:.1f}MB  '
                      f'({pct:.0f}% smaller)')

    print(f'\n将这些 .pt 文件复制到 Botzone/C++ model 目录:')
    print(f'  C++ 端使用: MODEL_DIR = "data/models"')
    print(f'  Bid  调用:  model.forward(z=(B,{BID_Z_ROWS},{Z_DIM}), x=(B,{BID_X_DIM}))')
    print(f'  Play 调用:  model.forward(src1=(B,{PLAY_Z_ROWS},{Z_DIM}), src2=(B,X_dim))')
    print(f'\n提示: 运行 verify_fp16.py 可进一步验证 FP16 模型输出精度。')


if __name__ == '__main__':
    main()
