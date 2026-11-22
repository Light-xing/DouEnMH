"""
FP16 模型精度验证脚本 (DouEnMH).

对比 FP32 与 FP16 导出模型的推理输出，验证量化精度损失在可接受范围内。

用法:
    h:/bot/environments/bridge_python/python38_full/python.exe \
      h:/bot/DouEnMH-design/bridge/verify_fp16.py \
      --model-dir h:/bot/DouEnMH-design/bridge/export_model \
      --suffix 4572800
"""
import os
import sys
import argparse
import torch
import numpy as np

MODEL_DIR = r'h:\bot\DouEnMH-design\bridge\export_model'
SUFFIX = '4572800'
BID_POSITIONS = ['first', 'second', 'third']
PLAY_POSITIONS = ['landlord', 'landlord_up', 'landlord_down']

# DouEnMH V1 input dimensions
BID_Z_ROWS = 2
BID_Z_FEAT = 120
BID_X_DIM = 3
PLAY_Z_ROWS = 41
PLAY_Z_FEAT = 120

# Acceptable precision thresholds
MAX_ABS_ERROR_THRESHOLD = 1e-3     # FP16 vs FP32 max absolute error
ARGMAX_AGREEMENT_MIN = 0.99        # argmax agreement minimum ratio (99%)


def _get_x_dim_from_model(fp32_path):
    """Try to infer x_dim from the saved model."""
    try:
        model = torch.jit.load(fp32_path)
        # The model performs: concat(state_out, action_out, src2) → MLP
        # We can infer x_dim by trying forward passes — but simplest is
        # to just try (B, 56) and fall back to (B, 52)
        s1 = torch.randn(1, PLAY_Z_ROWS, PLAY_Z_FEAT)
        s2 = torch.randn(1, 56)
        with torch.no_grad():
            model(s1, s2)
        return 56
    except Exception:
        return 52


def verify_bid_model(model_dir, name, suffix):
    """Verify bid model FP16 vs FP32."""
    fp32_path = os.path.join(model_dir, f'{name}_jit.pt')
    fp16_path = os.path.join(model_dir, f'{name}_jit_fp16.pt')

    if not os.path.exists(fp32_path):
        return {'name': name, 'status': 'SKIP (no FP32 model)'}
    if not os.path.exists(fp16_path):
        return {'name': name, 'status': 'SKIP (no FP16 model)'}

    fp32_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    fp16_mb = os.path.getsize(fp16_path) / (1024 * 1024)

    try:
        model_fp32 = torch.jit.load(fp32_path)
        model_fp32.eval()
        model_fp16 = torch.jit.load(fp16_path)
        model_fp16.eval()
    except Exception as e:
        return {'name': name, 'status': f'ERROR loading model: {e}'}

    max_abs_errs = []
    argmax_agreements = []

    for bs in [1, 2, 4, 8, 16, 32]:
        z = torch.randn(bs, BID_Z_ROWS, BID_Z_FEAT)
        x = torch.randn(bs, BID_X_DIM)

        with torch.no_grad():
            out_fp32 = model_fp32(z, x)
            out_fp16 = model_fp16(z.half(), x.half()).float()

        max_abs = (out_fp32 - out_fp16).abs().max().item()
        max_abs_errs.append(max_abs)

        argmax32 = out_fp32.squeeze(-1).argmax(-1)
        argmax16 = out_fp16.squeeze(-1).argmax(-1)
        agreement = (argmax32 == argmax16).float().mean().item()
        argmax_agreements.append(agreement)

    avg_max_abs = np.mean(max_abs_errs)
    avg_argmax_agree = np.mean(argmax_agreements)
    passed = (avg_max_abs < MAX_ABS_ERROR_THRESHOLD and
              avg_argmax_agree >= ARGMAX_AGREEMENT_MIN)

    return {
        'name': name,
        'status': 'PASS' if passed else 'FAIL',
        'fp32_mb': fp32_mb,
        'fp16_mb': fp16_mb,
        'avg_max_abs_err': avg_max_abs,
        'avg_argmax_agree': avg_argmax_agree,
        'max_abs_errs': max_abs_errs,
        'argmax_agreements': argmax_agreements,
    }


def verify_play_model(model_dir, name, suffix):
    """Verify play model FP16 vs FP32."""
    fp32_path = os.path.join(model_dir, f'{name}_jit.pt')
    fp16_path = os.path.join(model_dir, f'{name}_jit_fp16.pt')

    if not os.path.exists(fp32_path):
        return {'name': name, 'status': 'SKIP (no FP32 model)'}
    if not os.path.exists(fp16_path):
        return {'name': name, 'status': 'SKIP (no FP16 model)'}

    fp32_mb = os.path.getsize(fp32_path) / (1024 * 1024)
    fp16_mb = os.path.getsize(fp16_path) / (1024 * 1024)

    try:
        model_fp32 = torch.jit.load(fp32_path)
        model_fp32.eval()
        model_fp16 = torch.jit.load(fp16_path)
        model_fp16.eval()
    except Exception as e:
        return {'name': name, 'status': f'ERROR loading model: {e}'}

    x_dim = _get_x_dim_from_model(fp32_path)

    max_abs_errs = []
    argmax_agreements = []

    for bs in [1, 4, 16, 64, 128]:
        src1 = torch.randn(bs, PLAY_Z_ROWS, PLAY_Z_FEAT)
        src2 = torch.randn(bs, x_dim)

        with torch.no_grad():
            out_fp32 = model_fp32(src1, src2)
            out_fp16 = model_fp16(src1.half(), src2.half()).float()

        max_abs = (out_fp32 - out_fp16).abs().max().item()
        max_abs_errs.append(max_abs)

        argmax32 = out_fp32.squeeze(-1).argmax(-1)
        argmax16 = out_fp16.squeeze(-1).argmax(-1)
        agreement = (argmax32 == argmax16).float().mean().item()
        argmax_agreements.append(agreement)

    avg_max_abs = np.mean(max_abs_errs)
    avg_argmax_agree = np.mean(argmax_agreements)
    passed = (avg_max_abs < MAX_ABS_ERROR_THRESHOLD and
              avg_argmax_agree >= ARGMAX_AGREEMENT_MIN)

    return {
        'name': name,
        'status': 'PASS' if passed else 'FAIL',
        'fp32_mb': fp32_mb,
        'fp16_mb': fp16_mb,
        'avg_max_abs_err': avg_max_abs,
        'avg_argmax_agree': avg_argmax_agree,
        'max_abs_errs': max_abs_errs,
        'argmax_agreements': argmax_agreements,
    }


def main():
    parser = argparse.ArgumentParser(description='DouEnMH FP16 vs FP32 precision verification')
    parser.add_argument('--model-dir', default=MODEL_DIR)
    parser.add_argument('--suffix', default=SUFFIX)
    args = parser.parse_args()

    print(f'Model directory: {args.model_dir}/{args.suffix}')
    print(f'Thresholds: max_abs_err < {MAX_ABS_ERROR_THRESHOLD}, '
          f'argmax_agree >= {ARGMAX_AGREEMENT_MIN*100:.0f}%')
    print()

    results = []

    # Bid models
    print(f'{"="*70}')
    print(f'  Bid Models')
    print(f'{"="*70}')
    for name in BID_POSITIONS:
        r = verify_bid_model(args.model_dir, name, args.suffix)
        results.append(r)
        if 'SKIP' in r['status']:
            print(f'  {name:12s}  {r["status"]}')
        elif 'ERROR' in r['status']:
            print(f'  {name:12s}  {r["status"]}')
        else:
            print(f'  {name:12s}  {r["status"]:4s}  '
                  f'FP32={r["fp32_mb"]:.1f}MB  FP16={r["fp16_mb"]:.1f}MB  '
                  f'save={((1-r["fp16_mb"]/r["fp32_mb"])*100):.0f}%  '
                  f'max_abs_err={r["avg_max_abs_err"]:.2e}  '
                  f'argmax_agree={r["avg_argmax_agree"]*100:.1f}%')

    # Play models
    print(f'\n{"="*70}')
    print(f'  Play Models')
    print(f'{"="*70}')
    for name in PLAY_POSITIONS:
        r = verify_play_model(args.model_dir, name, args.suffix)
        results.append(r)
        if 'SKIP' in r['status']:
            print(f'  {name:20s}  {r["status"]}')
        elif 'ERROR' in r['status']:
            print(f'  {name:20s}  {r["status"]}')
        else:
            print(f'  {name:20s}  {r["status"]:4s}  '
                  f'FP32={r["fp32_mb"]:.1f}MB  FP16={r["fp16_mb"]:.1f}MB  '
                  f'save={((1-r["fp16_mb"]/r["fp32_mb"])*100):.0f}%  '
                  f'max_abs_err={r["avg_max_abs_err"]:.2e}  '
                  f'argmax_agree={r["avg_argmax_agree"]*100:.1f}%')

    # Summary
    passed = sum(1 for r in results if r['status'] == 'PASS')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    skipped = sum(1 for r in results if 'SKIP' in r['status'])

    print(f'\n{"="*70}')
    print(f'  Summary: {passed} PASS, {failed} FAIL, {skipped} SKIP')
    if failed > 0:
        print(f'  WARNING: {failed} model(s) failed FP16 verification')
    elif passed > 0:
        print(f'  OK: All models passed FP16 verification')
    print(f'{"="*70}')


if __name__ == '__main__':
    main()
