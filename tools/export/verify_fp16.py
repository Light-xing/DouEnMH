"""
FP16 模型精度验证脚本 (DouEnMH).

对比 FP32 与 FP16 导出模型的推理输出，验证量化精度损失在可接受范围内。
需要在 Python 3.8 + PyTorch 1.5.0 的桥接环境中运行（下文记作 <BRIDGE_PYTHON>），
因为导出的 .pt 就是给 LibTorch 1.5.0 用的。

用法（在仓库根目录执行）:
    <BRIDGE_PYTHON> tools/export/verify_fp16.py \
        --model-dir tools/export/export_model \
        --suffix 620000000

    # 省略 --suffix 时自动选择 export_model/ 下编号最大的子目录
    <BRIDGE_PYTHON> tools/export/verify_fp16.py --model-dir tools/export/export_model
"""
import os
import sys
import argparse
import torch
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, 'export_model')
# None = 自动探测 MODEL_DIR 下编号最大的后缀子目录
SUFFIX = None
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


def _find_latest_suffix(model_dir):
    """返回 model_dir 下编号最大的后缀子目录名，找不到返回 None。"""
    best = None
    for name in os.listdir(model_dir):
        if not os.path.isdir(os.path.join(model_dir, name)):
            continue
        if name.isdigit():
            if best is None or int(name) > int(best):
                best = name
    return best


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


def verify_bid_model(model_dir, name):
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
        # LibTorch 1.5.0 CPU 没有 FP16 算子，C++ 端加载后也是转 FP32 推理；
        # 这里同样把权重转 FP32 再喂 FP32 输入，与部署路径一致。
        model_fp16 = torch.jit.load(fp16_path).float()
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
            out_fp16 = model_fp16(z, x)

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


def verify_play_model(model_dir, name):
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
        # LibTorch 1.5.0 CPU 没有 FP16 算子，C++ 端加载后也是转 FP32 推理；
        # 这里同样把权重转 FP32 再喂 FP32 输入，与部署路径一致。
        model_fp16 = torch.jit.load(fp16_path).float()
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
            out_fp16 = model_fp16(src1, src2)

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
    parser.add_argument('--model-dir', default=MODEL_DIR,
                        help=f'导出根目录 (默认: {MODEL_DIR})')
    parser.add_argument('--suffix', default=SUFFIX,
                        help='导出后缀子目录名 (默认: 自动选择编号最大的子目录)')
    args = parser.parse_args()

    suffix = args.suffix
    if suffix is None:
        if not os.path.isdir(args.model_dir):
            print(f'ERROR: model directory not found: {args.model_dir}')
            sys.exit(1)
        suffix = _find_latest_suffix(args.model_dir)
        if suffix is None:
            print(f'ERROR: no numeric suffix subdirectory under {args.model_dir}')
            print('请先运行 export_douenmh.py, 或用 --suffix 指定')
            sys.exit(1)
    model_dir = os.path.join(args.model_dir, str(suffix))

    print(f'Model directory: {model_dir}')
    print('FP16 模型加载后转 FP32 推理 (与 C++ 端 LibTorch 1.5.0 CPU 路径一致)')
    print(f'Thresholds: max_abs_err < {MAX_ABS_ERROR_THRESHOLD}, '
          f'argmax_agree >= {ARGMAX_AGREEMENT_MIN*100:.0f}%')
    print()

    results = []

    # Bid models
    print(f'{"="*70}')
    print(f'  Bid Models')
    print(f'{"="*70}')
    for name in BID_POSITIONS:
        r = verify_bid_model(model_dir, name)
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
        r = verify_play_model(model_dir, name)
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
