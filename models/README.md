# 模型权重

本仓库**不包含**训练 checkpoint 与导出的 TorchScript 模型权重：单个 `.pt` 约 80 MB，
单次导出的 6 个模型合计约 250 MB，训练 checkpoint 目录则以 GB 计，均超出 GitHub 的仓库建议体积
（且单文件上限 100 MB）。权重请按下面的方式在本地生成。

## 一、训练产物（`.ckpt`）

`python/scripts/train.sh` 训练时会把 checkpoint 写到 `python/douzero_checkpoints/<XPID>/`，
每个位置一个文件，文件名格式为 `{position}_{frame}.ckpt`：

```
python/douzero_checkpoints/DouEnMH_v1.2/
├── first_611140100.ckpt          # 叫分位 0
├── second_611140100.ckpt         # 叫分位 1
├── third_611140100.ckpt          # 叫分位 2
├── landlord_611140100.ckpt       # 地主出牌
├── landlord_up_611140100.ckpt    # 农民上家
└── landlord_down_611140100.ckpt  # 农民下家
```

## 二、导出 TorchScript（`.pt`）

```bash
cd tools/export
python export_douenmh.py \
    --checkpoint-dir ../../python/douzero_checkpoints/DouEnMH_v1.2 \
    --output-dir     export_model
# 或指定步数后缀：--suffix 611140100
```

导出结果（每个后缀一个子目录）：

```
tools/export/export_model/611140100/
├── first_jit.pt   second_jit.pt  third_jit.pt
├── landlord_jit.pt  landlord_up_jit.pt  landlord_down_jit.pt
```

导出脚本默认使用 PyTorch 1.5.0 兼容环境（`BRIDGE_PYTHON` 环境变量可覆盖解释器路径），
详见 [../docs/model-export.md](../docs/model-export.md)。

## 三、C++ 推理端的模型目录

`cpp/src/main.cpp` 从工作目录下的 `data/models/` 读取模型，因此本地运行前把 6 个 `.pt`
放进 `cpp/build/data/models/`：

```bash
cd tools/export
cp export_model/611140100/*.pt ../../cpp/build/data/models/
```

Botzone 提交时，模型上传到平台的 User Storage，对应路径同样是 `data/models/`。

## 四、评测用的固定牌局

`python/eval_data.pkl` 是 1000 局固定发牌的评测集（136 KB），随仓库提供以保证评测可复现，
`python/evaluate.py`、`python/auto_test.py` 默认读取它。
