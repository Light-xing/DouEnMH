# DouEnMH

DouEnMH：PyTorch 训练，TorchScript 导出，C++ + LibTorch 1.5.0 在 [Botzone](https://botzone.org.cn) 实时对战。
这是一个基于Transformer的斗地主AI，在 DMC 分布式 actor-learner 框架上挂了 6 个监督头：除了经典的胜率 / 胜时价值 / 负时价值三头，还多出三个"神谕头"——用完全信息下"这手牌最少还要几步打完"的 DP结果做辅助监督，训练前期强推、后期退火，让价值网络先学会牌理再自己下棋。
本项目采用 8×15 = 120 维卡牌矩阵对游戏进行建模，显式标出卡牌断点、单顺/连对/飞机起点，替代 DouZero 的 54 维编码；外加 22 维对手建模——手牌强度、炸弹概率、七类牌型的出牌偏好、压制意愿与节奏——模型不只看自己手里有什么，也看对手想干什么。
本项目在Python训练端使用 PyTorch 2.x，CPP部署端为兼容Botzone使用 LibTorch 1.5.0：仓库自带 PyTorch 1.5 兼容模型层、TorchScript 导出与 FP16 精度对比，改一处特征两端都要同步。另附一个内嵌裁判的本地对局runner，不连平台也能把完整对局跑起来、按胜率验证改动。
本项目基于Douzero与 https://github.com/DouMH-Official/DouMH 中的代码进行开发，大量借用了DouMH项目中的思想与代码。

## 项目结构

```
python/            训练端：DouZero DMC 框架、8×15 环境、6 头模型、神谕 DP、对手建模、训练/评测/推理脚本
cpp/               推理端：Botzone C++ 源码（LibTorch 1.5.0）、构建脚本、单文件合成与提交打包
tools/export/      checkpoint → TorchScript 导出、FP16 精度对比
tools/runner/      本地对局 runner + 内嵌裁判（不依赖外部仓库）
docs/              架构、编码、神谕、对手建模、训练端、推理端、导出、runner 文档
models/            权重放置说明（权重本身不入库）
```

## 快速开始

### 0. 环境

| 用途 | 要求 |
|------|------|
| 训练 / 评测 | Python ≥ 3.8，PyTorch（CUDA 推荐），`pip install -r python/requirements.txt` |
| 模型导出 | Python 3.8 + PyTorch 1.5.0（与 Botzone 的 LibTorch 对齐，见 [docs/model-export.md](docs/model-export.md)） |
| C++ 推理端 | MSVC 2019+（或 GCC 7+）、LibTorch 1.5.0 CPU 版 |

模型权重不在仓库里（单个 `.pt` 约 80 MB），获取与放置方式见 [models/README.md](models/README.md)。

### 1. 训练

```bash
cd python
pip install -r requirements.txt
bash scripts/train_full.sh          # 全量配置；dev 小模型用 scripts/train.sh
# 常用覆盖：XPID=my_run D_MODEL=384 NHEAD=8 BATCH_SIZE=16 bash scripts/train_full.sh
# 续训：bash scripts/train_full.sh --load_model
```

checkpoint 写到 `python/douzero_checkpoints/<XPID>/{position}_{frame}.ckpt`。

### 2. 评测

```bash
cd python
python evaluate.py \
  --player_1_playcard douzero_checkpoints/DouEnMH_v1.2/landlord_611140100.ckpt \
  --player_2_playcard douzero_checkpoints/DouEnMH_v1.2/landlord_down_611140100.ckpt \
  --player_3_playcard douzero_checkpoints/DouEnMH_v1.2/landlord_up_611140100.ckpt
```

固定 1000 局发牌（`python/eval_data.pkl`）保证可复现；叫分模型默认用 random，可用
`--player_1_bid` 等参数换成 checkpoint 或 `Supervised`。

### 3. 导出 TorchScript

```bash
python tools/export/export_douenmh.py \
  --checkpoint-dir python/douzero_checkpoints/DouEnMH_v1.2 \
  --output-dir     tools/export/export_model
```

产出 `export_model/<suffix>/{first,second,third,landlord,landlord_up,landlord_down}_jit.pt`。
详细参数与 FP16 导出见 [docs/model-export.md](docs/model-export.md)。

### 4. 构建 C++ 推理端

```bat
cd cpp
rem 在 “x64 Native Tools Command Prompt for VS” 里（cl 已在 PATH）：
set LIBTORCH_DIR=<LibTorch 1.5.0 目录>
scripts\build_local.bat

rem 或使用便携 MSVC（目录内含 setup_x64.bat）：
set MSVC_DIR=<MSVC 目录>
set LIBTORCH_DIR=<LibTorch 1.5.0 目录>
scripts\build_local.bat
```

Linux 只用于验证 GCC 兼容性：`LIBTORCH_DIR=/path/to/libtorch bash scripts/build_botzone.sh`。

产物 `cpp/build/bot.exe`；把 6 个 `.pt` 放进 `cpp/build/data/models/`（见 [models/README.md](models/README.md)）。
环境变量、Linux 构建、Botzone 单文件提交见 [docs/cpp.md](docs/cpp.md)。

### 5. 本地对局测试

```bash
# 一个 bot 对两个随机对手，跑 100 局
python tools/runner/runner.py --bot cpp/build/bot.exe --games 100 --seed 42

# 三席位自定义 / 双 bot 互换身份对抗
python tools/runner/runner.py --p0 cpp/build/bot.exe --p1 greedy --p2 pass
python tools/runner/runner.py --bot cpp/build/bot_a.exe --vs cpp/build/bot_b.exe --vs-games 100
```

完整参数与统计字段见 [docs/runner.md](docs/runner.md)。

### 6. 提交到 Botzone

```bash
python cpp/tools/prepare_botzone.py          # 打包多文件 zip
bash cpp/scripts/amalgamate.sh               # 或合成单文件（需 Amalgamate 工具）
```

按 [docs/cpp.md](docs/cpp.md) 的"打包提交"一节选择编译器并把模型上传到 User Storage。

## 文档

| 文件 | 内容 |
|------|------|
| [docs/architecture.md](docs/architecture.md) | 全链路、6 模型、一次决策的数据流 |
| [docs/encoding.md](docs/encoding.md) | 三套卡牌编码、8×15 矩阵、Z/X 布局、bid 特征 |
| [docs/oracle.md](docs/oracle.md) | 神谕 DP：5 维特征、组合生成、memo 与优化 |
| [docs/opponent-modeling.md](docs/opponent-modeling.md) | OHS/OBP/CTP/SWI/TP 与未知牌池 |
| [docs/python.md](docs/python.md) | 训练端：架构、流程、参数、评测/推理脚本 |
| [docs/cpp.md](docs/cpp.md) | 推理端：API、KEEP_RUNNING、Botzone 协议、构建与提交 |
| [docs/model-export.md](docs/model-export.md) | PyTorch → TorchScript → LibTorch、FP16 精度验证 |
| [docs/runner.md](docs/runner.md) | 本地对局 runner：模式、参数、日志与统计 |
| [models/README.md](models/README.md) | 权重获取与放置 |

## 参考文献

S. Han, Z. Jin, Q. Liu, and C. Xu, "DouMH: Improving DouZero with multiple heads," in Proc. 2024 7th Int. Conf. Computer Information Science and Artificial Intelligence (CISAI), Shaoxing, China, Sep. 2024, pp. 297-302, doi: 10.1145/3703187.3703237.

## 许可

代码以 [Apache-2.0](LICENSE) 发布。其中 `python/douzero/` 源自 [DouZero](https://github.com/kwai/DouZero)，
`cpp/src/jsoncpp*` 为 [jsoncpp](https://github.com/open-source-parsers/jsoncpp) amalgamated 版本，
`tools/runner/judge/` 为 Botzone 裁判逻辑的本地副本。详见 [THIRD_PARTY.md](THIRD_PARTY.md)。
