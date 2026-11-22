# C++ 推理端（Botzone）

本文件面向 C++ 推理端的部署与维护者：说明 `cpp/` 下每个文件的职责、一次决策的完整管线、
特征编码的落点、LibTorch 推理 API、KEEP_RUNNING 主循环、Botzone 交互协议、构建/运行/打包流程，
以及内存与耗时预算。训练端、模型导出与本地 runner 分别见 [python.md](python.md)、
[model-export.md](model-export.md)、[runner.md](runner.md)；三者的衔接见 [architecture.md](architecture.md)。

推理端的目标运行时是 **Botzone “带很多库” 编译器：G++ 7.2.0 / C++17 + LibTorch 1.5.0**，
本机 Windows 用 MSVC + 同一份 LibTorch 1.5.0 做等价验证。

## 一、架构总览

```
                    stdin: 一行 JSON（首回合全量 或 后续回合增量）
                                  │
                    ┌─────────────▼─────────────┐
                    │ game_state.h BotzoneIO::read │
                    │ 含 "requests" → 全量重建      │
                    │ 否则          → 增量更新      │
                    └─────────────┬─────────────┘
                                  │ GameState（跨回合常驻）
          ┌───────────────────────▼────────────────────────┐
          │ main.cpp 主循环                                 │
          │ 叫分: legalBidActions → buildBidFeatures        │
          │       → ModelInference::predictBid → argmax     │
          │ 出牌: LegalMoves.generate → 逐动作 oracle        │
          │       → 对手建模 C3/C4/C5 → buildPlayFeatures     │
          │       → ModelInference::predictPlay → argmax    │
          └───────────────┬────────────────────────────────┘
                          │ response JSON + KEEP_RUNNING 标记
                          ▼
                        stdout（Botzone 判题端）
```

- 进程内只有一份 `GameState` 与一份 `ModelInference`，全局静态对象跨回合存活（`main.cpp` 顶部）。
- 决策与状态更新是**同一个回合内**完成的：先输出决策，再按决策更新自己的状态（见第六节）。
- 除 `jsoncpp/` 外，所有源文件都是 header-only 或只有 `main.cpp`、`jsoncpp.cpp`、`oracle.cpp`
  三个编译单元。

### 1.1 源文件职责

| 文件 | 职责 | 关键接口 |
|------|------|----------|
| `src/main.cpp` | 唯一入口与决策调度：叫分/出牌选择、模型首次加载与按角色切换、状态更新闭环、KEEP_RUNNING 标记、异常兜底 | `main()`、`decideBidModel`、`decideBidFallback`、`decidePlayModel`、`ensurePlayModel` |
| `src/game_state.h` | 对局状态 `GameState`、首回合全量重建、后续回合增量更新、Botzone 读写、状态更新辅助函数 | `rebuildFromFullHistory`、`applyIncrementalUpdate`、`BotzoneIO::read/bid/play`、`removeFromMyHand`、`addToPlayedCards`、`isBombCards`、`updateSpring`、`computeOtherHandCards` |
| `src/card_defs.h` | 两套牌编码的基础类型：Botzone `Card`（0–53）与忽略花色的 `Level`（0–14），以及 `card2level` | `Card`、`Level`、`CardCount`、`card2level` |
| `src/card_encoding.h` | 8×15=120 维卡牌矩阵 `cards2array_v2`、Botzone↔env 牌号转换、出牌历史编码（pass/padding 区分）、炸弹 one-hot、C1 手牌统计 | `cards2array_v2`、`botzoneCardToEnvCard`、`handToEnvCards`、`processActionSeq_typed`、`actionSeqList2Array_typed`、`getOneHotBomb`、`computeHandStats` |
| `src/feature_encoder.h` | 把 `GameState` + 合法动作 + C2–C5 特征组装为模型输入：`z_batch`（V1，41×120）与 `x_batch`（52/56），并同时产出 V2 拆分格式 | `buildPlayFeatures`、`buildBidFeatures`、`PlayFeatureBatch`、`BidFeatureBatch` |
| `src/combo.h` | 牌型检测与大小比较；18 种 `ComboType`（含航天飞机 15–17 与 INVALID） | `Combo::detect`、`Combo::can_be_beaten_by`、`Combo::is_pass/is_invalid/is_bomb_type` |
| `src/move_generator.h` | 从手牌枚举 17 种牌型的全部组合；压牌时按指定长度生成；带牌组合用栈式组合枚举 | `gen_type_1_single` … `gen_type_17_sshuttle_4`、`gen_all_moves` |
| `src/legal_moves.h` | 合法动作过滤：自由出牌给全量；压牌时按上一手牌型路由到对应生成器并过滤主牌点数，再补炸弹/火箭与 PASS | `LegalMoves::generate` |
| `src/decision_router.h` | 另一套“启发式/模型”切换实现（`USE_HEURISTIC_BID`/`USE_HEURISTIC_PLAY`）。**当前不参与编译**：`main.cpp` 不包含它，且它 `#include` 的 `mcts_globals.h`、`mcts_state.h`、`mcts_bid.h`、`mcts_flat_mc.h` 不在 `cpp/src/` 内。实际路由逻辑在 `main.cpp` | `DecisionRouter`（未实例化） |
| `src/model_inference.h` | LibTorch 1.5.0 推理封装：模型加载/卸载、FP16→FP32 转换、分 chunk 推理、argmax、角色→模型名映射、路径拼接 | `ModelInference::loadModel/unloadModel/isLoaded/predictBid/predictPlay/predictPlayCrossAttention/modelPath/bidModelName/playModelName/predictPlayModel` |
| `src/oracle.h` / `src/oracle.cpp` | 神谕 DP：组合生成、`N_active`/`N_preserve` 双输出、`N_passive` 启发式、归一化、memo 缓存 | `compute_oracle_raw`、`normalize_oracle`、`dp_min_steps`、`dp_clear_cache`、`estimate_opponent_n_active` |
| `src/opponent_modeling.h` | 对手建模特征 C3（OHS+OBP，4 维）、C4（CTP，14 维）、C5（SWI+TP，4 维），全部从当前可见信息回放计算 | `computeOpponentFeatures`、`computeCtpFeatures`、`computeC5Features`、`buildUnknownPool` |
| `src/jsoncpp/` + `src/jsoncpp.cpp` | 仓库内置的 JsonCpp（头文件 + 单文件实现），用于解析请求与输出响应；Botzone 平台另提供 `jsoncpp/json.h`，单文件合成提交时保留该 include | `Json::Value`、`Json::parseFromStream`、`Json::writeString` |
| `src/card_defs.h` 之外的工具 | `cpp/tools/main_pass.cpp` 是 Pass Bot 基线（不依赖 LibTorch），`cpp/tools/prepare_botzone.py` 是提交打包脚本 | — |

编译单元：`src/main.cpp`、`src/jsoncpp.cpp`、`src/oracle.cpp`。其余 `.h` 通过 include 内联。

## 二、推理管线（5 阶段）

### ① JSON 解析与状态重建（`game_state.h`）

`BotzoneIO::read(GameState&)` 从 stdin 读一行 JSON：

- **首回合**：输入含 `"requests"` 键（`{"requests": [...], "responses": [...]}`）→
  `rebuildFromFullHistory(requests, responses)` 遍历全部历史重建 `GameState`。
- **后续回合**（KEEP_RUNNING）：输入只有本回合的 request（不含 `"requests"`）→
  `applyIncrementalUpdate(state, input)` 原地更新。

`rebuildFromFullHistory` 的三步：

1. 从 `requests[0]["own"]` 取初始手牌；从 `requests[0]["bid"].size()` 推断 `bidPosition`；
   用 `bid` 数组回填 `bidInfo`/`bidInput`。
2. 逐回合回放：首次出现 `publiccard` 时确定 `landlordPosition`、`finalbid`、`myPosition`，
   把底牌加入地主手牌；每个 request 的 `history[0..1]` 回放两个对手的动作（剩余牌数、已出牌、
   炸弹数、春天、`lastValidCombo`）；`responses[i]` 回放自己过往的叫分或出牌（整数=叫分，
   数组=出牌）。
3. 收尾：最后一回合含 `history` 即进入 `PLAYING`；地主未定时 `myPosition = bidPosition`。

### ② 合法动作生成（`combo.h` + `move_generator.h` + `legal_moves.h`）

- 自由出牌（`lastValidCombo` 为空/PASS）：`MoveGenerator::gen_all_moves()` 枚举 17 种牌型全部组合。
- 必须压牌：`LegalMoves::generate()` 按上一手 `ComboType` 路由到对应生成器（顺子/连对/飞机等
  只生成与上一手**同长度**的序列），用主牌点数过滤，再追加炸弹、火箭；最后始终追加 PASS（空数组）。
- 组内牌张排序，保证同一局面的候选顺序确定。

叫分阶段的合法动作由 `main.cpp` 的 `legalBidActions(maxBid)` 生成（见第七节）。

### ③ 特征编码（`feature_encoder.h` + `card_encoding.h` + oracle + opponent）

出牌阶段 `decidePlayModel()` 的组装顺序：

1. `LegalMoves` 得到候选动作列表；
2. 对完整手牌做一次 `compute_oracle_raw` 预热 DP memo；
3. 对每个候选动作，从其“出牌后剩余手牌”计算 C2 神谕特征（memo 命中）；
4. `computeOpponentFeatures`（C3）、`computeCtpFeatures`（C4）、`computeC5Features`（C5）各算一次；
5. `buildPlayFeatures(state, legalInts, perActionOracle, oppVec, ctpVec, c5Vec)` 产出
   `z_batch=(B,41,120)`、`x_batch=(B,52或56)`（同时产出 V2 的 `state_z/action_z/history_z`，当前不用）。

叫分阶段 `buildBidFeatures(hand, bidInfo, legalActions)` 产出 `z=(B,2,120)`、`x=(B,3)`。

编码细节（8×15 矩阵逐行含义、Z 行布局、X 各段偏移、pass/padding、相对位置）统一见
[encoding.md](encoding.md)；本文件不重复。

### ④ LibTorch 推理（`model_inference.h`）

分 chunk 组装张量、`forward`、把输出拼回、在候选动作维取 argmax。API 细节见第五节。

### ⑤ 决策输出与状态更新闭环（`main.cpp`）

拿到 `moves[bestIdx]` 后：

1. `BotzoneIO::play(chosenInts)` 输出本回合 response；
2. 按同一决策执行状态更新六步（第六节），使常驻 `GameState` 与外部世界保持一致；
3. 输出 `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<` 并 flush。

## 三、编码与特征的落点（结论）

以下只列 C++ 端用到的结论，逐字段定义与训练端对照见 [encoding.md](encoding.md)。

- **卡牌矩阵**：`cards2array_v2(envCards)` 输出 120 维 float，8 行 × 15 列，按**列主序**
  （`result[col*8 + row]`）展平。列 0–12 是 3,4,…,A,2；列 13/14 是小王/大王。
  行 0–3 为增量独热（≥1/≥2/≥3/=4 张），行 4 为断点，行 5/6/7 分别为单顺/连对/飞机起点。
- **牌号转换**：`card2level(c) = c/4 + c/53`（0–51→0–12，52→13，53→14）；
  `botzoneCardToEnvCard` 把 2 映射为 17（列 12）、小王 20（列 13）、大王 30（列 14）。
- **Z（出牌，V1）**：`z_batch=(B,41,120)`。每行：第 0 行是候选动作本身的 8×15 编码；
  第 1–5 行是 5 行静态状态（自己手牌、对手手牌并集=未知牌池、自己/上手/下手累计已出）；
  第 6–40 行是最近 35 手历史（最近的在最前；真实 PASS 编码为全零，不足 35 手的 padding 为全 −1）。
- **X（出牌）**：52 维 = bidInfo(3,÷3) + 三人剩余牌数(3,÷20) + 春天(1) + 已出炸弹 one-hot(15)
  + C1 手牌统计(7) + C2 神谕(5) + C3 对手信念(4) + C4 CTP(14)；启用 C5 时追加 SWI/TP(4) → 56 维。
  逐动作只有 C2 段（偏移 29）不同，其余段所有候选共享。
- **叫分特征**：`z=(B,2,120)`，第 0 行是 120 维全等于该叫分动作值的广播，第 1 行是手牌矩阵；
  `x=(B,3)` 为三人叫分原始值（不归一化）。

## 四、神谕与对手建模

### 4.1 神谕（Oracle）

入口 `compute_oracle_raw(const CardCount& hand_counts)`（`oracle.cpp`）：

1. `generate_all_combos` 生成标准 14 种牌型的全部组合（**不含**航天飞机 15–17）；
2. `dp_unified_impl` 一次递归同时求 `N_active`（不惩罚拆弹）与 `N_preserve`（每拆一个 4 张等级 +1 步）；
3. `N_passive` 用 `dp_passive_steps_heuristic`（各牌型不同主等级数的最大值减炸弹数，至少 1）；
4. `delta_n = max(0, n_passive − n_active)`；`self_adv` 留给调用方。

`normalize_oracle` 的饱和归一化：`a/(a+10)`、`p/(p+15)`、`ΔN/(ΔN+8)`、`pres/(pres+10)`、
`tanh(self_adv/5)`。`compute_oracle_raw` 内部的 DP 结果由全局 memo 哈希表缓存
（`MEMO_BITS=20`，1,048,576 槽位，源码注释标注约 16 MB），`dp_clear_cache()` 清空。

`main.cpp` 的调用时机：**每次出牌决策**。先用完整手牌预热 memo，再对每个候选动作计算
“出牌后剩余手牌”的神谕值；`self_adv` 用 `n_active − estimate_opponent_n_active(oppRemaining)`
补齐，`oppRemaining` 在本地主视角取两名农民剩余牌数的较小值，农民视角取地主剩余牌数。
`dp_clear_cache()` 在**每局第一次出牌决策**前调用一次（`g_firstPlayOfEpisode` 由叫分阶段置真、
首次出牌时清零）。

修复记录（`cpp/src/oracle.cpp` 的 `first_ci_for_rem` 构建）：原实现按 r 从 0 升序构建查找表，
而 combos 已按 size 降序排列，21 个表项于是都等于 combo 数；`dp_unified_internal` 的组合循环
因此一次都不执行，`N_active` / `N_preserve` 实际退化为 `greedy_bound` 给出的贪心上界。
修复方式：把构建循环改为 r 从 20 递减（ci 单调递增），与 Python `oracle.py::_precompute_combo_data`
一致，并在源码处补注释说明不能用升序。修复前后用插桩副本实测（同一手牌）：

| 手牌 | 修复前 | 修复后 |
|------|--------|--------|
| 17 张（4 个炸弹 + 单张） | `dp_calls=1`、`N_active=2`、`N_preserve=6`（贪心上界） | `dp_calls=24815`、`N_active=2`、`N_preserve=5` |
| 20 张（上述手牌 + 底牌 48,49,50） | `N_active=3`、`N_preserve=7`（贪心上界） | `N_active=3`、`N_preserve=4` |

修复后单次完整手牌 DP 冷启动约 1.3 ms（17 张）/ 6.6 ms（20 张），memo 热态约 0.01 ms。
`main.cpp` 先用整手牌预热 memo、再逐动作查询剩余手牌，因此逐动作多为命中；整机首手决策耗时
修复前后在测量噪声内（见第十一节）。修复后 C2 神谕特征重新与训练端一致。

公式、复杂度与跨语言对照见 [oracle.md](oracle.md)。

### 4.2 对手建模

三个入口都在 `opponent_modeling.h`，`main.cpp` 每次出牌决策各调用一次：

| 入口 | 维度 | 内容 |
|------|------|------|
| `computeOpponentFeatures(state, upPos, downPos, [outPoolVec, outPoolTotal])` | 4 | OHS（按 rank 权重加权的对手手牌强度，÷100 截断）+ OBP（超几何查表得到对手有炸弹/火箭的概率） |
| `computeCtpFeatures(state, upPos, downPos)` | 14 | 7 类牌型 × 2 对手，EMA γ=0.9，只统计“主动领出” |
| `computeC5Features(state, upPos, downPos)` | 4 | SWI（压制意愿 = EMA(实际压制)/EMA(p_can)）+ TP（每轮首手出牌数/6 的 EMA），按历史逐手推进、用当刻牌池计算 p_can |

未知牌池由 `buildUnknownPool` 统一构建：完整牌库 − 己方手牌 − 所有已出牌 −（非地主视角下的）
尚未打出的底牌。C3 的输出可复用给 C5，`main.cpp` 当前未复用（C5 内部按历史重建牌池以保证每个
历史动作使用当刻快照）。设计细节见 [opponent-modeling.md](opponent-modeling.md)。

## 五、ModelInference API

`model_inference.h` 的 `class ModelInference` 非线程安全，一个进程一个实例即可。

### 5.1 接口

```cpp
bool loadModel(const std::string& name, const std::string& path);
void unloadModel(const std::string& name);
bool isLoaded(const std::string& name) const;

int predictBid(const std::string& modelName,
               const std::vector<float>& z,          // (B,2,120) flat
               const std::vector<float>& x,          // (B,3) flat
               int batchSize);

int predictPlay(const std::string& modelName,
                const std::vector<float>& z_batch,   // (B,41,120) flat
                const std::vector<float>& x_batch,   // (B,52或56) flat
                int batchSize);

int predictPlayCrossAttention(const std::string& modelName,
                              const std::vector<float>& state_z,   // (1,5,120)
                              const std::vector<float>& action_z,  // (B,1,120)
                              const std::vector<float>& history_z, // (1,35,120)
                              const std::vector<float>& x_batch,   // (B,52或56)
                              int batchSize);

static const char* bidModelName(int bidPosition);
static const char* playModelName(int myPosition, int landlordPosition);
static const char* predictPlayModel(int bidPosition);
static std::string modelPath(const std::string& dir, const std::string& name,
                             bool fp16 = false);
```

### 5.2 语义

- **`loadModel(name, path)`**：`torch::jit::load(path)` 后 `eval()`；若模块第一个参数的
  `scalar_type()` 是 `at::kHalf`，整体 `to(torch::kFloat32)`（LibTorch 1.5.0 CPU 不支持 FP16 张量
  运算，FP16 模型必须在加载后转 FP32 才能推理）。成功返回 `true`；抛异常时捕获并返回 `false`
  （调用方据此走启发式回退）。检测实现只检查迭代到的第一个参数，FP16 导出模型所有参数均为 half，
  因此能正确触发转换。
- **`unloadModel(name)`**：从 `unordered_map` 删除模块，释放权重；切换到不同角色模型时先卸载再加载。
- **`isLoaded(name)`**：名字是否在表中。
- **`predictBid`**：chunk = 16；`z` 按 `MI_Z_ROWS*MI_Z_FEAT = 240` 步长、`x` 按 3 步长切分，
  `from_blob(...).clone()` 后 `NoGradGuard` 下 `forward({z,x}).toTensor()`，各 chunk 输出 `cat`
  后 `argmax(0).item<int>()`。B≤4（叫分合法动作最多 4 个），实际不会多 chunk。
- **`predictPlay`**：chunk = 50；`z` 步长 `41*120 = 4920`；`x_dim` 由
  `x_batch.size()/batchSize` 自动检测（52 或 56）；同样 `forward({z,x})` → `cat` → `argmax(0)`。
  候选动作多时（自由出牌）分多 chunk，压低单次 forward 的峰值内存。
- **`predictPlayCrossAttention`**：V2 拆分格式接口，输入 `(1,5,120)` / `(B,1,120)` /
  `(1,35,120)` / `(B,xDim)`，`forward` 接收 4 个张量；`main.cpp` 当前未使用（走 V1 `predictPlay`）。
- **`bidModelName(bidPosition)`**：`bidPosition % 3` → `"first"` / `"second"` / `"third"`。
- **`playModelName(myPos, landlordPos)`**：`myPos==landlordPos` → `"landlord"`；
  `myPos==(landlordPos+1)%3` → `"landlord_down"`（地主下手）；否则 `"landlord_up"`。
- **`predictPlayModel(bidPosition)`**：忽略参数，固定返回 `"landlord_down"`，用于第一次出牌前的预加载
  （地主未定时先按最常见的农民下手模型占位，真正出牌前由 `ensurePlayModel` 校正）。
- **`modelPath(dir, name, fp16=false)`**：拼成 `{dir}/{name}_jit.pt`；`fp16=true` 时为
  `{dir}/{name}_jit_fp16.pt`。`dir` 末尾已有 `/` 或 `\` 时不重复补分隔符；`dir` 为空时只用文件名。

### 5.3 输入形状、Q 值与模型文件命名

| 用途 | 输入 | 输出 |
|------|------|------|
| 叫分 | `z=(B,2,120)`、`x=(B,3)` | 每个候选叫分一个 Q 值；C++ 取 argmax |
| 出牌（V1） | `z=(B,41,120)`、`x=(B,52或56)` | 每个候选动作一个 Q 值；C++ 取 argmax |
| 出牌（V2） | `state_z=(1,5,120)`、`action_z=(B,1,120)`、`history_z=(1,35,120)`、`x=(B,52或56)` | 同上 |

Q 值由模型内部 6 个头合成，C++ 只做 argmax：

```
Q = w̃ · LPW + (1 − w̃) · LPL
w̃ = (tanh(LWR) + 1) / 2
```

`LWR` 为预测胜率头，`LPW` 为胜时 ADP 头，`LPL` 为负时 ADP 头；`LOR_a` / `LOR_p` / `LOR_d`
是神谕辅助头，只参与训练损失，不进入 Q。导出的 TorchScript `forward` 直接返回每个候选动作的
Q 向量（形状 `(B,1)`，因此 `argmax(0)` 就是在候选动作维取最大）。

模型文件命名固定为 `{name}_jit.pt`（FP32）与 `{name}_jit_fp16.pt`（FP16）；
`{name}` 取 `first`/`second`/`third`/`landlord`/`landlord_up`/`landlord_down` 六个之一。
导出流程见 [model-export.md](model-export.md)。

## 六、主循环与 KEEP_RUNNING

`main()` 是无限循环：`BotzoneIO::read` 阻塞读一行 → 决策 → 输出 → 输出标记。进程不退出，
模型与 `GameState` 保持常驻。

### 6.1 模型加载与角色切换

- **首回合**（`g_modelsLoaded == false`）：`loadModelsFirstRound()`
  加载 `bidModelName(bidPosition)`（叫分模型），并预加载 `predictPlayModel(...)`（固定
  `landlord_down`）。叫分模型加载失败不阻止流程，`decideBidModel` 会退回启发式。
- **每次出牌决策**：`ensurePlayModel()` 按 `playModelName(myPosition, landlordPosition)` 计算所需模型；
  与当前已加载模型不同则**先 `unloadModel` 旧模型再 `loadModel` 新模型**，只保留一个出牌模型以省内存。
- 模型目录是相对工作目录的 `data/models`（`main.cpp` 顶部 `static const string MODEL_DIR`）。

### 6.2 首回合全量重建 vs 增量更新

| 模式 | 触发 | 代价 |
|------|------|------|
| 全量 `rebuildFromFullHistory` | 输入含 `"requests"`（首回合） | 遍历全部历史，`O(N)`；代码注释标注设计目标 <5 ms |
| 增量 `applyIncrementalUpdate` | 输入不含 `"requests"`（KEEP_RUNNING 后续回合） | 只处理本回合两个对手的动作与可能出现的 `publiccard`，`O(1)`；代码注释标注设计目标 <1 ms |

增量更新不处理自己的动作：自己的出牌由决策后状态更新闭环写回。`publiccard` 首次出现时同时完成
“确定地主/阶段切换/把底牌加入地主手牌/补全缺失的 `bidInfo`”。

### 6.3 状态更新六步闭环

每手出牌决策并输出后，`main.cpp` 依次执行：

1. `removeFromMyHand(g_state, chosenInts)`：从手牌删除并同步自己的剩余牌数；
2. `g_state.cardPlayActionSeq.push_back({myPosition, chosenInts})`：追加历史序列；
3. `addToPlayedCards(g_state, myPosition, chosenInts)`：按角色累计到已出牌；
4. `isBombCards(chosenInts)` 为真则 `g_state.bombNum++`（含火箭）；
5. `updateSpring(g_state, myPosition, chosenInts)`：农民出牌→非春天；地主第 2 手起→非春天（不可逆）；
6. 非空出牌时 `g_state.lastValidCombo = chosenInts`。

叫分回合则把叫分写回 `bidInfo[bidPosition]` 并追加到 `bidInput`。

异常安全：整个决策包在 `try/catch` 中，任何异常都输出保守动作（叫分 0 / PASS），不退出进程。

### 6.4 模型缺失时的启发式回退

- **叫分**（`decideBidFallback`）：大小王各计 4、每张 2 计 2 得到 `handStrength`；
  `maxBid==0` 且 strength≥4 → 叫 3；`maxBid≤1` 且 strength≥2 → 叫 `maxBid+1`；
  `maxBid≤2` 且 strength≥6 → 叫 3；否则不叫。
- **出牌**（`decidePlayModel` 中 `!isLoaded(playName)` 分支）：自由出牌返回 `moves[0]`；
  需要压牌时返回第一个能压过 `lastValidCombo` 的候选；都不行则返回空数组（PASS）。
- 模型加载失败、推理异常时 `predictBid/predictPlay` 返回 0，配合 `isLoaded` 检查即落到上述回退。

### 6.5 KEEP_RUNNING 协议

- 首回合按普通协议输出 response 后，额外输出一行
  `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<` 并 flush；`main.cpp` 在**每回合输出后**都打印该标记。
- Botzone 收到标记后向进程组发 `SIGSTOP` 挂起，下回合轮到自己时发 `SIGCONT` 唤醒；
  后续回合输入不再包含历史（只有当前 request）。
- 回合之间不得占用 CPU：期间的 CPU 时间会算进下回合的 1 秒限额。
- 长时运行下 `debug`、`data`、`globaldata` 功能失效。

## 七、Botzone 交互协议

### 7.1 request JSON 字段

以下字段以 `game_state.h` 的实际解析为准（与平台协议一致）：

| 字段 | 出现阶段 | 含义 |
|------|----------|------|
| `own` | 叫分 + 出牌 | 自己最初拥有的牌（Botzone Card ID 0–53 数组） |
| `bid` | 叫分 | 前面玩家的叫分决策数组（长度 0/1/2 → 自己是 0/1/2 号叫分位） |
| `history` | 出牌 | 总是两项 `[上上家, 上家]`，每项是牌数组，空数组表示 PASS 或尚未轮到 |
| `publiccard` | 出牌（确定地主后首次及随后若干回合） | 三张底牌（地主公开牌） |
| `landlord` | 出牌 | 地主位置 0/1/2 |
| `pos` | 出牌 | 自己的玩家位置 0/1/2 |
| `finalbid` | 出牌 | 叫分阶段的最终底分 1/2/3 |

Botzone 每次输入是单行 JSON。首回合（含全量历史）格式：
`{"requests": [request0, request1, ...], "responses": [response0, ...]}`；KEEP_RUNNING 后续回合
只有当前 request。`reponses` 中叫分是整数、出牌是数组。`requests[0]["bid"].size()` 决定
`bidPosition`；`requests[0]["own"]` 是初始手牌。

### 7.2 response 格式

| 阶段 | 输出 | 说明 |
|------|------|------|
| 叫分 | `{"response": 0..3}` | 0 不叫，1/2/3 叫分 |
| 出牌 | `{"response": [cardId, ...]}` | 空数组 = PASS |

输出后紧跟 KEEP_RUNNING 标记行（6.5）。

### 7.3 叫分合法动作

设 `maxBid = max(bidInput)`：动作列表固定从 `{0}` 开始；`maxBid < 1` 时追加 1，`maxBid < 2`
时追加 2，`maxBid < 3` 时追加 3。模型在这些候选上取 argmax，`legalActions[bestIdx][0]`
就是输出的叫分值。

### 7.4 平台约束

- 每步决策 **1 秒**内完成；
- 进程内存 **256 MB** 内；
- 平台“带很多库”编译器提供 G++ 7.2.0（`-O2`、C++17）、jsoncpp、Eigen、LibTorch 1.5.0；
  选择该编译器时平台自带的 JSONCPP 版本与普通编译器不同，仓库内置了 `src/jsoncpp/` 以避免差异。

## 八、构建

### 8.1 环境变量

| 变量 | 必需 | 含义 | 示例 |
|------|------|------|------|
| `MSVC_DIR` | 可选* | 便携版 MSVC 根目录，必须含 `setup_x64.bat`；*在 “x64 Native Tools Command Prompt for VS” 中 `cl.exe` 已在 PATH，可不设 | `set "MSVC_DIR=path\to\msvc"` |
| `LIBTORCH_DIR` | 必需 | LibTorch 1.5.0 根目录，必须含 `include\torch\script.h` | `set "LIBTORCH_DIR=path\to\libtorch"` |
| `MODEL_DIR` | 可选 | 6 个 `.pt` 所在目录；未设置或不存在只警告，不使构建失败 | `set "MODEL_DIR=path\to\models"` |
| `AMALGAMATE` | 打包单文件时需要 | Amalgamate 可执行文件路径（也可放在 PATH 中） | `set "AMALGAMATE=path\to\Amalgamate.exe"` |

脚本内没有任何本机绝对路径：所有路径都从上述环境变量读取，缺失时打印明确错误与示例后退出。
`MSVC_DIR` / `LIBTORCH_DIR` / `MODEL_DIR` 允许传相对路径，脚本按调用者当前目录解析为绝对路径。
若 `MSVC_DIR` 未设置但 `cl.exe` 已在 PATH 中，脚本直接复用当前 MSVC 环境（跳过 `setup_x64.bat`）。

依赖目录结构：

```
<MSVC_DIR>/
  setup_x64.bat                 # 设置 cl.exe / INCLUDE / LIB
  VC/Tools/MSVC/<version>/...   # 编译器与 CRT DLL
<LIBTORCH_DIR>/
  include/torch/script.h
  include/torch/csrc/api/include/
  lib/torch.lib, torch_cpu.lib, c10.lib
  lib/torch.dll, torch_cpu.dll, c10.dll, asmjit.dll, fbgemm.dll, libiomp5md.dll
```

### 8.2 Windows（MSVC）

- `cpp/scripts/build_local.bat`：完整流程。
  - 编译器来源：设置 `MSVC_DIR`（含 `setup_x64.bat`），或直接在 VS 开发者命令提示符里运行（`cl` 已在 PATH 时自动跳过 setup）；
  - 编译器来源：设置 `MSVC_DIR`（含 `setup_x64.bat`），或直接在 VS 开发者命令提示符里运行（`cl` 已在 PATH）；
  - `build_local.bat` Release；`build_local.bat debug` Debug（`/Od /Zi /DDEBUG`）；
  - `build_local.bat run [N]` 构建后经 `tools/runner/runner.py` 跑 N 局；
  - `build_local.bat clean` 删除 `cpp/build`；
  - 输出 `cpp/build/bot.exe`；复制 LibTorch DLL 与 MSVC CRT DLL（CRT 目录从 `VCToolsInstallDir`
    或 `MSVC_DIR` 下的工具集目录自动定位，不写死工具集版本）；
  - 若 `MODEL_DIR` 存在，把模型链接/复制到 `cpp/build/data/models`；否则只警告。
- `cpp/scripts/build.bat`：最小构建，行为与 `build_local.bat` 的 Release 路径一致。
- 编译命令（等价）：

  ```
  cl /std:c++17 /EHsc /MD /O2 /W3 /utf-8 /D_CRT_SECURE_NO_WARNINGS /DNDEBUG ^
     /I"<cpp>/src" /Fe"<cpp>/build/bot.exe" ^
     src/main.cpp src/jsoncpp.cpp src/oracle.cpp ^
     /link torch.lib torch_cpu.lib c10.lib
  ```

### 8.3 Linux / GCC 参考构建

`cpp/scripts/build_botzone.sh` 用于验证 GCC 兼容性，对应 Botzone 的 G++ 7.2.0：

- `LIBTORCH_DIR` 默认 `/usr/local/libtorch`，可用环境变量覆盖；`CXX` 默认 `g++`；
- 输入同样三个编译单元，输出 `cpp/build/botzone/bot`；
- 链接 `-ltorch -ltorch_cpu -lc10 -lpthread -lstdc++fs`。

### 8.4 Pass 基线

Pass Bot（`cpp/tools/main_pass.cpp`）不依赖 LibTorch，用于验证 JSON/状态机逻辑：

- `cpp/scripts/build_pass.bat mingw`：MinGW g++；`build_pass.bat msvc`：优先用 PATH 中的 `cl`，
  否则用 `MSVC_DIR` 初始化环境；
- `cpp/scripts/build_pass.sh`：自定位仓库根，输出 `cpp/build/pass_bot`；
- 两者均不接受外部固定路径，缺少编译器/`MSVC_DIR` 时打印提示。

## 九、运行与模型放置

- 构建产物：`cpp/build/bot.exe`（Linux 参考构建为 `cpp/build/botzone/bot`）。
- `cpp/scripts/run_game.bat` 是对 `tools/runner/runner.py` 的包装：`run_game.bat [N] [--quiet]`
  会以 `--bot cpp/build/bot.exe` 调起 runner；runner 以 **bot.exe 所在目录为子进程工作目录**，
  因此模型必须放在 `cpp/build/data/models/`（与 `main.cpp` 的 `data/models` 相对路径对应）。
- 模型来源与导出：见 [models/README.md](../models/README.md) 与 [model-export.md](model-export.md)。
  导出目录 → `cpp/build/data/models/` 的 6 个文件：
  `first_jit.pt`、`second_jit.pt`、`third_jit.pt`、`landlord_jit.pt`、`landlord_up_jit.pt`、
  `landlord_down_jit.pt`。
- 手工冒烟测试（stdin 一行 JSON，需在 `cpp/build` 下运行，否则读不到 `data/models` 会走启发式回退）：

  ```bash
  cd cpp/build
  echo '{"requests":[{"own":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16],"bid":[]}],"responses":[]}' \
    | ./bot.exe
  ```

  期望输出一行 `{"response": <0..3>}` 与 KEEP_RUNNING 标记行。`data/models` 缺失时仍会输出
  （叫分走启发式、出牌走 PASS/最小可压牌），但这不是模型行为。
- runner 的完整参数、内置 agent 与日志字段见 [runner.md](runner.md)。

## 十、打包提交（两套方案）

### 10.1 方案 A：单文件合成（`cpp/scripts/amalgamate.sh`）

用 Amalgamate 从 `main.cpp` 出发内联所有本地头文件，并保留 `#include "jsoncpp/json.h"`
（提交时由 Botzone 平台提供 jsoncpp）：

```bash
export AMALGAMATE=/path/to/Amalgamate        # 或把 Amalgamate 放到 PATH
bash cpp/scripts/amalgamate.sh
```

脚本步骤：

1. 临时移走 `cpp/src/jsoncpp/`，让 Amalgamate 找不到而原样保留该 include；
2. 运行 Amalgamate 生成 `cpp/build/amalgamated_bot.cpp`；
3. 追加 `oracle.cpp` 的实现（Amalgamate 只跟随 include，不会自动带上独立 `.cpp`），
   并去掉其本地 include；
4. 恢复 `cpp/src/jsoncpp/`（中断时由 `trap` 恢复）；
5. 输出行数、内联文件数、保留的 jsoncpp include 数、oracle 定义引用数。

产物是单文件，可直接粘贴到 Botzone 编辑器提交。`AMALGAMATE` 未设置时脚本从 PATH 查找
`Amalgamate`/`Amalgamate.exe`，找不到会给出下载与用 MSVC 自建的提示。
该产物不内联 `jsoncpp.cpp` 的实现：`jsoncpp/json.h` 由 Botzone 平台提供，实现由平台 jsoncpp 库
在链接期提供；因此它不能脱离平台 jsoncpp 库单独本地链接。

### 10.2 方案 B：多文件 zip（`cpp/tools/prepare_botzone.py`）

```bash
python cpp/tools/prepare_botzone.py            # 生成 cpp/botzone_submit/ + cpp/botzone_submit.zip
python cpp/tools/prepare_botzone.py --check    # 只校验文件清单
python cpp/tools/prepare_botzone.py --amalgamate   # 单文件（内联 jsoncpp.cpp 实现；本地以 cpp/src 为 include 路径可编译）
```

- 清单 `SOURCES` 与 `main.cpp` 的实际 include 闭包一致：`main.cpp`、`game_state.h`、
  `card_encoding.h`、`card_defs.h`、`combo.h`、`move_generator.h`、`legal_moves.h`、
  `feature_encoder.h`、`model_inference.h`、`oracle.h`、`oracle.cpp`、`opponent_modeling.h`，
  外加 `jsoncpp/json.h`、`jsoncpp/json-forwards.h`、`jsoncpp.cpp`。
- `--check` 会校验：文件存在；清单内每个本地 include 都能被清单覆盖；`main.cpp` 不包含
  `decision_router.h`（其 MCTS 依赖不在 `cpp/src/`，不随提交）。
- 输出目录 `cpp/botzone_submit/` 与 `cpp/botzone_submit.zip` 均已在 `.gitignore` 中。

### 10.3 提交步骤

1. Botzone 选择编译器“带很多库”（LibTorch 1.5.0 + jsoncpp + Eigen）；
2. 上传 `cpp/botzone_submit.zip`（或粘贴单文件产物）；
3. 把 6 个 `.pt` 上传到 User Storage 的 `data/models/`（与 `main.cpp` 的 `MODEL_DIR` 对应）。

## 十一、内存与性能预算

平台硬约束：**每步 1 秒、进程 256 MB**。下表区分“平台/设计目标”与“本机实测”。
实测条件：Windows + 便携版 MSVC 19.29（`/O2`）+ LibTorch 1.5.0 CPU，模型为仓库外导出的 FP32
`_jit.pt`；数据来自 `cpp/build/bot.exe` 的进程内计时与 Windows 工作集采样，**不是 Botzone 实机数据**。

| 指标 | 数值 | 性质 |
|------|------|------|
| 单步决策时间上限 | 1 s | 平台约束 |
| 进程内存上限 | 256 MB | 平台约束 |
| 首次响应（进程启动 + 加载叫分模型与 `landlord_down` + 叫分决策） | 约 0.17–0.23 s（两次运行中位 0.20） | 实测（磁盘缓存已热） |
| 第一个出牌决策（自由出牌，17 张手牌，农民下手视角，无需切换模型） | 约 0.89 s（两次 0.887 / 0.889） | 实测（修复后；修复前约 0.95 s） |
| 第一个出牌决策（地主视角，20 张手牌，需卸载 `landlord_down` 再加载 `landlord`） | 约 1.40–1.47 s（两次中位 1.44） | 实测（修复后；含模型切换；修复前约 1.46 s） |
| 后续小牌局面出牌决策（4–5 张手牌） | 约 0.03–0.04 s | 实测 |
| 单次完整手牌神谕 DP（冷启动 / memo 热态） | 约 1.3 ms / 6.6 ms（17 / 20 张）；热态约 0.01 ms | 实测（修复后） |
| 加载叫分模型 + 一个出牌模型后的工作集 | 约 147 MB | 实测 |
| 首个出牌决策后的工作集 | 约 188–189 MB | 实测 |
| 峰值工作集 | 约 198–207 MB（地主首手切换模型时最高） | 实测 |
| 全量历史重建 | `O(N)`，代码注释标注 <5 ms；N 约 ≤100 手 | 设计目标，未在仓库内实测 |
| 增量更新 | `O(1)`，代码注释标注 <1 ms | 设计目标，未在仓库内实测 |
| 神谕 DP memo | `MEMO_BITS=20` → 1,048,576 槽位，源码注释标注约 16 MB | 源码常量/注释 |
| 单 chunk 输入张量 | 出牌 `50×41×120×4 ≈ 0.94 MB`（`z`）、`x` 另计；叫分 `16×2×120×4 ≈ 15 KB` | 由 chunk 常量估算 |
| `z_batch` 整体 | 候选数 B × 41 × 120 × 4 字节（B=200 时约 3.9 MB） | 由形状估算 |

两点需要部署者注意（均为实测现象，不是设计目标）：

- **首个出牌决策可能接近或超过 1 秒**：修复后本机实测农民首手约 0.89 s、地主首手约 1.40–1.47 s
  （后者包含卸载旧出牌模型、加载新模型的耗时，本机已超过 1 s 平台限额）。决策耗时的大头在
  出牌模型切换与特征组装，神谕 DP 修复只增加个位数毫秒量级（单次完整手牌 DP 冷启动
  1.3–6.6 ms，且逐动作多为 memo 命中）；Botzone 实机性能不同，但这是最坏情况的主要风险点，
  后续中残局决策只需几十毫秒。
- **内存余量约 50 MB**：峰值约 207 MB 对 256 MB 上限。`loadModel`/`unloadModel` 按角色只保留
  一个出牌模型、推理分 chunk 都是为压低峰值而采用的做法。

首次加载与全量重建的耗时依赖磁盘与判题端缓存，上表“首次响应”是热缓存下的数字。
