# 本地对局 runner

本文件讲 `tools/runner/runner.py`：它**不连 Botzone**，而是在本地用仓库内嵌的官方裁判逻辑跑完整斗地主对局，用来验证 C++ 推理端 `cpp/build/bot.exe` 和三个内置基线 agent。

## 一、定位与用途

runner 把三样东西接在一个进程里：

- **裁判**：`tools/runner/judge/judge_botzone_ver.py`，Botzone 平台斗地主裁判逻辑，原样收编（来源见 [THIRD_PARTY.md](../THIRD_PARTY.md)）。
- **玩家**：每个席位可以是一个 `bot.exe` 子进程，也可以是进程内的 `random` / `greedy` / `pass` 基线 agent。
- **主循环**：`runner.py` 负责发牌种子、逐回合向裁判要"该谁动、请求是什么"、把请求转给对应玩家、把响应写回裁判日志，直到裁判给出 `finish`。

它解决的是"不部署到平台也能完整跑一局"的问题：

- 验证 C++ 推理端能正确解析 Botzone JSON、维持 KEEP_RUNNING 状态、返回合法牌型；
- 用内置 agent 当基线对手，做语法/协议冒烟测试和粗略强弱对比；
- 单局复现 bug、批量跑胜率、两个 bot 同牌互换身份做对抗验证。

它**不做**的事：不提交 Botzone、不模拟平台的 1 秒/256 MB 限制、不做训练或模型导出。平台提交与构建见 [cpp.md](cpp.md)。

## 二、架构

```
┌──────────────────────────────────────────────────────────────┐
│ runner.py 主进程                                              │
│                                                              │
│  ┌────────────────────────┐   ┌───────────────────────────┐  │
│  │ judge_botzone_ver.py   │   │ GameRunner 主循环          │  │
│  │ （in-process 调用）     │◄──│  full_input.log ⇄ 裁判     │  │
│  │ 发牌/叫分/牌型校验/结算 │   │  逐回合转发请求/收回响应    │  │
│  └────────────────────────┘   └───────┬──────────┬────────┘  │
│                                       │          │           │
│                    ┌──────────────────▼──┐   ┌───▼────────┐  │
│                    │ RandomAgent /       │   │ BotPlayer  │  │
│                    │ GreedyAgent /       │   │ 子进程 ×1  │  │
│                    │ PassPlayer（进程内） │   │ 每席位一个  │  │
│                    └─────────────────────┘   └───┬────────┘  │
└──────────────────────────────────────────────────┼───────────┘
                                                   │ stdin/stdout 逐行 JSON
                                          ┌────────▼────────┐
                                          │ cpp/build/bot.exe│
                                          │ （KEEP_RUNNING） │
                                          └─────────────────┘
```

三者的调用方式：

| 组件 | 运行位置 | 通信方式 | 启动开销 |
|------|----------|----------|----------|
| 裁判 | 主进程内 | 直接调用 `main(full_input)`，捕获 stdout | 仅一次 `import` |
| bot | 每席位一个子进程 | stdin/stdout 逐行 JSON，stderr 重定向到日志文件 | 每局每席位启动一次进程，重新加载模型 |
| 内置 agent | 主进程内 | 函数调用，输入同样形式的 request 字典 | 无 |

裁判被放在同一进程内是基于性能考虑：若每回合起一个 Python 子进程调裁判，一局几十回合会额外付出可观的解释器启动时间；进程内调用则只有一次函数调用。代价是裁判的 `exit()`、全局 `errored` 状态和 `print()` 输出需要 runner 做适配（见[第九节](#九keep_running-握手与裁判调用细节)）。

## 三、前置条件

### 3.1 Python

- Python ≥ 3.8（代码使用 `math.comb`、f-string 与 `subprocess` 的 `text=` 参数）。
- 只用标准库（`subprocess` / `json` / `random` / `argparse` / `io` / `datetime` 等），不需要安装任何 pip 包。
- runner 启动时会把 stdout/stderr 强制为 UTF-8，避免 Windows 下重定向输出时中文摘要因 ANSI 代码页（如 cp936）编码失败而中断。

### 3.2 裁判

裁判已内嵌，不需要仓库外的任何 judge 目录：

```
tools/runner/
├── runner.py
├── judge/
│   └── judge_botzone_ver.py     # 官方裁判逻辑，进程内 import
└── log/                         # 运行日志，自动创建，已被 .gitignore 忽略
```

runner 把 `tools/runner/judge` 插入 `sys.path` 后 `from judge_botzone_ver import main, errored, checkPokerType`；文件名与前缀保持平台原样以免混淆。

### 3.3 bot 可执行文件

bot 需要先按 [cpp.md 的构建章节](cpp.md#八构建)（或 `cpp/scripts/build_local.bat`）编译，产物为 `cpp/build/bot.exe`：

```
cpp/build/
├── bot.exe
├── *.dll                        # LibTorch/运行时依赖，与 exe 同目录
└── data/models/                 # 6 个 TorchScript 模型
```

runner 以 **bot.exe 所在目录** 作为子进程工作目录，因此：

- exe 同目录的 DLL 能被 Windows 正常找到；
- C++ 端固定的 `MODEL_DIR = "data/models"` 相对路径也能命中 `cpp/build/data/models/`。

默认席位配置就是 `cpp/build/bot.exe`（相对 `tools/runner/` 解析），所以 bot 构建好后直接运行即可；其他平台或自定义构建用 `--p0/--p1/--p2/--bot` 指定实际可执行文件。

### 3.4 模型

6 个模型放 `cpp/build/data/models/`：

```
first_jit.pt  second_jit.pt  third_jit.pt
landlord_jit.pt  landlord_up_jit.pt  landlord_down_jit.pt
```

导出方式见 [model-export.md](model-export.md) 与 [models/README.md](../models/README.md)，C++ 端的目录约定见 [cpp.md 的运行与模型放置](cpp.md#九运行与模型放置)。模型缺失或加载失败时 C++ 端不会报错退出，而是退回内置启发式策略继续打（叫分按手牌强度、出牌能压则压）；此时 runner 仍能跑完，但结果不代表模型强度。

### 3.5 日志目录

`tools/runner/log/` 在首次写日志时自动创建（`os.makedirs(..., exist_ok=True)`）。该目录与 `*.log` 都已被 `.gitignore` 忽略，不要把日志提交进仓库。

## 四、命令行：模式分派与参数表

### 4.1 模式分派

```
python tools/runner/runner.py [参数]
```

在仓库根（或 `tools/runner/` 下）都能运行。分派顺序如下：

1. 给了 `--vs` → **VS 对抗模式**。bot A 取 `--bot`，没给 `--bot` 时取 `--p0`；bot B 取 `--vs`。此模式下 `--games`、`--seed`、`--opponent`、`--p?`（除作为 bot A 来源外）都不生效。
2. 否则给了 `--p0/--p1/--p2` 中任意一个 → **按席位配置**，未给出的席位默认 `random`；`--bot` 与 `--opponent` 被忽略。
3. 否则给了 `--bot` → 三个席位为 `[--bot, --opponent, --opponent]`。
4. 什么都没给 → 默认 `[../../cpp/build/bot.exe, random, random]`。

然后按 `--games` 分：`--games 1` 为单局模式，`--games ≥ 2` 为批量模式（批量模式不能用 1 局）。VS 模式与 `--games` 无关。

`--p0/--p1/--p2/--bot/--vs` 里的相对 bot 路径按以下顺序解析：**先按 shell 当前目录**，当前目录下不存在时再按 `tools/runner/`。所以在仓库根运行 `--bot cpp/build/bot.exe` 和 `--bot ../../cpp/build/bot.exe` 都能命中；绝对路径原样使用。普通模式与 VS 模式都会在开跑前检查可执行文件是否存在，不存在则打印 `ERROR` 并退出码 1。

### 4.2 完整参数表

| 参数 | 类型 | 默认值 | 含义 |
|------|------|--------|------|
| `--p0 SPEC` | str | 未指定 | 席位 0 玩家：bot 可执行文件路径，或 `random` / `greedy` / `pass` |
| `--p1 SPEC` | str | 未指定 | 席位 1 玩家，取值同上 |
| `--p2 SPEC` | str | 未指定 | 席位 2 玩家，取值同上 |
| `--bot PATH` | str | 未指定 | 只测一个 bot 的快捷写法：该 bot 占 P0，P1/P2 用 `--opponent` 填充 |
| `--opponent {random,greedy,pass}` | str | `random` | `--bot` 模式下非 bot 席位的 agent 类型 |
| `--games N` | int | `1` | 对局总数；`1` = 单局模式，`≥2` = 批量模式 |
| `--seed S` | int | 未指定（随机） | 随机种子；单局模式直接作为该局发牌种子，批量模式作为随机数序列起点 |
| `--verbose`, `-v` | flag | 关 | 打印裁判调用、bot 收发的 JSON（`[JUDGE]`/`[BOT Pn IN]`/`[BOT Pn OUT]`） |
| `--timeout T` | float | `30.0` | 单次调用超时（秒）。**仅被解析和打印，当前不生效**，见第十节 |
| `--no-log` | flag | 关 | 关闭逐回合日志。单局/VS 模式生效；批量模式本来就不写逐回合日志。不影响 `batch_stats_*.log` / `vs_batch_*.log` |
| `--vs PATH` | str | 未指定 | bot B 路径；给出即进入 VS 对抗模式（必须同时给出 `--bot` 或 `--p0` 作为 bot A） |
| `--vs-games M` | int | `100` | VS 比赛场数，每场 2 局（同一副牌、互换地主/农民） |
| `--vs-seed S` | int | 未指定（随机） | VS 主种子，决定整条对局种子序列，便于复现 |
| `--vs-unpaired` | flag | 关 | 退回旧行为：一场的两局使用不同种子（不推荐，牌组差异会被误读成引擎强弱） |

说明：

- 单局模式 `--seed` 会同时用于 `random.seed()` 与裁判的 `initdata.seed`，因此同一 `--seed` 下内置 agent 的随机决策也可复现。
- 批量模式的每局种子由 `random.randint()` 依次生成；`--seed` 只固定这条随机数序列的起点，同一命令、同一 `--seed` 可复现，但**不能**把 `--seed` 与某个具体对局号简单对应起来。
- VS 模式用独立的 `random.Random(master_seed)` 生成对局种子，不污染全局随机状态，给定同一 `--vs-seed`，前 k 场的种子与结果固定。

## 五、三种运行模式

### 5.1 单局模式（`--games 1`，默认）

跑一局并在结束打印结果。默认开启逐回合日志（`game_state_*.log`、`json_io_*.log`），`--no-log` 可关闭。

```bash
# 仓库自带 bot 对两个随机 agent，完整 I/O
python tools/runner/runner.py --verbose

# 每个席位单独指定（bot vs greedy vs pass）
python tools/runner/runner.py --p0 ../../cpp/build/bot.exe --p1 greedy --p2 pass --seed 42

# 不依赖 bot 的空跑：三席位全用内置 agent
python tools/runner/runner.py --p0 random --p1 greedy --p2 pass --seed 2 --no-log
```

注意：**单局模式不要求存在 bot**。三个席位全用内置 agent 也能跑完并打印结果，只是没有 bot 席位时 `Result` 恒为 `LOSE`（判定规则见 7.1）。

`cpp/scripts/run_game.bat` 是调用本 runner 的薄封装：默认跑 1 局（`--verbose`、带逐回合日志），`run_game.bat 50` 跑 50 局批量，`--quiet` 去掉 verbose。

### 5.2 批量模式（`--games N`，N ≥ 2）

连续跑 N 局，按 bot 席位汇总胜率与分角色胜率，并写一份 `batch_stats_*.log`。批量模式**不写**逐回合日志（避免日志量爆炸），但汇总日志一定会写。

```bash
# bot 对随机对手 100 局
python tools/runner/runner.py --bot ../../cpp/build/bot.exe --games 100 --seed 42

# 不依赖 bot 的批量冒烟：3 局内置 agent 对局
python tools/runner/runner.py --p0 random --p1 greedy --p2 pass --games 3 --seed 1
```

每一局都会重新创建三个玩家对象：bot 席位会重启进程、重新加载模型，所以批量模式下每局都有进程与模型加载开销（100 局 = 100 次重启），`Games/sec` 会明显低于纯 agent 对局。

### 5.3 VS 对抗模式（`--vs`）

两个 bot 同牌对抗做引擎强弱验证。每场 = 2 局，**两局使用同一副牌**，只交换地主/农民：

| 对局 | P0（地主席位） | P1（农民） | P2（农民） |
|------|----------------|------------|------------|
| G1 | bot A，强制叫 3 | bot B，强制叫 0 | bot B，强制叫 0 |
| G2 | bot B，强制叫 3 | bot A，强制叫 0 | bot A，强制叫 0 |

叫分阶段被跳过（强制 P0 当地主），因此正常情况下每局底分固定为 3、分数是 3 的倍数（裁判判错走的兜底结算除外，见 9.3）。每场输出配对（同牌）结论，另外还有符号检验，用来看差异是否显著。

```bash
# 两个 bot 打 100 场（200 局），主种子固定，前 k 场可复现
python tools/runner/runner.py --bot ../../cpp/build/bot_a.exe \
    --vs ../../cpp/build/bot_b.exe --vs-games 100 --vs-seed 42

# 冒烟：同一构建自打 2 场
python tools/runner/runner.py --bot ../../cpp/build/bot.exe \
    --vs ../../cpp/build/bot.exe --vs-games 2 --vs-seed 1 --no-log

# 配对关闭（旧行为，不推荐）
python tools/runner/runner.py --bot ../../cpp/build/bot_a.exe \
    --vs ../../cpp/build/bot_b.exe --vs-games 50 --vs-seed 42 --vs-unpaired
```

VS 模式要求两个路径都是**已存在的可执行文件**，不能用 `random/greedy/pass` 当对手（会被文件存在性检查拒绝）。`--no-log` 只关闭逐回合日志，`vs_batch_*.log` 仍会写。

## 六、内置 agent

内置 agent 在 runner 进程内运行，直接消费裁判 request 字典。它们的作用是：不编译 bot 也能验证裁判与 runner 主循环（协议冒烟）、给 bot 提供基线对手、回归时确认 bot 能稳定赢下弱对手。它们**没有搜索也没有模型**，不能用来评估真实强度。

| agent | 叫分策略 | 出牌策略 | 典型用途 |
|-------|----------|----------|----------|
| `random` | 在当前最高叫分允许的合法叫分（0 及更高分）里均匀随机 | 首家出牌：出最小单张；跟牌：70% 过牌，30% 随机出一张（**可能压不住或牌型不符**） | 最弱基线、制造随机局面 |
| `greedy` | 按手牌强度打分：王×5 + 炸弹×3 + 2×2；强度 ≥6 叫 3、≥3 叫 2、≥1 叫 1，且不超过当前最高分 | 首家出：最小单张；跟牌：上一手是单张且有更大的牌时出最小能压的单张，否则过牌 | 比 random 稍强的固定基线 |
| `pass` | 永远叫 0 | 永远过牌 | 协议占位、构造"某席位不叫不压"的边界局面 |

行为细节与注意点：

- `random` 的 30% 跟牌是**随机单张**，并不保证大过上家、也不保证与上家同牌型，因此经常触发裁判的非法出牌判定；`pass` agent 在最需要它出牌（自己首家出牌）时也只会过牌，同样非法。这类对局会提前结束，结算走 runner 的"裁判出错兜底"路径（见 9.3）。
- 三席位全 agent 时单局 `Result` 恒为 `LOSE`、批量模式也不打印各席位胜率（因为没有 bot 席位可统计）；这是正常现象，只说明主循环与日志工作正常。
- agent 的随机性来自进程全局 `random`，会在每局开始时按该局种子重播；同一 `--seed` 可复现。

## 七、控制台输出与统计字段

### 7.1 单局模式

实际输出示例（bot 对两个 random）：

```text
Players: P0=bot.exe  P1=random  P2=random
Judge:   <repo>/tools/runner/judge/judge_botzone_ver.py (in-process)
Timeout: 30.0s (not enforced)

============================================================
Game Result
============================================================
  Result:    WIN
  Landlord:  P0
  Scores:    P0=+2  P1=-1  P2=-1
  Turns:     4
  Bot turns: 2
```

| 字段 | 含义 |
|------|------|
| `Players` | 三个席位的标签：agent 名为 `random`/`greedy`/`pass`，bot 为文件名（`os.path.basename`） |
| `Judge` | 内嵌裁判文件路径（运行时打印绝对路径） |
| `Timeout` | 传入的 `--timeout` 值；打印同时标注 `not enforced` |
| `Result` | `WIN` / `LOSE`。判定为：裁判给出的获胜席位（见下）是否属于 bot 席位集合 |
| `Landlord` | 地主席位 `P0/P1/P2`；`P-1` 表示本局未能确定地主（异常局） |
| `Scores` | 三席位积分，来自裁判 `finish` 的 `content`。普通模式地主为 ±2×底分×倍数、农民为 ∓1×底分×倍数；炸弹/火箭/春天会翻倍。VS 模式底分固定为 3 |
| `Turns` | 本次对局裁判被调用的次数，**包含叫分回合** |
| `Bot turns` | 其中由 bot 席位回应的回合数（叫分 + 出牌），全 agent 时为 0 |

`Result` 的具体判定是：先取裁判 `finish` 里**第一个正分席位**作为"获胜席位"，若它是 bot 席位则 `WIN`，否则 `LOSE`。由此带来一个边界：bot 是农民、而另一农民席位的下标更小（先被扫到正分）时，即使 bot 一方赢了也会打印 `LOSE`。批量与 VS 模式按席位分别统计，不受这个边界影响。

控制台不单独打印叫分和倍数，只打印最终 `Scores`。叫分过程可在 `json_io_*.log` / `game_state_*.log` 中逐回合看到；结算规则是：底分 = 叫分阶段最高叫分（三家都不叫则 P0 当地主、底分 1），倍数 = `2^(炸弹数+火箭数)`，再按春天/反春天各翻一倍；地主赢得 `+2×底分×倍数`、两个农民各 `-1×底分×倍数`，农民赢则符号相反。VS 模式固定 P0 叫 3，因此底分恒为 3。

若运行发生致命异常（如 bot 路径在启动阶段失效），会打印 `ERROR: ...` 代替上面这段结果。单局模式下每个玩家的非致命错误（bot 掉线后回退过牌等）不会打印，只能从 `json_io_*.log` 与 bot 的 stderr 日志里看到。

### 7.2 批量模式

实际输出示例（bot 对两个 random，5 局）：

```text
Players: P0=bot.exe  P1=random  P2=random
Judge:   <repo>/tools/runner/judge/judge_botzone_ver.py (in-process)
Timeout: 30.0s (not enforced)

Running 5 games...
  P0: bot.exe
  P1: random
  P2: random
============================================================

============================================================
Results (5 games, 4.6s)
============================================================
  Players: P0=bot.exe  P1=random  P2=random
  P0 (bot.exe):
    Win rate:    5/5 (100.0%)
    As landlord: 5/5 (100.0%)
  Games/sec:   1.09
  Batch log:   <repo>/tools/runner/log/batch_stats_<时间戳>.log
```

| 字段 | 含义 |
|------|------|
| `Running N games...` + 三行 `Pn: 标签` | 本次批量配置 |
| `Game %4d: WIN/LOSE/ERR landlord=Pn scores=[...]` | 进度行，每 10 局打印一次（`--verbose` 时每局都打）。`WIN` = 至少一个 bot 席位得分为正；`LOSE` = 没有 bot 席位得分为正；`ERR` = 本局发生致命异常（后面跟错误信息） |
| `Results (N games, Ts)` | 总局数与耗时 |
| `Win rate: w/N` | 该 bot 席位在 N 局中得分为正的局数（农民胜利时两个农民席位都得正分，因此都会被计入各自的胜场） |
| `As landlord: wl/lg` | 该席位当地主的局里的胜率；只有当过地主才打印这一行 |
| `As farmer: fw/fg` | 该席位当农民的局里的胜率；`fg = N - lg`，只有当过农民才打印这一行 |
| `Errors: k` | 发生**致命异常**的局数，最多列 3 条 `seed=...` 与错误摘要；非致命回退错误不计入 |
| `Games/sec` | 吞吐，用来观察 bot 每局重启 + 模型加载的开销 |
| `Batch log` | 汇总日志路径 |

没有 bot 席位时（全 agent 批量），不会打印 `Win rate`/`As landlord`/`As farmer` 块，其余字段照常。

### 7.3 VS 模式与配对统计

实际输出示例（同一构建自打 2 场）：

```text
VS Mode: bot.exe  vs  bot.exe
Matches: 2  (4 games with swapped roles)
Master seed: 1
============================================================

============================================================
VS Results (2 matches, 4 games, 12.2s)
============================================================
  bot.exe (<repo>/cpp/build/bot.exe):
    As landlord:  0W / 2L  (0.0%)
    As farmer:    2W / 0L  (100.0%)
    Overall:      2W / 2L  (50.0%)
  bot.exe (<repo>/cpp/build/bot.exe):
    As landlord:  0W / 2L  (0.0%)
    As farmer:    2W / 0L  (100.0%)
    Overall:      2W / 2L  (50.0%)

  注意: 配对模式下 Overall 天然接近 50%（每对牌两 bot 各当一次地主），
        判断强弱请看下面的配对结论，而不是 Overall。

  配对结果（同一副牌交换角色, 2 对）:
    A 两局全胜:   0    B 两局全胜:   0    同向(牌力/角色决定):   2
    地主净分差(A−B): 合计 +0   均值 +0.00/对
  VS log:      <repo>/tools/runner/log/vs_batch_<时间戳>.log
```

汇总表字段：

| 字段 | 含义 |
|------|------|
| `As landlord:  wW / lL (%)` | 该 bot 当地主时的胜负与胜率（每场 1 次地主） |
| `As farmer:    wW / lL (%)` | 该 bot 当农民时的胜负与胜率（每场 1 次农民；同场两个农民席位属于同一 bot） |
| `Overall` | 上面两者合计；配对模式下每场两个 bot 各当一次地主，所以整体天然接近 50%，**不能用 Overall 判断强弱** |
| `Match` 进度行 | 每 10 场打印一次；每场给出 G1/G2 的 `ll=Pn scores=[...]`，异常局显示 `ERR(...)` |

配对统计（同一副牌，A/B 互换角色）怎么读：

| 字段 | 含义 |
|------|------|
| `A 两局全胜` | 该副牌上 A 当地主赢、且 A 当农民也赢（A 两局都赢） |
| `B 两局全胜` | 同上，B 两局都赢 |
| `同向(牌力/角色决定)` | 两局都是地主赢或都是农民赢，说明该副牌的结果由牌力/角色决定，与两个引擎强弱无关 |
| `异常` | 该配对中至少一局异常（前面 `ERR`），不参与结论 |
| `决定性配对里 A 胜率` | `A 两局全胜 / (A 两局全胜 + B 两局全胜)`；同向配对不计入 |
| `符号检验 p` | 对决定性配对的二项双侧符号检验 p 值（零假设 A/B 各 50%）。p 小说明差异不像随机波动，可作为 A/B 强弱有别的证据 |
| `地主净分差(A−B)` | 每对牌里（A 当地主得分 − B 当地主得分）的累加与均值；正数偏 A、负数偏 B |

`--vs-unpaired` 下不做同牌配对，控制台会明确提示配对统计不适用，`vs_batch_*.log` 里也不会写 `Paired` 段。

### 7.4 退出码

| 情况 | 退出码 |
|------|--------|
| 正常运行结束（含单局结果为 `LOSE`） | 0 |
| `--help` | 0 |
| bot 可执行文件不存在（普通模式预检查 / VS 模式校验） | 1 |
| `--vs` 没给 bot A（既无 `--bot` 也无 `--p0`） | 1 |
| 未捕获的致命异常（如日志目录不可写、bot 进程启动失败） | 1（Python traceback） |

## 八、日志文件

所有 runner 日志写入 `tools/runner/log/`（自动创建、git 忽略）。时间戳格式为 `YYYYmmdd_HHMMSS`。

| 文件 | 产生模式 | 内容 | 关闭方式 |
|------|----------|------|----------|
| `game_state_<ts>.log` | 单局、VS（逐回合日志开启时） | 每回合：轮次、席位、角色（叫分/地主/农民）、该席位手牌、动作；文末附 `本局出牌统计`（每席位各牌型出牌次数） | `--no-log` |
| `json_io_<ts>.log` | 同上 | 每回合：发给该席位的裁判 request JSON 与玩家返回的 response JSON（记录的是裁判请求，不含首回合 KEEP_RUNNING 外层包装） | `--no-log` |
| `batch_stats_<ts>.log` | 批量模式 | 汇总：玩家配置、每席位当庄次数/获胜次数/各牌型出牌次数（按频率排序） | 无开关，批量必写 |
| `vs_batch_<ts>.log` | VS 模式 | 摘要：双方标签与路径、场数、主种子、耗时、分角色胜负、配对结论与符号检验、逐场逐局明细（`seed`/`spec`/`landlord`/`scores`/`ERROR`） | 无开关，VS 必写 |
| `bot_stderr_p<seat>.log` | 有 bot 席位时 | bot 子进程的 stderr，写在 **bot.exe 所在目录**（不是 `tools/runner/log/`）；每次启动以写模式打开（覆盖） | 无开关 |

注意：

- `--no-log` 只关逐回合文件；`batch_stats_*` / `vs_batch_*` 是汇总记录，仍会写。
- 日志文件名只精确到秒。VS 模式下一场两局若在同一秒内结束，两次 `GameLogger` 会用同一时间戳、以写模式打开同名文件，后一局覆盖前一局的逐回合日志；排查快速对局时注意这一点。
- `bot_stderr_p<seat>.log` 每局覆盖。批量模式每局重启 bot，因此它只保留**最后一局**的 stderr。
- 若日志目录不可写，单局/VS 模式会在开局创建日志文件时抛出异常；批量模式的汇总日志在全部对局结束后写，失败会让进程以异常结束。此时可用 `--no-log` 跳过逐回合日志。

## 九、KEEP_RUNNING 握手与裁判调用细节

### 9.1 bot 子进程

`BotPlayer` 用 `subprocess.Popen([exe], stdin=PIPE, stdout=PIPE, stderr=stderr_file, text=True, bufsize=1, cwd=exe所在目录)` 启动，每席位一个进程，每局重新启动、局末关闭 stdin 并 kill。stdout 按行读取，stderr 直接重定向到文件。

给 bot 的输入分两种形态：

首次请求（该 bot 进程的第一次调用）：

```json
{"requests": [<裁判 request>], "responses": []}
```

后续请求（同一进程的第二次及以后）：

```json
<裁判 request>
```

例如叫分请求与出牌请求（真实对局日志节选）：

```json
{"own": [43, 20, 2, 14, 22, 28, 51, 44, 10, 24, 0, 46, 41, 26, 1, 18, 13], "bid": []}

{"history": [[], []], "own": [...], "publiccard": [29, 53, 30],
 "landlord": 0, "pos": 0, "finalbid": 3}

{"history": [[0], [5]]}
```

C++ 端通过"输入里有没有 `requests` 键"区分全量重建与增量更新（`BotzoneIO::read`），所以 runner 只在首次调用时套一层 `requests`。首次请求可能是叫分请求（该 bot 参与叫分），也可能直接是出牌请求（叫分已在它之前结束，例如别人先叫了 3 分）。

bot 的每回合输出是一行 JSON：

```json
{"response": 3}
{"response": [0, 1, 2]}
{"response": []}
```

叫分是整数，出牌是牌 ID 数组，空数组为 PASS。

### 9.2 读取响应的规则

每次 `respond` 最多读 10 行，规则是：

1. 以 `>>>BOTZONE_REQUEST_KEEP_RUNNING<<<` 开头的行直接跳过（C++ 端每回合决策后会打印该标记并继续等待下一行输入，runner 不需要对标记做任何回应，只把它当噪声）；
2. 空行或其它不以 `{` 开头的行跳过；
3. 第一条以 `{` 开头的行作为 JSON 响应解析，取其中的 `response` 字段；
4. 读 10 行仍未拿到 JSON，或读流遇到 EOF，或 JSON 解析失败 → 抛 `RuntimeError`。

`RuntimeError` 会被主循环捕获：该席位本回合按"叫 0 分 / 不出"处理，并把一条错误记入局内 `stats["errors"]`（注意这条信息不会出现在控制台汇总里，见第十节）。如果回退的"不出"本身非法（例如轮到自己首家出牌），裁判会判该玩家出牌错误，对局随后按 9.3 的路径结束。

### 9.3 裁判的进程内调用

每个回合 `GameRunner.call_judge(full_input)` 做四件事：

1. 把裁判模块的全局 `errored = [None, None, None]` 重置；
2. 用 `io.StringIO` 临时接管 `sys.stdout`，调用 `judge_botzone_ver.main(full_input)`；
3. `main()` 内部通过 `exit(0)`/`sys.exit()` 结束一次调用，runner 在 import 前把这两个入口改成抛 `SystemExit`，在这里捕获；同时吞掉裁判在边界状态下可能出现的 `IndexError` / `KeyError` / `TypeError`；
4. 恢复 stdout，取捕获文本的**最后一行** JSON 作为本次命令（`request` 或 `finish`）返回。

`full_input` 结构：

```python
{
  "log": [ <display 条目>, <response 条目>, <display 条目>, ... ],
  "initdata": {"seed": "<seed 字符串>"},
}
```

- 裁判用 `log` 中交替的"展示块 / 响应块"推进对局：`display` 来自裁判自己的输出，`response` 是玩家动作 `{str(player): {"verdict": "OK", "response": ...}}`。
- `initdata.seed` 会先被裁判 `json.loads` 再用于 `random.seed()`，所以同一 seed 得到同一副牌。

裁判的两种正常输出：

```json
{"command": "request", "content": {"<席位>": <请求>}, "display": {...}}
{"command": "finish", "content": {"0": <分>, "1": <分>, "2": <分>}, "display": {"landlord": n, ...}}
```

`finish` 后 runner 以 `content` 的三个分数为积分，取第一个正分席位为获胜方。

**兜底路径**：若玩家非法出牌/非法叫分，裁判的 `setError()` 只有在**三名玩家都已出错**时才自己打印 `finish`；只错一人时它什么都不打印。runner 捕获到空输出后用 `_build_error_finish()` 合成一个 `finish`：

- 从对局日志前 3 个叫分响应推断地主（取最高叫分，全部不叫则 P0）；
- 地主出错 → 地主 `-2`、两个农民各 `+1`；
- 农民出错 → 地主 `+2`、两个农民各 `-1`；
- 无法确定地主，或虽确定地主但三名玩家都没有被标记出错（例如裁判自身异常被吞掉）→ 三家都是 `0`、`winner` 为 `None`。

这个合成结算的分数**不乘底分与倍数**（固定 ±1/±2）。注意底分为 1、无炸弹/春天的正常结算也是 ±2/±1，因此不能只凭分数判断是否走了兜底路径；`Turns` 明显偏短、`Landlord: P-1`、以及 `json_io_*.log` 中出现非法动作，才是更可靠的线索。

### 9.4 VS 模式的强制叫分

VS 模式用 `ForcedBidWrapper` 包住全部三个席位，`force_bids = {0: 3, 1: 0, 2: 0}`。它保证"P0 一定当地主、其余两家一定是农民"，从而每场两局可以同牌互换角色：

- **P0（bot A 或 bot B）**：裁判会真实发来叫分请求。wrapper 先把请求**原样转给 bot**（让 bot 完成首回合初始化、加载叫分模型、退出首回合全量重建状态），然后丢弃 bot 的真实叫分值，返回固定 `3`。裁判收到 3 分立刻结束叫分。
- **P1 / P2（农民席位）**：裁判不会再发叫分请求（叫分已在 P0 处结束）。wrapper 在该席位**第一次非叫分调用**时，先注入一个合成叫分请求

  ```json
  {"own": [<该席位手牌>], "bid": [3]}
  ```

  交给 bot（同样为了让 bot 走一次叫分初始化），随后再把真实的出牌请求转给 bot，最后给裁判的返回值固定为 `0`。合成请求的 `bid` 长度为 1，所以 bot 内部据此推断的叫分位置可能与该席位实际位置不同；紧接着的真实出牌请求带有 `publiccard` / `landlord` / `pos` / `finalbid`，C++ 端会据此把出牌阶段的位置与状态纠正过来，因此这个偏差只影响叫分阶段内部状态。
- 结果：每局 `finalbid = 3`，底分 3，分数为 3 的倍数（例如地主 ±6/农民 ∓3，有炸弹再翻倍）。普通模式则使用真实叫分结果作为底分（三家都不叫时 P0 默认当地主，底分 1）。

## 十、已知限制

1. **`--timeout` 不生效**。参数只被解析、打印，`GameRunner.timeout` 存下来后没有被任何逻辑使用；bot 的 stdout 用阻塞式 `readline()` 读取，没有读超时。若 bot 卡死（例如等待不存在的输入、死循环），runner 会一直阻塞，需要手动终止进程。裁判是进程内调用，同样无法被超时打断。
2. **bot 运行时故障不会显式报错**。开跑前只检查可执行文件是否存在：缺 DLL 等导致进程根本起不来时，`Popen` 抛出的 `OSError` 没有被捕获，会以 traceback 结束；进程能起来但运行中崩溃/退出的，表现为"读响应 EOF/非法 JSON → 回退过牌"，对局可能以兜底结算提前结束。这两类问题都要看 bot.exe 所在目录的 `bot_stderr_p<seat>.log` 和 `json_io_*.log`。模型缺失时 C++ 端会静默退回启发式策略，runner 无法区分。
3. **非致命错误不汇总**。玩家回退错误记录在每局 `stats["errors"]`，但单局、批量、VS 的汇总都不打印它；批量汇总的 `Errors`、VS 进度行/逐场明细里的 `ERR` 以及配对里的 `异常` 都只统计导致整局异常终止的致命错误。
4. **单局 `Result` 的判定边界**：以"第一个正分席位是否 bot"判定，而不是"是否有 bot 席位得分为正"。bot 是农民而另一农民下标更小时可能误显示 `LOSE`（详见 7.1）；批量/VS 的按席位统计不受影响。
5. **日志时间戳只到秒**：VS 每场两局若在同一秒内跑完，逐回合日志同名覆盖，只保留后一局。需要完整两局日志时放慢对局（例如 `--verbose` 会增加 I/O）或分场运行。
6. **批量每局重启 bot**：每局都重新起进程、重新加载模型，批量吞吐明显低于单进程常驻推理；`Games/sec` 可供对照。
7. **批量模式下 `--no-log` / 逐回合日志**：批量模式构造 `GameRunner` 时没有传 `enable_log`（默认关闭），所以批量模式永远不写逐回合日志，`--no-log` 在批量模式下无实际差别。`--timeout` 在批量与 VS 模式下也没有被传给 `GameRunner`（即便传了也不会生效）。
8. **兜底结算与真实结算可能同形**：裁判判错后合成 `finish` 的分数固定为 ±1/±2，而底分为 1 的正常结算也是 ±1/±2，且 `Result` 仍按正分席位打印 `WIN/LOSE`；需要结合 `Turns` 是否异常短、`json_io_*.log` 里是否有非法动作来判断。
9. **批量统计的边界情况**：出现"地主未确定"的异常局时，`As farmer` 的分母 `N - lg` 会把这局算进去，而它的胜负只体现在 `Win rate` 中；`--games` / `--vs-games` 小于 1 时会直接报 `ERROR` 并以退出码 1 结束（不做截断或补默认值）。
10. **相对路径基准**：`--p0/--p1/--p2/--bot/--vs` 的相对路径以 `tools/runner/` 为基准，而不是调用时的 shell 当前目录；在仓库根写 `--bot ../../cpp/build/bot.exe` 才能命中 `cpp/build/bot.exe`。
11. **Windows 编码**：runner 启动时已把 stdout/stderr 强制为 UTF-8（`errors="replace"`）；日志文件本身也是 UTF-8。用旧版 Python 或直接把输出喂给非 UTF-8 消费者时，仍可能看到乱码。bot 的 stdin/stdout 是 JSON，内容为 ASCII，不受影响。
12. **单局与批量不打印本局种子**：结果块里没有 seed，只有异常局的错误行、以及 VS 模式的 `vs_batch_*.log` 会记录种子。要复现某局需自己保留命令里的 `--seed`（批量模式还需知道是第几局）。
13. **VS 模式只能用两个 bot**：bot A/bot B 都必须是存在的可执行文件，不能拿 agent 当陪练；agent 只能用于普通模式的席位。
