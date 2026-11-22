# 第三方代码与致谢

本仓库中部分代码源自第三方项目或平台，按原许可使用，说明如下。

| 位置 | 来源 | 许可 | 说明 |
|------|------|------|------|
| `python/douzero/` | [DouZero](https://github.com/kwai/DouZero)（快手 AI 平台） | Apache-2.0 | DMC 分布式训练框架、斗地主环境（发牌/牌型/动作生成）、评估与模拟框架。本项目在其基础上扩展了 8×15 编码、神谕头、对手建模与多监督头模型。 |
| `cpp/src/jsoncpp.cpp`、`cpp/src/jsoncpp/` | [jsoncpp](https://github.com/open-source-parsers/jsoncpp)（amalgamated 版本） | MIT / Public Domain | JSON 解析。Botzone 编译环境自带同名库，此处保留一份以保证本地可编译。 |
| `tools/runner/judge/judge_botzone_ver.py` | Botzone 平台斗地主裁判逻辑 | 版权归 Botzone / 原作者 | 仅用于本地对局与 runner 测试，不参与线上提交。文件名与前缀保留以免与平台代码混淆。 |
| 算法思想与部分实现 | [DouZero](https://github.com/kwai/DouZero)、[PerfectDou](https://github.com/NKUPHW/PerfectDou)、DouMH | — | 多监督头、神谕（oracle）特征、对手建模等设计参考了上述工作；其中 DouMH 公开源码中的部分思路被本项目改编沿用，致谢见下节。 |

## 关于 DouMH

DouMH（S. Han, Z. Jin, Q. Liu, C. Xu, "DouMH: Improving DouZero with Multiple Heads", CISAI 2024, DOI [10.1145/3703187.3703237](https://doi.org/10.1145/3703187.3703237)）是本项目多监督头、神谕特征等设计的重要灵感来源。其代码在笔者获取时由原作者以源码压缩包形式公开发布，该仓库现已不可访问；这里保留致谢与论文引用，感谢原作者的开源工作。

## 训练数据

`python/eval_data.pkl` 为 1000 局固定发牌的评测集，由本项目脚本生成，不含第三方数据。
