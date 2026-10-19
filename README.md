# MiniCPM5-2B 推理吞吐优化 · FlagOS 开放计算全球大赛 S2 赛道二

> **FlagOS Open Compute Global Challenge · Season 2 · Track 2**
> 在固定框架（`vllm-plugin-FL` + `FlagGems`）、固定硬件（天数 BI-V150 / 沐曦曦云 C500）、
> 固定 serve 参数的前提下，把 **MiniCPM5-2B** 的 `total tokens/s` 顶上去。

| | |
|---|---|
| 赛事平台 | <https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn> |
| 模型 | [`OpenBMB/MiniCPM5-2B`](https://modelscope.cn/models/OpenBMB/MiniCPM5-2B) |
| 框架 | [`vllm-plugin-FL`](https://github.com/flagos-ai/vllm-plugin-FL) `flagos-2026-s2` |
| 算子库 | [`FlagGems`](https://github.com/flagos-ai/FlagGems) `v5.3.5` |
| 提交截止 | **2026-11-20 23:59 (UTC+8)** |

---

## 当前状态

```
[✅] 赛事情报收集        平台 API 全量抓取，规则/基线/赛程本地化
[✅] 工程骨架 + 远程仓库  GitHub: SOLECA794/minicpm5-2b-infer-opt-flagos
[✅] 源码调研            vllm-plugin-FL / FlagGems 优化面测绘（附 file:line 证据）
[✅] 根因定位            天数 16k 的 599s TTFT → attention 后端走错路径
[✅] 技术方案正文         proposals/01-initial-technical-plan.md
[✅] PDF 生成流水线       scripts/build_proposal.py（缺团队信息则拒绝生成）
[✅] 团队介绍            单人参赛（苏子恒），proposals/team.yaml 已填全
[✅] 算力申请材料        proposals/算力申请-苏子恒.pdf（9 页，0.53 MB）
[⬜] 提交算力申请         ← 需在平台上传，先到先得
[⬜] 环境复现 / 优化      ← 等算力
```

**关键路径**：算力申请截止 **11-10**，但遵循**先到先得**且审核需 5 个工作日——
材料必须在 **10-21 前**投出，否则拿到卡的时间不够跑完两轮实验。

### 核心发现

天数 BI-V150 与沐曦 C500 的 **attention 后端不同**：

| | 天数 | 沐曦 |
|---|---|---|
| 配置指定 | `[flagos, vendor:iluvatar, reference]` | `[vendor:metax]` |
| 实际后端 | **vLLM 通用 `TRITON_ATTN`** | **MACA 原生 `FLASH_ATTN`** |

天数源码注释直接写着 *"flash_attn is not available on iluvatar"*。
而 FlagGems v5.3.5 **已经带完整的 FA2 实现**，vllm-plugin-FL **也已经写好了接入代码**
（`AttentionFLBackend`，857 行），只是默认值 `VLLM_FL_USE_FLAGGEMS_ATTN="0"` 把它关掉了。

详见 [`docs/research/root-cause-tianshu-16k.md`](docs/research/root-cause-tianshu-16k.md)。

---

## 快速开始

```bash
# 1. 拉取赛事元数据（可复现，支持 diff 检测官方改动）
python scripts/fetch_race.py --all
python scripts/fetch_race.py --diff        # CI 友好：有变动则 exit 1

# 2. 查看官方基线 + 评分规则，并跑自检
python scripts/score.py --demo

# 3. M0 门禁：先确认这台机器复现得出官方 baseline（误差 ≤1%）
python scripts/score.py --verify-baseline bench/summary_m0.csv --vendor tianshu

# 4. 算一次成绩（CSV 来自 benchmark_throughput_serve.py 的输出）
python scripts/score.py --before bench/summary_base.csv \
                        --after  bench/summary_opt.csv  \
                        --acc 0.958 --expert 85
```

### 生成算力申请 PDF

```bash
# 1. 填写团队信息（把 TODO 全部替换掉）
$EDITOR proposals/team.yaml

# 2. 先校验，再生成
python scripts/build_proposal.py --check
python scripts/build_proposal.py
#   → proposals/算力申请-<队名>.pdf
```

`build_proposal.py` 在 `team.yaml` 还有任何 `TODO` 时会**拒绝生成**，
防止误提交未填完的材料。技术方案正文在
[`proposals/01-initial-technical-plan.md`](proposals/01-initial-technical-plan.md)，
PDF 生成时自动拼接团队介绍。

依赖：`reportlab`、`PyYAML`。字体自动从 Windows 字体目录解析中文字形，
找不到时退回内置 Adobe CJK 字体。

### 校验生成的 PDF

```bash
python scripts/pdf_textdump.py proposals/算力申请-<队名>.pdf
```

四个脚本都只用标准库（`build_proposal.py` 额外需要 reportlab + PyYAML）。

---

## 目录结构

```
.
├── docs/
│   ├── competition/            赛事官方信息（本地镜像 + 提炼）
│   │   ├── README.md           速查：规则、评分、红线
│   │   ├── rules-s2-track2.md  赛题原文（Markdown 化，含全部命令）
│   │   ├── baselines.md        官方 baseline + 差距诊断 + 目标
│   │   ├── timeline.md         倒排工期 + 风险登记
│   │   ├── raw/                平台 API 原始 JSON
│   │   └── *.png               官方 baseline 截图
│   └── research/               源码调研（全部附 file:line 证据）
│       ├── plugin-surface-map.md     vllm-plugin-FL 优化面测绘
│       ├── flaggems-op-map.md        FlagGems 算子覆盖与机会
│       └── root-cause-tianshu-16k.md 599s TTFT 根因分析
├── proposals/                  算力申请材料
│   ├── 01-initial-technical-plan.md  初步技术方案（评审 60%）
│   ├── team.yaml                     团队介绍数据源（评审 40%），已填全
│   ├── 算力申请-苏子恒.pdf            提交件（9 页，0.53 MB）· 含联系方式，已 gitignore
│   └── team.example.yaml             虚构示例，用于冒烟测试 PDF 流水线
├── scripts/                    工具脚本
│   ├── fetch_race.py           赛事元数据抓取 + diff
│   ├── score.py                成绩计算器 + 护栏校验 + M0 baseline 门禁
│   ├── build_proposal.py       Markdown → PDF（含 TODO 门禁）
│   └── pdf_textdump.py         PDF 文本导出（无阅读器时校验用）
├── src/                        我们自己的优化实现
├── configs/                    实验配置
├── benchmarks/results/         实验数据
└── upstream/                   官方仓库（gitignored，可重新 clone）
    ├── vllm-plugin-FL/         branch flagos-2026-s2
    └── FlagGems/               tag v5.3.5
```

---

## 评分公式

```
总分 = 性能分(天数) + 性能分(沐曦) + 创新分

性能分 = 100 × 提升比例 × 0.70        # 单场景提升 < 1% 不计分
创新分 = 专家评分 × 0.30              # 技术报告创新性

每个 vendor 的性能分 = mean(4k 场景, 16k 场景)
```

**护栏**（任一不过即成绩无效）：

```
精度   math_500 Level 3  acc ≥ 0.95   (baseline 0.962)
TTFT   ≤ baseline × 1.01
```

### 为什么要有 M0 门禁

评分是**相对提升**，分母就是官方 baseline。如果实验机器本身偏差 20%
（共享卡、降频、`VLLM_PLUGINS` 没设对、漏了天数的 `--compilation-config`、
框架 commit 与 pinned 版本不一致），后面每一次「提升」都是错误分母的产物。
所以拿到算力后的**第一件事**是 `--verify-baseline`，吞吐与 TTFT 双双落在
±1% 内才允许开始优化。这个门禁同时把「机器状态波动」和「真的优化了」区分开。

---

## 绝对红线

* 换模型（只能用 MiniCPM5-2B）
* 改 `vllm serve` 参数
* 改超参开量化 / 投机采样
* 调优 benchmark 脚本
* 无算子优化就切换算子 / 删除框架的算子选择逻辑
* 直接 merge 上游最新分支代码
* 报告与代码不一致、方案无法复现

详见 [`docs/competition/rules-s2-track2.md`](docs/competition/rules-s2-track2.md) §4.1。

---

## 复现上游环境

```bash
mkdir -p upstream && cd upstream
git clone --depth 1 --branch flagos-2026-s2 https://github.com/flagos-ai/vllm-plugin-FL.git
git clone --depth 1 --branch v5.3.5        https://github.com/flagos-ai/FlagGems.git
```

---

## License

Apache-2.0（与上游 `vllm-plugin-FL` / `FlagGems` 保持一致）。
