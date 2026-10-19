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
[✅] 工程骨架            目录结构、脚本、远程仓库
[🔄] 源码调研            vllm-plugin-FL / FlagGems 优化面测绘
[⬜] 算力申请材料        技术方案 + 团队介绍 → 算力申请-[队名].pdf
[⬜] 环境复现            两卡 baseline 自测
[⬜] 优化与提交
```

**关键路径**：算力申请截止 **11-10**，但遵循**先到先得**且审核需 5 个工作日——
材料必须在 **10-21 前**投出，否则拿到卡的时间不够跑完两轮实验。

---

## 快速开始

```bash
# 1. 拉取赛事元数据（可复现，支持 diff 检测官方改动）
python scripts/fetch_race.py --all
python scripts/fetch_race.py --diff        # CI 友好：有变动则 exit 1

# 2. 查看官方基线 + 评分规则，并跑自检
python scripts/score.py --demo

# 3. 算一次成绩（CSV 来自 benchmark_throughput_serve.py 的输出）
python scripts/score.py --before bench/summary_base.csv \
                        --after  bench/summary_opt.csv  \
                        --acc 0.958 --expert 85
```

两个脚本都只用标准库，无第三方依赖。

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
│   └── research/               源码调研（优化面测绘）
├── proposals/                  算力申请材料正文
├── scripts/                    工具脚本（无依赖）
│   ├── fetch_race.py           赛事元数据抓取 + diff
│   └── score.py                成绩计算器 + 护栏校验
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
