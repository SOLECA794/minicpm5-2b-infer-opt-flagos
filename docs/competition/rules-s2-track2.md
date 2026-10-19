# 赛道二官方赛题原文（Markdown 化）

> 来源：`https://flagos.io/flagos/api/v1/races/539vlt2p` → `data.content.content` → `tabs[0].content`（HTML）
> 原始 JSON：[`raw/race_539vlt2p.json`](raw/race_539vlt2p.json)
> 抓取时间：2026-10-19 ｜ 平台标注 `metadata.last_updated = 2026-07-28T11:16:36`
> 本文档为**忠实转写**，未删改技术内容。命令块保持原样（含官方笔误）。

---

## 1、赛道综述

千亿级大模型规模化落地后，高算力、高显存开销带来的推理延迟高、吞吐低、资源利用率不足等问题，严重制约了产业规模化应用。在固定硬件资源、保证推理精度与稳定性的前提下，实现高效分布式端到端推理，已成为行业核心痛点。本次比赛聚焦于在固定硬件集群环境中，对给定的大模型进行高效推理优化。

参赛者需基于提供的 FlagOS 推理框架（<https://github.com/flagos-ai/vllm-plugin-FL>，**branch: `flagos-2026-s2`**），与高性能算子库 FlagGems（<https://github.com/flagos-ai/FlagGems>，**tag: `v5.3.5`**），充分发挥硬件性能潜力，实现模型推理在速度、资源利用率等方面的显著提升。

## 2、竞赛须知

本赛题旨在探索大语言模型在推理上的性能提升方案，要求参赛选手基于 FlagOS 推理框架，对面壁模型 **MiniCPM5-2B** 在 **天数 BI-V150**、**沐曦-曦云 C500（显存 64 GB）** 上的推理性能提升，充分发挥硬件潜能。

* 参赛团队围绕赛题目标提出完整技术方案，并在**全部评测集**上展开实验验证，提交相应源码、技术报告和复现说明，按照文档能复现性能情况。
  * 为统一模型加载与运行环境，保障结果的公平性与可复现性，参赛团队**必须使用 FlagOS 推理框架指定版本**作为大语言模型完成本次竞赛任务，否则视为成绩无效。
* 所有参赛队伍须通过官方网站完成报名，报名成功后方获得参赛资格；未在官网报名、通过其他渠道提交的作品，均不纳入最终评审范围。

### 2.1 报名须知

本赛道欢迎国内外企业、高校、科研院所、创新团队、科研开发者等人员参加。

* 每一个参赛队伍**不超过 3 个成员**，支持单人组队。
* **算力支持**：
  * 本赛道需使用官方提供的**统一算力支持**，本着"不浪费算力资源"的原则，参赛团队请在规定时间内提交申请材料。申请材料需至少包含：**初步技术方案、团队介绍（团队成员个人简历以及团队成员参与相关项目和竞赛的经历说明等）。** 组委会将依据综合评审结果，对算力资源进行统一分配。
  * **天数算力**需审批通过后在线上实验室平台（<https://flagos.io/OnlineLab>）预约使用。
  * **沐曦算力**需领取算力券后登录模力方舟（<https://ai.gitee.com/compute>）平台兑换使用（温馨提示：算力券额度有限，未开发时段请及时关机。另为保证数据不丢失，建议每 36 小时开关机一次）。
* **技术指导**：来自众智 FlagOS 社区成员单位的北京智源人工智能研究院专家为本次竞赛提供全程技术指导。

### 2.2 竞赛时间安排（UTC+8）

| 阶段 | 时间 |
|---|---|
| 报名阶段 | 2026-07-19 ── 2026-11-10 |
| 官方算力申请 | 2026-09-14 ── 2026-11-10 |
| 算力评审及发放 | 2026-09-21 ── 2026-11-15 |
| 开发与提交阶段 | 2026-09-21 ── 2026-11-20 |
| 评审阶段 | 2026-12 月初 |
| 结果发布 | 2026-12 月中旬 |

*\*时间如有调整，将及时更新信息，最终安排以官方通知为准*

### 2.3 报名操作流程

1. 打开赛事平台链接：<https://flagos.io/race-detail-season2?id=8q4m2x7p&lang=cn>
2. 找到 FlagOS 开放计算全球大赛「赛季二」，点击立即报名

### 2.4 算力申请

参赛队伍可申请使用大赛官方提供算力资源，选手需以 **`算力申请-[队名].pdf`** 命名提交**技术方案报告**开展算力申请审核。材料提交后，主办方将在 **5 个工作日内**完成审核。

> ⚠️ 备注：官方算力资源有限，申请遵循**先到先得**原则。

#### 2.4.1 申请时间

2026-09-14 ── 2026-11-10（UTC+8）

## 3、赛题概览

本赛题聚焦在严格保持模型精度与不显著增加时延的前提下，对面壁模型 **MiniCPM5-2B** 进行极限推理吞吐优化。参赛者需综合利用并行策略、显存管理、计算内核等全栈技术，在固定框架与硬件上挖掘最大性能潜力。

## 4、赛题详情

### 4.1 赛题要求

* 参赛队伍基于 FlagOS 推理框架（<https://github.com/flagos-ai/vllm-plugin-FL>）对面壁模型 MiniCPM5-2B 进行推理相关优化。本阶段重点考察选手在不影响计算精度的前提下，对**吞吐**指标的性能优化能力。
* **模型链接**：ModelScope <https://modelscope.cn/models/OpenBMB/MiniCPM5-2B>
* **优化目标**：提升 `total tokens/s`。
* **基本指标**：保持精度。
* **优化手段说明**：
  * 允许的优化手段包括但不限于：**显存管理（KV Cache 优化、动态批处理）、计算逻辑优化（算子融合、内核重写）、自行开发的量化策略**等需要在框架有开发量的手段进行优化。提交的技术报告中的策略必须体现在提交代码中，并在最终版本使用上对应的优化策略。
  * 优先鼓励具有**泛化能力高**的优化策略。
  * 要求 `vllm serve` 的参数和基线一致。
  * **禁止**通过修改超参开启量化、投机采样。
  * **禁止** benchmark 脚本调优。
  * **禁止**没有算子优化的情况下切换算子，删掉框架主要算子选择逻辑等方式来进行模型性能优化。
  * **禁止**通过直接 merge FlagOS 推理框架最新分支代码获得性能优化。
* **成绩无效情况说明**：
  * 精度降幅超过规定范围
  * 模型核心能力出现明显下降
  * 输出结果异常或无法完成指定 Benchmark
  * 修改模型行为，导致评测结果失去可比性
  * 使用禁止的优化策略
* **开发环境**：
  * 模型：`/workspace/MiniCPM5-2B`
  * evalscope 评测数据集：`/workspace/evalscope-datasets/math_500`
  * vllm-plugin-FL：`/workspace/vllm-plugin-FL`
  * FlagGems：`/workspace/FlagGems`
  * vllm-plugin-FL 和 FlagGems 源码更新：`pip install --no-build-isolation`

### 4.2 提交成果要求

#### 4.2.1 提交成果

提交截止时间：**2026-11-20 23:59（UTC+8）**

提交内容：参赛者需要以 `参赛队伍名.zip` 提交作品，其中包含源代码（若在两款芯片都有性能优化，需将两个平台源代码合并提交）、技术报告和复现说明。

```
参赛队伍名
├── FlagGems
├── readme.md
├── report.pdf
└── vllm-plugin-FL
```

* **代码仓库**：
  * **vllm-plugin-FL**：包含所有源代码和编译脚本，代码规范，确保在指定环境中能够顺利运行。代码需具有良好的注释和清晰的结构，便于评审人员理解和复现。
  * 【可选】**FlagGems**：涉及到算子相关改动，提交到 FlagGems。
* **技术报告（`report.pdf`）**：完整阐述方案设计思路、算法原理、实现架构、优化策略、实验环境、测试结果与效果分析；图文结合，逻辑清晰，完整支撑作品方案，便于评审人员理解整体技术路线与创新点。
* **复现说明（`readme.md`）**：清晰列明软硬件环境、依赖版本、环境部署步骤、编译执行流程、数据集准备方式、运行命令与结果校验方法，步骤完整可落地，保障评审人员依据文档能够顺利复现实验结果。

#### 4.2.2 提交地址

* **FlagOS 平台**：<https://flagos.io/race-detail-season2?id=539vlt2p&lang=cn>
* **获奖者代码须命名为**：
  `【FlagOS开放计算全球挑战赛S2.赛题2】【xxx队伍】特性简要描述xxxx`
* 代码整理成 PR 并在评审结束后 **3 个工作日内**提交至 GitHub：
  <https://github.com/flagos-ai/vllm-plugin-FL/tree/flagos-2026-s2>
  （如未按时提交，取消获奖资格）
* **PR 样例**：<https://github.com/flagos-ai/vllm-plugin-FL/pull/463>

### 4.3 评审方案

#### 任务启动 —— 天数

```bash
export VLLM_PLUGINS=fl

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

#### 任务启动 —— 沐曦

```bash
export VLLM_PLUGINS=fl

vllm serve /workspace/MiniCPM5-2B \
  --port 9031 \
  --served-model-name minicpm \
  --gpu-memory-utilization 0.85 \
  --max-model-len 131072
```

> 注意差异：**天数带 `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`，沐曦不带。**
> 这条差异是我们做方案设计时必须尊重的硬约束——沐曦侧的 cudagraph 覆盖需要靠
> 插件内部或 config 层解决，不能靠 serve 参数。

#### 简单测试

```python
import requests
import json

response = requests.post(
    'http://127.0.0.1:9031/v1/chat/completions',
    headers={'Content-Type': 'application/json'},
    json={
        'model': 'minicpm',
        'messages': [{'role': 'user', 'content': 'introduce llm'}],
        'max_tokens': 256
    }
)
print(response.json())
```

#### 评测脚本

赛事方将提供基于 **evalscope** 工具（镜像内置）的评测脚本和详尽的评测依赖（镜像内置），确保评测环境一致性与结果的公正性。选手需提交评测结果，用于确保优化前后的正确性。

评测数据集指定：**math_500 的 level3 子集**
（<https://evalscope.readthedocs.io/zh-cn/v1.5.1/benchmarks/math_500.html>）

```bash
evalscope eval \
  --model minicpm \
  --api-url http://127.0.0.1:9031/v1/chat/completions \
  --api-key EMPTY \
  --eval-type openai_api \
  --datasets math_500 \
  --dataset-args '{"math_500": {"dataset_id": "/workspace/evalscope-datasets/math_500", "subset_list": ["Level 3"]}}' \
  --eval-batch-size 8 \
  --timeout 3600 \
  --generation-config '{"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768}' \
  --work-dir /workspace/evalscope-datasets/level3 \
  --ignore-errors
```

#### Benchmark 脚本

赛事方提供基于 vllm 的 benchmark 脚本，参赛队伍需要提交 benchmark 的运行结果，**排名以组委会统一复现结果为准**。

* 评测说明：性能指标是 `total tokens/s`。
* 测试场景 `[input_len, output_len, concurrency, num_prompts]`：
  * **4k 场景**：`[4096, 1024, 64, 256]`
  * **16k 场景**：`[16384, 1024, 64, 128]`
* benchmark 日志路径：`/workspace/benchmark_results/summary_xxxx.csv`

```bash
cd /workspace
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B \
  --served-model-name minicpm \
  --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

#### 基线

见 [`baselines.md`](baselines.md)（含截图原件）。

#### 4.3.1 技术评审说明

* **基础要求**：
  * **数据集评测**：基线 0.962，要求精度 **≥ 0.95**
  * **性能评测**：性能指标偏差 1% 以内视为正常波动；
    4k 和 16k 场景 `total tokens/s` 限制 `(baseline-1%~, )`，
    `ttft` 限制 `(, ~baseline+1%)`
* **赋分说明**：单一场景性能指标超过 1% 赋分，最终得分取 2 个场景（4k 和 16k）平均。

#### 4.3.2 总分计算

```
总分 = 性能分（天数卡） + 性能分（沐曦卡） + 创新分
```

1. **性能分：70% 性能提升比例。**
   例：baseline 200，提升到 210，提升了 5%，性能分 = `100 × 5% × 70%`
2. **创新分：30% 专家评分（技术报告创新性）。**
   创新分 = `专家评分 × 30%`

**说明**：

* **技术报告创新性评判维度**：系统级调度、算子编译优化创新等原创优化思路
* **评分要求**：排名过程中，若出现性能提升相同的情况会根据参赛队伍提供的技术报告评估技术创新性（由赛事方投票决定）
* **模型使用限制**：参赛过程中禁止使用 MiniCPM5-2B 以外的模型。如违反该要求，其参赛成绩将被认定为无效。
* **开发框架要求**：参赛团队必须使用 FlagOS 推理框架 vllm-plugin-FL 和高性能算子库 FlagGems 作为开发工具。经过组委会完成代码合规性审核后，符合要求的参赛团队方可获得最终排名。
* **方案复现与有效性认定**：在技术方案评审环节，组委会将对参赛团队提交的方案进行复现验证。对于无法成功复现的方案，视为成绩无效。
* **未及时提交 PR**：获奖选手未及时提交 PR，视为自动放弃获奖资格。

#### 4.3.3 奖项设置

| 奖项 | 奖金 |
|---|---|
| 一等奖 | 30,000 元 |
| 二等奖 | 20,000 元 |
| 三等奖 | 10,000 元 |

*\*本次大赛最终解释权由 FlagOS 开放计算全球大赛组委会所有*
