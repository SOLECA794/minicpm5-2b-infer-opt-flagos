# 初步技术方案

| 项目 | 内容 |
|---|---|
| 赛题 | FlagOS 开放计算全球大赛 · 赛季二 · 赛道二（MiniCPM 模型推理吞吐性能优化） |
| 目标硬件 | 天数智芯 BI-V150、沐曦曦云 C500（64 GB） |
| 目标模型 | MiniCPM5-2B |
| 基准框架 | vllm-plugin-FL `flagos-2026-s2`、FlagGems `v5.3.5` |

## 摘要

本方案针对赛道二在固定模型、固定框架、固定启动参数条件下提升 MiniCPM5-2B 推理吞吐的要求，提出以执行路径归因为起点、以算子实现改写为手段的优化路线。

对官方基线数据的核算表明，其数值在内部完全自洽，可确认为真实测量结果。据此反推的有效算力为：天数 BI-V150 在 16k 场景下约 9.2 TFLOP/s，沐曦曦云 C500 约 70.7 TFLOP/s，前者为后者的 13%。该量级的差距超出两款芯片的硬件规格差异所能解释的范围，其成因指向执行路径的选择，而非参数调优的充分程度。

对 vllm-plugin-FL 与 FlagGems 的源码核查确认了这一判断。两款芯片的 attention 后端分属不同实现：沐曦使用 MACA 原生 FA2，天数使用 vLLM 通用 TRITON_ATTN。FlagGems v5.3.5 已包含完整的 FA2 实现，vllm-plugin-FL 亦已包含将其接入 vLLM 的完整代码，但该接入处于默认关闭状态。

据此，本方案将工作划分为四个阶段：P0 恢复天数侧的正确 attention 路径；P1 基于实测数据重新收敛两款芯片的 autotune 参数；P2 补齐缺失的融合算子；P3 修复框架层的已知降级项。其中 P0 为结构性收益的主要来源，P1 与 P2 同时构成技术报告创新性的依据。

资源申请为天数 BI-V150 17 卡日、沐曦曦云 C500 6 卡日，合计 23 卡日，已含约 30% 冗余。

## 1. 赛题约束与优化空间

### 1.1 固定条件

赛题对实验条件作出完整限定，以下维度均不可变更：

| 维度 | 约束 |
|---|---|
| 模型 | 仅限 MiniCPM5-2B |
| 启动参数 | 必须与基线一致，不得增删 |
| 量化与投机采样 | 禁止启用 |
| benchmark 脚本 | 禁止修改或调优 |
| 上游代码 | 禁止直接合并最新分支 |
| 算子选择逻辑 | 禁止在无对应算子优化的前提下删除 |

### 1.2 可改动范围

在上述约束下，可改动的对象为框架与算子库的内部实现：

| 对象 | 说明 |
|---|---|
| vllm-plugin-FL 内部实现 | 分发策略、后端选择、平台适配层、worker 与调度路径 |
| FlagGems 内部实现 | 算子实现、融合算子、autotune 搜索空间与选择策略 |

因此本赛题的工作性质是在固定命令行语义下改写框架与算子库的代码，优化成果须落在代码中，不能落在启动命令中。

### 1.3 评分结构对工作重心的约束

```
总分 = 性能分(天数) + 性能分(沐曦) + 创新分
性能分(厂商) = mean( 100 × 提升比例(4k) × 0.70, 100 × 提升比例(16k) × 0.70 )
创新分       = 专家评分 × 0.30
```

该结构产生三条直接约束：

1. 两款芯片的性能分为相加关系，任一侧缺失即损失约一半性能分，两侧均须产出有效结果。
2. 性能分系数为 0.70，而创新分为独立的 0.30 权重板块。同等工程投入下，若成果具备可量化的机理解释，其在创新分上的回报不低于性能分。方案因此要求每一项优化同时满足「可测量」与「可解释」。
3. 单场景提升低于 1% 不予计分。优化须作用于结构性瓶颈，边际改进在本赛制下不产生收益。

### 1.4 硬性护栏

```
精度：evalscope math_500 Level 3   acc ≥ 0.95   （基线 0.962）
时延：Mean TTFT ≤ 基线 × 1.01
```

TTFT 的 +1% 上限构成对「以时延换吞吐」类手段的硬约束。任何吞吐优化在实施前须先核算 TTFT 预算。16k 场景的基线 TTFT 本身处于极低水平，其可改善空间与 TTFT 余量均为两个场景中最大者，故列为 P0 的主攻场景。

## 2. 现状分析

### 2.1 官方基线数据

| 场景 | 天数 tok/s | 沐曦 tok/s | 倍数 | 天数 TTFT | 沐曦 TTFT | 倍数 |
|---|---:|---:|---:|---:|---:|---:|
| 4k `[4096,1024,64,256]` | 2028.01 | 5089.65 | 2.51× | 11.57 s | 3.20 s | 3.6× |
| 16k `[16384,1024,64,128]` | 915.15 | 7029.68 | 7.68× | 599.26 s | 27.20 s | 22.0× |

### 2.2 基线数据的内部一致性验证

以「总 token 数 ÷ 总时长」对两组数据做交叉核算，结果与官方公布值完全一致：

```
16k 场景：2,228,224 tokens
  天数 2,228,224 / 2434.82 s =   915.15 tok/s
  沐曦 2,228,224 /  316.975 s = 7029.68 tok/s
4k  场景：1,310,720 tokens
  天数 1,310,720 /  646.31 s = 2028.01 tok/s
  沐曦 1,310,720 /  257.525 s = 5089.65 tok/s
```

四项核算误差均为零，可确认基线数据来自真实测量而非估算。

### 2.3 有效算力反推

以 16k 场景的总计算量（线性层约 10.6 PFLOP、attention 二次项约 11.8 PFLOP，合计约 22.4 PFLOP）除以实测时长，得到两款芯片在该场景下的有效算力：

| 平台 | 有效算力 |
|---|---|
| 天数 BI-V150 | 约 9.2 TFLOP/s |
| 沐曦曦云 C500 | 约 70.7 TFLOP/s |

对 2.5B 量级模型在 prefill 阶段而言，9.2 TFLOP/s 显著低于该类硬件的合理区间。该结果不构成调优余量的度量，而指示执行路径本身存在偏差。

### 2.4 差异归因

对 vllm-plugin-FL 的分发配置与后端注册逻辑进行核查，两款芯片的 attention 后端归属如下：

| 环节 | 天数 BI-V150 | 沐曦曦云 C500 |
|---|---|---|
| 分发配置 | `iluvatar.yaml`：`[flagos, vendor:iluvatar, reference]` | `metax.yaml`：`[vendor:metax]` |
| `flagos` 分支返回 | `TRITON_ATTN`（`VLLM_FL_USE_FLAGGEMS_ATTN` 默认 `"0"`） | 未参与 |
| `vendor:*` 分支返回 | `TRITON_ATTN` | `FLASH_ATTN`（MACA 原生） |
| 实际后端 | vLLM 通用 `TRITON_ATTN` | MACA 原生 FA2 |

结论：沐曦侧的 attention 由厂商原生 FlashAttention 实现承担，天数侧由 vLLM 的通用 Triton 实现承担。二者在 16k 长序列 prefill 上的效率差异，与 2.1 节观测到的 7.68 倍吞吐差距及 22.0 倍 TTFT 差距在量级上相符。

关键代码依据：

```python
# vllm_fl/dispatch/backends/flaggems/flaggems.py:179-187
use_flaggems_attn = os.environ.get(
    "VLLM_FL_USE_FLAGGEMS_ATTN", "0"
).lower() in ("1", "true", "yes")
if use_flaggems_attn:
    return "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"
return AttentionBackendEnum.TRITON_ATTN.get_path()
```

```python
# vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py:480-486
# flash_attn is not available on iluvatar. Use TRITON_ATTN (the vllm default).
return AttentionBackendEnum.TRITON_ATTN.get_path()
```

### 2.5 天数路径上叠加的降级项

除 attention 后端之外，天数的适配层另含五处降级实现，均构成对性能或可诊断性的约束：

| 降级项 | 位置 | 影响 |
|---|---|---|
| Triton 缺少 TMA 描述符支持，以空实现替代 | `iluvatar.py:22-64` | prefill 阶段 K/V 访存无硬件加速 |
| Triton 不支持链式 `or`，以正则改写 vLLM 源码 | `iluvatar.py:67-156` | 补丁仅覆盖单一文件，维护性差 |
| sampler 的 `torch.compile` 被强制解包 | `iluvatar.py:210-235` | decode 阶段每个 token 增加一次解释器调用 |
| triton autotune 的性能模型以常量 1500 MHz 估算主频 | `iluvatar.py:159-203` | autotune 的配置选择可能偏离最优 |
| 缺少 NVML，设备 UUID 为合成值 | `platform.py:412-417` | 监控与瓶颈定位能力缺失 |

### 2.6 修复所需组件的完备性

本方案的 P0 阶段不需要从零实现 FlashAttention，所需组件已全部具备。

FlagGems v5.3.5 包含完整的 FA2 前向实现与 paged KV 写入：

```
src/flag_gems/ops/attention.py
  L35   @triton.jit  _attn_fwd_inner
  L176  @triton.jit  _attn_fwd
  L1284 flash_attn_varlen_func        支持 GQA 与 block_table（paged KV）
src/flag_gems/fused/reshape_and_cache_flash.py
  L77   reshape_and_cache_flash       paged KV 写入
```

vllm-plugin-FL 包含将该实现接入 vLLM 的完整代码（857 行，由 vLLM 的 `flash_attn.py` 改编）：

```
vllm_fl/dispatch/backends/flaggems/impl/attention.py
  L43   from flag_gems import flash_attn_varlen_func, reshape_and_cache_flash
  L460  self.vllm_flash_attn_version = 2      仅支持 FA2
  L488  reshape_and_cache_flash(...)
  L593  flash_attn_varlen_func(...)
```

该接入当前处于默认关闭状态，即 2.4 节所示的 `VLLM_FL_USE_FLAGGEMS_ATTN="0"`。

MiniCPM5-2B 的结构与该实现相匹配：`num_attention_heads = 16`、`num_key_value_heads = 2`，构成 8:1 的 GQA 分组，满足 `num_heads % num_kv_heads == 0`；`head_dim = 128`，属 FA2 的典型最优取值。

## 3. 技术方案

### 3.1 总体路线

```
P0  恢复天数侧 attention 执行路径      结构性收益
    天数：TRITON_ATTN → FlagGems FA2
                    ↓
P1  按实测数据重新收敛 autotune        算子库收益
    BI-V150 / C500 分别标定
                    ↓
P2  补齐缺失的融合算子                 创新性收益
    fused_add_rms_norm / RoPE+KV 写入
                    ↓
P3  修复框架层降级项                   稳健增量
    sampler compile / block_size
```

### 3.2 P0：恢复天数侧的正确 attention 路径

**工作内容**

1. 在代码层将天数侧的 attention 后端默认值修正为 `AttentionFLBackend`。评测方以 `VLLM_PLUGINS=fl` 固定启动，不通过环境变量施加改动，故须修改框架内部分发逻辑的默认行为。
2. 在 BI-V150 上验证 `AttentionFLBackend` 的数值正确性，包括与基线输出的逐 token 对齐及 math_500 精度回归。
3. 若 FlagGems FA2 在天数上不可用，转入备选方案：以 `AttentionFLBackend` 为骨架，为 BI-V150 实现独立的 attention 后端，替换其 kernel 调度策略。

**收益估计**

16k 场景的 TTFT 当前为 599.26 s。若该场景恢复至与沐曦同量级，`total tok/s` 将由 915 提升至 3000 量级以上。按评分公式，单场景提升 200% 对应 140 分。该收益量级高于其余各阶段之和。

**风险与控制**

| 风险 | 控制措施 |
|---|---|
| FlagGems FA2 在天数上存在数值偏差 | 逐层对拍定位至具体 kernel；修正偏差本身即为可写入报告的成果 |
| 天数 Triton 3.2.x 不支持部分语言特性 | 已有 `patch_triton_chained_or_for_iluvatar` 先例，可依同一机制扩展 |
| 切换后启动失败 | 保留 `TRITON_ATTN` 作为配置级回退路径，不影响框架可用性 |

### 3.3 P1：基于实测数据重新收敛 autotune

FlagGems 已为各厂商提供 attention 的 autotune 搜索空间：

```
_iluvatar/tune_configs.yaml:15-40
  BLOCK_M [64, 128] × BLOCK_N [32, 64, 128] × PRE_LOAD_V [true, false]
  × num_warps [4, 8] × num_stages [1, 2, 3]        共 72 组

_metax/tune_configs.yaml:15-38
  同上，num_stages 仅 [1]                          共 24 组
```

该搜索空间仅在 FlagGems 的 attention 被实际调用时生效。当前天数侧的 attention 由 `TRITON_ATTN` 承担，上述 72 组配置从未被使用。

**工作内容**

1. P0 生效后，以真实 benchmark 场景（4k 与 16k）驱动 autotune，替代合成 micro-benchmark。
2. 扩展搜索空间，纳入 `BLOCK_K`、`num_ctas` 等维度，对 BI-V150 与 C500 分别收敛。
3. 天数侧 Triton 的性能模型以常量 1500 MHz 估算主频（`iluvatar.py:159-203`），autotune 的配置选择因此建立在失真的硬件参数之上。方案改为以实测 profile 驱动选择，绕过该性能模型。

### 3.4 P2：补齐缺失的融合算子

Llama 类结构在每一层包含两处 residual 与 RMSNorm 的相邻运算：

```
h   = x + attn(rms_norm(x))
out = h + mlp(rms_norm(h))
```

FlagGems v5.3.5 未提供 `fused_add_rms_norm`（`_iluvatar/ops/` 与 `_metax/ops/` 下均无 `rms_norm.py`）。

**工作内容**

1. 实现 `fused_add_rms_norm` Triton kernel，使 residual add 与 RMSNorm 在一次显存往返内完成。42 层各两处，合计 84 次 kernel 调用合并为 42 次，同时消除 42 次 hidden state（2048 × bf16 × batch）的显存读写往返。
2. 实现 RoPE 与 paged KV cache 写入的融合 kernel。
3. 在天数侧已有的 `fused/matmul_bias_activation.py` 基础上，将 qkv 三路 matmul 接入 dispatch 层。

**收益依据**

三项均为具有明确硬件理由的 kernel 设计，而非参数调整：`fused_add_rms_norm` 降低的是 HBM 带宽占用，对带宽受限的 decode 阶段直接有效；RoPE 与 KV 写入的融合降低的是 kernel 启动开销，对 decode 阶段的每 token 固定开销有效。两者均不依赖特定厂商指令集，可在 FlagGems 通用层实现，符合赛题对泛化性优化策略的优先鼓励方向。

### 3.5 P3：框架层降级项修复

| 项目 | 位置 | 做法 | 预期效果 |
|---|---|---|---|
| sampler `torch.compile` 恢复 | `iluvatar.py:210-235` | 以 flagtree 的 inductor 后端重新编译，替代解包 | decode 每 token 开销下降 |
| KV cache `block_size` | `platform.py:183-193` | 当前固定为 16，实验 32 与 64 | block table 索引开销下降 |
| 沐曦 `disable_cascade_attn` | `platform.py:254-256` | 评估该限制是否过度保守 | 沐曦 16k 场景 |
| `opaque_attention_op` | `platform.py:381-384` | 评估 attention 进入计算图的可能性 | decode 全图化（高风险，置于最后） |

## 4. 实施计划

### 4.1 里程碑

| 阶段 | 时间 | 内容 | 验收判据 |
|---|---|---|---|
| M0 环境复现 | 算力到位后 3 日内 | 两款芯片各复现一次官方基线 | 实测 tok/s 与 TTFT 相对偏差均 ≤ 1% |
| M1 归因确认 | M0 后 2 日 | 以 profiler 分离 prefill 与 decode 的 kernel 占比 | 确认或修正 2.4 节的归因结论 |
| M2 P0 打通 | M1 后 5 日 | FlagGems FA2 在两款芯片生效 | 16k tok/s 提升 ≥ 30%，精度 ≥ 0.95 |
| M3 P1 调优 | M2 后 5 日 | 两款芯片 autotune 收敛 | 在 M2 基础上再提升 10% |
| M4 P2 融合 | M3 后 5 日 | 三项融合算子 | 单场景再提升 5%~15% |
| M5 P3 收尾 | M4 后 3 日 | 框架降级项 | 视剩余时间取舍 |
| M6 报告与提交 | 11-18 至 11-20 | report.pdf、readme.md、代码包 | 可复现，报告与代码逐条对应 |

### 4.2 实验规程

* 每次改动须同时产出四项数据：4k tok/s、16k tok/s、TTFT、math_500 精度。缺项不计为完成。
* 精度回归随每轮优化执行，不集中至末期。
* 每项优化独立为 patch，可单独启停，以保证报告中每一条策略均对应一个可复现的 commit。
* 每轮实验前重跑基线，以排除机器状态漂移。

### 4.3 算力需求

| 用途 | 天数 BI-V150 | 沐曦曦云 C500 |
|---|---|---|
| 基线复现 | 0.5 卡日 | 0.5 卡日 |
| 归因 profiling | 1 卡日 | 0.5 卡日 |
| P0 开发与验证 | 3 卡日 | 1 卡日 |
| P1 autotune（搜索与验证） | 3 卡日 | 1 卡日 |
| P2 融合 kernel | 3 卡日 | 1.5 卡日 |
| P3 与最终回归 | 2 卡日 | 1.5 卡日 |
| **合计（含 30% 冗余）** | **17 卡日** | **6 卡日** |

天数侧占比更高，原因在于其性能损失更大、需要更多的 profiling 与 autotune 迭代。

资源使用遵循以下原则：不做无假设的暴力搜索，每项实验均设定明确的假设与证伪判据；非实验时段及时释放资源；沐曦侧按平台建议每 36 小时执行一次开关机，以避免数据丢失。

## 5. 创新性说明

赛题对技术报告创新性的评判维度为系统级调度与算子编译优化创新。本方案在此框架下提出三项可量化、可解释的改进。

### 5.1 以实测 profile 替代常量估算的 autotune 选择

天数侧 Triton 的 `get_clock_rate_in_khz` 因缺少 `ixsmi` 与 NVML 支持而被替换为常量 1500 MHz（`iluvatar.py:159-203`），导致 autotune 的性能模型基于失真的硬件参数进行配置选择。

本方案在 autotune 完成后，以真实场景（4k 与 16k 的 prefill、decode）回测全部候选配置，建立「场景—最优配置」映射表，绕过性能模型。该改进不限于天数，对任何缺少硬件查询接口的平台均适用。

### 5.2 面向 decode 阶段的带宽与延迟分解

decode 阶段每生成一个 token 需读取全部 2.52B 参数，bf16 下约 5 GB。按实测 53.83 tok/s 计算，等效带宽约 270 GB/s，显著低于该代加速器的 HBM 带宽水平。据此判断 decode 阶段的主要限制因素为每步的 kernel 启动与同步开销，而非显存带宽。

本方案据此将 decode 路径上的小粒度 kernel（RMSNorm、residual add、RoPE、KV 写入、sampling）合并，降低每 token 的启动次数。该方向由上述带宽核算直接导出，具备量化依据。

### 5.3 跨厂商泛化的 kernel 抽象

`fused_add_rms_norm` 与 RoPE+KV 写入融合均不依赖特定厂商指令集，仅依赖 Triton 的通用能力。方案在 FlagGems 通用层实现，通过既有的 per-vendor `tune_configs.yaml` 完成参数特化，形成一套代码服务两款芯片、并可移植至 FlagGems 所支持全部 16 个厂商后端的实现。

## 6. 可复现性保障

赛题规定无法成功复现的方案视为成绩无效。本方案自起始阶段即将可复现性作为交付要件：

1. 全部实验脚本入库，包括 benchmark 调用、精度回归与结果解析。
2. `readme.md` 按「评审人员依文档可独立跑通」的标准编写，涵盖软硬件环境、依赖版本、编译步骤、数据准备、运行命令与结果校验方法。
3. 提供 `scripts/score.py`，内置官方评分公式与护栏校验，输入两份 CSV 即输出得分与合规判定，评审可独立验算。
4. 每项优化对应独立 patch，报告中每条策略标注对应的 commit 或 patch 名，保证报告与代码一致。
5. 保留每次实验的原始 CSV（`benchmark_results/summary_*.csv`），不只保留最优结果。
6. 提供 `scripts/score.py --verify-baseline`，用于在实验开始前校验本机是否复现官方基线，防止以失真的分母计算提升比例。

## 7. 风险与控制措施

| 风险 | 概率 | 影响 | 控制措施 |
|---|---|---|---|
| 天数 prefill 瓶颈位于驱动或编译器层，框架层无法改动 | 中 | 高 | P0 优先验证；若归因被推翻，转向 P1 与 P2 的框架层收益 |
| FlagGems FA2 在天数上存在数值问题 | 中 | 中 | 逐层对拍定位并修正；修正本身计入创新性成果 |
| 沐曦侧不提供 cudagraph 配置，图覆盖手段受限 | 高 | 中 | 在插件配置层注入等价配置，或论证 cudagraph 在该场景非必要 |
| 精度低于 0.95 | 低 | 致命 | 每轮回归，不集中至末期 |
| TTFT 超出基线 1% | 中 | 致命 | 每项吞吐优化实施前先核算 TTFT 预算 |
| 算力到位晚于预期 | 中 | 高 | 无卡期间完成全部静态分析与脚本；同步准备自备算力预案 |

## 8. 前期工作基础

在申请算力的同时，已完成以下不依赖 GPU 的工作：

| 产出 | 内容 |
|---|---|
| `docs/competition/` | 赛事规则、评分公式、基线与赛程的本地化镜像，含平台 API 原始 JSON，抓取过程可复现 |
| `docs/research/plugin-surface-map.md` | vllm-plugin-FL 逐文件测绘：入口、环境变量、分发层、attention 路径、厂商适配 workaround，均附 `file:line` |
| `docs/research/flaggems-op-map.md` | FlagGems v5.3.5 算子覆盖矩阵、MiniCPM5-2B 热路径映射、融合机会与 autotune 差异 |
| `docs/research/root-cause-tianshu-16k.md` | 16k 场景 599 秒 TTFT 的归因分析与最小验证实验设计 |
| `scripts/fetch_race.py` | 赛事元数据抓取与变更检测，官方规则变动可即时发现 |
| `scripts/score.py` | 官方评分公式实现、精度与 TTFT 护栏校验、基线复现门禁，附单元自检 |

上述工作的直接效果是：算力到位当日即可进入 M1 阶段，无需从零熟悉框架。

此外，已具备沐曦曦云 C500 的实验环境（MACA 3.7.1.5、驱动 3.8.30、16 GB 显存切片），可用于框架安装验证、算子功能验证与 kernel 级微基准测试。该环境不足以承载 16k 场景的基线复现（该场景需要完整 64 GB 显存），故 4.3 节仍申请完整规格的 C500 卡日，用于测量与调优阶段。
