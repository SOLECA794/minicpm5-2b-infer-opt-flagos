# 官方 Baseline 与差距分析

来源：赛事平台「赛题」Tab 内嵌截图（本地副本
[`tianshu_baseline.png`](tianshu_baseline.png) / [`metax_baseline.png`](metax_baseline.png)）。
抓取时间 2026-10-19，赛道 `rid=539vlt2p`。

---

## 1. 天数智芯 BI-V150

| Prefill | Decode | Conc | Num Prompts | Duration (s) | Output tok/s | **Total tok/s** | Mean TTFT (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1024 | 64 | 256 | 646.31 | 405.6 | **2028.01** | 11,573.53 |
| 16384 | 1024 | 64 | 128 | 2434.82 | 53.83 | **915.15** | 599,262.03 |

## 2. 沐曦曦云 C500 (64 GB)

| Prefill | Decode | Conc | Num Prompts | Duration (s) | Output tok/s | **Total tok/s** | Mean TTFT (ms) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 1024 | 64 | 256 | 257.525 | 1017.93 | **5089.645** | 3,199.435 |
| 16384 | 1024 | 64 | 128 | 316.975 | 413.51 | **7029.675** | 27,197.135 |

---

## 3. 对比矩阵

| 场景 | 天数 tok/s | 沐曦 tok/s | 沐曦 / 天数 |
|---|---:|---:|---:|
| 4k  (4096/1024/c64/n256)  | 2028.01 | 5089.65 | **2.51×** |
| 16k (16384/1024/c64/n128) | 915.15 | 7029.68 | **7.68×** |

### TTFT

| 场景 | 天数 TTFT | 沐曦 TTFT | 倍数 |
|---|---:|---:|---:|
| 4k  | 11.57 s | 3.20 s | 3.6× |
| 16k | **599.26 s** | 27.20 s | **22.0×** |

---

## 4. 诊断：这份 baseline 里藏着什么

### 4.1 天数的 16k TTFT = 599 秒，这是**病理级**的

128 个 prompt、每个 16384 token、并发 64。理想情况下 prefill 应该被并发消化，
TTFT 量级应该在秒级到十几秒。**599 秒**意味着平均每个 prefill 请求耗掉
数百秒——只有一种解释：

> **天数的 prefill attention 没有走 flash-attention 类融合内核，
> 而是退化到了 O(S²) 的 materialized attention（SDPA math 路径 /
> 朴素 einsum + 完整 attention matrix 落显存）。**

校验：16k 场景的 `Duration = 2434.82 s`，128 个 prompt。
如果 attention 是 `S² × head_dim` 的显存带宽受限路径，16k 相对 4k 的
计算量是 16×，而实测 duration 只涨了 3.8×（646→2435），说明 4k 场景本身
也已经被某个**非 attention 的瓶颈**（很可能是 decode 阶段）主导。
但 16k 的 TTFT 涨了 51.8×（11.6s → 599s），远超线性的 4×，
这就是 **attention 二次复杂度暴露**的典型指纹。

**这同时是最大的机会**：如果 16k 的 prefill attention 能被替换成
paged/flash 类内核，16k 的 `total tok/s` 从 915 提到 3000+ 是完全现实的
（沐曦在同一个模型上是 7029）。**单这一项就能贡献巨大的性能分。**

### 4.2 沐曦的 16k 反而比 4k 更快（7029 vs 5089）

正常现象：16k 场景 `num_prompts=128`（4k 是 256），单请求 prefill 更长，
prefill 阶段是 compute-bound，GPU 利用率更高；decode 阶段占比更低。
说明**沐曦的 prefill 路径是健康的**（TTFT 27s 对 16k×64 并发是合理量级）。

### 4.3 天数的 4k 也被 decode 拖住了

4k 场景 256 prompts × 1024 output，`Duration 646 s`，`Output tok/s 405.6`。
总输出 token = 256 × 1024 = 262144，`262144 / 646 ≈ 406` ✓。
并发 64 下输出吞吐只有 406 tok/s，即 **每 token 每 batch ~6.3 tok/s**——
decode 阶段 kernel launch 开销 / 小 batch 低效非常明显。

---

## 5. 目标设定（用于算力申请材料）

保守、可辩护、且能对上评分公式：

| 场景 | Baseline | 目标 | 提升 | 依据 |
|---|---:|---:|---:|---|
| 天数 4k  | 2028 | **2900** | +43% | decode 融合 + attention 后端修复 + cudagraph 覆盖 |
| 天数 16k | 915  | **2600** | +184% | prefill attention 走融合内核（对标沐曦 7029 的 37%） |
| 沐曦 4k  | 5090 | **6200** | +22% | 算子融合 + KV cache 布局 + 调度 |
| 沐曦 16k | 7030 | **8300** | +18% | prefill chunk 调优 + 内存复用 |

> 数字写进申请材料时**要标为「目标」而非「承诺」**，并附推导依据
> （见 [`../research/`](../research/) 的源码证据）。评审看的是
> 「技术路线是否合理且具有可复现性」，不是数字够不够大。

---

## 6. 护栏（任何优化都必须过这两关）

```
精度：evalscope math_500 · Level 3 子集，acc ≥ 0.95（baseline 0.962）
时延：Mean TTFT ≤ baseline × 1.01
```

**TTFT 是硬约束**。任何「用吞吐换 TTFT」的手段（例如把 prefill 拆得更碎、
激进 chunked prefill、加大 max_num_batched_tokens 挤压 prefill 优先级）
都必须先算 TTFT 预算。
