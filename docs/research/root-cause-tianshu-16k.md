# 根因分析：天数 BI-V150 为什么比沐曦 C500 慢 2.5× ~ 7.7×

> 状态：**代码层证据 SOLID**；**性能归因 PLAUSIBLE**（需上卡 profiler 才能升为 SOLID）。
> 本文诚实区分这两者。

---

## 1. 现象

| 场景 | 天数 tok/s | 沐曦 tok/s | 倍数 | 天数 TTFT | 沐曦 TTFT | 倍数 |
|---|---:|---:|---:|---:|---:|---:|
| 4k  | 2028.01 | 5089.65 | **2.51×** | 11.57 s | 3.20 s | 3.6× |
| 16k | 915.15 | 7029.68 | **7.68×** | **599.26 s** | 27.20 s | **22.0×** |

**599 秒的首 token 延迟不是一个「慢」的问题，是一个「路径错误」的问题。**

---

## 2. 算术：这两组数字意味着什么

### 2.1 校验 baseline 自洽

16k 场景 `[16384, 1024, 64, 128]`：

```
总输入 token = 128 × 16384 = 2,097,152
总输出 token = 128 ×  1024 =   131,072
总 token     = 2,228,224

天数: 2,228,224 / 2434.82 s =   915.15 tok/s  ✓ 与官方 915.15 完全吻合
沐曦: 2,228,224 /  316.975s =  7029.68 tok/s  ✓ 与官方 7029.675 完全吻合
```

4k 场景 `[4096, 1024, 64, 256]`：

```
总 token = 256 × (4096 + 1024) = 1,310,720
天数: 1,310,720 / 646.31 s = 2028.01 tok/s  ✓
沐曦: 1,310,720 / 257.525s = 5089.65 tok/s  ✓
```

两卡的数字都自洽，说明是真实测量，不是脏数据。

### 2.2 decode 吞吐

用 `总输出 token / 总时长` 反推（这正是官方 `Output tok/s` 列）：

```
天数 16k: 131,072 / 2434.82 = 53.83 tok/s   ← 官方列值，完全吻合
沐曦 16k: 131,072 /  316.975 = 413.51 tok/s  ← 完全吻合
```

**decode 慢 7.7×。**

### 2.3 prefill 吞吐（TTFT 反推）

并发 64，128 个 prompt = 2 个 wave。每个 wave 需要 prefill
`64 × 16384 = 1,048,576` token。

```
天数: 1,048,576 × 2 waves / 599.26 s ≈  3,500 tok/s 聚合 prefill
沐曦: 1,048,576 × 2 waves /  27.20 s ≈ 38,550 tok/s 聚合 prefill
```

> 这是**量级估算**（vLLM continuous batching 会交叠 prefill 与 decode，
> 实际 wave 结构更复杂）。但即使把 2 waves 换成 1 wave，结论不变：
> **天数 prefill 慢一个数量级。**

### 2.4 有效算力

16k 场景的计算量（MiniCPM5-2B，2.52B 参数）：

```
线性层 prefill : 2 × 2.52e9 × 2,097,152        ≈ 10.6 PFLOP
attention 二次项: 4 × 42层 × 2048 × S²  × 128请求 ≈ 11.8 PFLOP
                                        合计      ≈ 22.4 PFLOP

天数: 22.4 PFLOP / 2434.82 s ≈  9.2 TFLOP/s
沐曦: 22.4 PFLOP /  316.975s ≈ 70.7 TFLOP/s
```

**天数跑出了 9 TFLOP/s 的有效算力。** 任何一款 2026 年的 AI 加速器，
在 2.5B 模型的 prefill 上都不该低于 50 TFLOP/s。

> ⚠️ 这个 TFLOP/s 是**有效值**，含 decode（低算术强度）稀释。
> 它不应该被当作「硬件峰值达成率」引用。它的意义是**量级诊断**。

---

## 3. 代码层证据链（SOLID）

### 3.1 天数的 attention 后端 = `TRITON_ATTN`

`vllm_fl/dispatch/config/iluvatar.yaml:30-35`
```yaml
op_backends:
  # All operators: prioritize FlagGems (Triton), fallback to vendor, then reference
  attention_backend:
    - flagos
    - vendor:iluvatar
    - reference
```
→ 第一顺位是 `flagos`。

`vllm_fl/dispatch/backends/flaggems/flaggems.py:179-187`
```python
use_flaggems_attn = os.environ.get(
    "VLLM_FL_USE_FLAGGEMS_ATTN", "0"
).lower() in ("1", "true", "yes")

if use_flaggems_attn:
    print("Using FlagGems attention backend.")
    return "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"

return AttentionBackendEnum.TRITON_ATTN.get_path()
```
→ **默认 `"0"`，直接返回 `TRITON_ATTN`。** 候选顺序里后面的
`vendor:iluvatar`、`reference` 根本没机会被尝试（没有抛异常）。

即便走到了 `vendor:iluvatar`：

`vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py:480-486`
```python
# flash_attn is not available on iluvatar. Use TRITON_ATTN (the vllm
# default). The tl.make_tensor_descriptor stub and perf model patches
# are already applied at module level (above), so triton JIT's
# DependencyFinder can hash the kernel without AttributeError on
# triton < 3.3.  The kernel itself uses USE_TD=False at runtime, so
# the stub is never called.
return AttentionBackendEnum.TRITON_ATTN.get_path()
```
→ **原文注释直接承认 `flash_attn is not available on iluvatar`。**

而且 `vendor/iluvatar/` 目录下**根本没有 attention 实现文件**：
```
vendor/iluvatar/impl/activation.py
vendor/iluvatar/impl/normalization.py
vendor/iluvatar/impl/rotary.py
```

### 3.2 沐曦的 attention 后端 = `FLASH_ATTN`（MACA 原生）

`vllm_fl/dispatch/config/metax.yaml:28-31`
```yaml
op_backends:
  # attention_backend: prioritize vendor (MACA attention)
  attention_backend:
    - vendor:metax
```
→ **唯一来源，不允许回退。**

`vllm_fl/dispatch/backends/vendor/metax/metax.py:21-29`
```python
def register_attention_backends():
    register_backend(
        AttentionBackendEnum.FLASHMLA,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.mla.flashmla.MacaFlashMLABackend",
    )
    register_backend(
        AttentionBackendEnum.FLASH_ATTN,
        class_path="vllm_fl.dispatch.backends.vendor.metax.impl.attention.flash_attn.MacaFlashAttentionBackend",
    )
```

`metax.py:135-151`
```python
# Default to FLASH_ATTN
return AttentionBackendEnum.FLASH_ATTN.get_path()
```

沐曦有**真正的 FA 实现**（1085 + 906 + 637 行）：
```
vendor/metax/impl/attention/flash_attn.py
vendor/metax/impl/attention/ops/triton_unified_attention.py
vendor/metax/impl/attention/ops/triton_decode_attention.py
```

### 3.3 决定性对照

| | 天数 BI-V150 | 沐曦 C500 |
|---|---|---|
| 配置第一顺位 | `flagos` | `vendor:metax` |
| 最终解析到的后端 | **`TRITON_ATTN`** | **`FLASH_ATTN`** |
| 后端实现位置 | vLLM 自带通用 Triton | MACA 原生，与硬件协同调优 |
| 有无 TMA（`tl.make_tensor_descriptor`） | ❌ triton < 3.3，只有 stub | ✅ |
| triton 版本 | 3.2.x（源码注释） | 未在源码标注（UNVERIFIED） |

### 3.4 放大器：天数的其他降级

| 降级 | 位置 | 影响 |
|---|---|---|
| 无 TMA 描述符 | `iluvatar.py:22-64` | prefill 的 K/V 访存无硬件加速 |
| 链式 `or` 靠正则改写源码 | `iluvatar.py:67-156` | 只覆盖 1 个文件，脆弱 |
| sampler 的 `torch.compile` 被解包 | `iluvatar.py:210-235` | decode 每 token 多一次 Python 调用 |
| triton autotune 用常量 1500 MHz 估算 | `iluvatar.py:159-203` | autotune 选择**可能是次优的** |
| 无 NVML | `platform.py:412-417` | 只能给合成 UUID |

**这些全部叠加在 `TRITON_ATTN` 这条本就慢的路径上。**

---

## 4. 为什么这是「可修」的（关键）

**FlagGems v5.3.5 里已经有一套完整的 FA2 实现**：

```
src/flag_gems/ops/attention.py
  L35   @triton.jit  def _attn_fwd_inner(...)
  L176  @triton.jit  def _attn_fwd(...)
  L1284 def flash_attn_varlen_func(...)      ← FA2 前向，支持 GQA + paged KV
src/flag_gems/fused/reshape_and_cache_flash.py
  L77   def reshape_and_cache_flash(...)     ← paged KV 写入
```

**vllm-plugin-FL 里已经有把它接进 vLLM 的 wiring**：

`vllm_fl/dispatch/backends/flaggems/impl/attention.py`（857 行，改编自 vLLM `flash_attn.py`）
```
L43   from flag_gems import flash_attn_varlen_func, reshape_and_cache_flash
L460  self.vllm_flash_attn_version = 2  # FlagGems only supports FA2
L488  reshape_and_cache_flash(...)
L593  flash_attn_varlen_func(...)
```

**它只是被默认关掉了**（`VLLM_FL_USE_FLAGGEMS_ATTN` 默认 `"0"`）。

> 这不是「要从零写一个 FlashAttention」的命题，
> 而是「**把已经写好的两个东西接通，并为 BI-V150 把 kernel 调好**」的命题。

---

## 5. 最小验证实验（拿到卡后第一步做）

```bash
# 环境（赛事镜像内）
export VLLM_PLUGINS=fl

# --- A: 复现 baseline，确认 TRITON_ATTN ---
vllm serve /workspace/MiniCPM5-2B --port 9031 \
  --compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}' \
  --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072
# 启动日志里应出现：
#   "Using attention backend via dispatch (use_mla=False, use_sparse=False): <path>"
# 记录这个 path —— 它必须等于 TRITON_ATTN 的路径，否则假设不成立

# --- B: 切到 FlagGems FA2 ---
export VLLM_FL_USE_FLAGGEMS_ATTN=1
# 同上启动，日志里应出现 "Using FlagGems attention backend."
# 并确认 backend path 变成
#   vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend

# --- C: 同样的 benchmark ---
python3 /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
  --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
  --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]'
```

### 判定标准

| 结果 | 结论 | 下一步 |
|---|---|---|
| B 的 16k tok/s 显著高于 A（≥ +30%） | 假设成立，主线确定 | 进入 kernel 调优（autotune 重扫） |
| B 与 A 持平 | FlagGems FA2 在天数上也没优势 | 转 kernel 层：block/warp/stage 重扫 |
| B 正确性不过 | FlagGems FA2 在天数上有数值或 bug 问题 | 定位并修，这本身就是加分项 |
| B 启动失败 | wiring 有缺口 | 修 wiring，也是加分项 |

> **无论哪个分支，都有可写进技术报告的产出。**

---

## 6. 诚实边界

* ✅ **代码事实**（哪条路径、哪个默认值）—— 逐行读过，SOLID
* ⚠️ **性能归因**（599s TTFT 确由 `TRITON_ATTN` 造成）—— 强推断，**未经上卡 profiler 证实**
* ❓ 未排除的其他可能：驱动层、编译器层、MACA vs Corex 的访存子系统差异、
  天数卡本身的 HBM 带宽劣势

**必须在上卡第一天用 `torch.profiler` 或厂商 profiler 分离 prefill/decode 的
kernel 时间占比，才能把这条归因从 PLAUSIBLE 升级到 SOLID。**
在此之前，技术报告里应写「首要假设」，不写「已确认结论」。