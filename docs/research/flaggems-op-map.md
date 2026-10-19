# FlagGems 算子覆盖与机会测绘

> 源码：`upstream/FlagGems` @ tag `v5.3.5`
> 方法：逐文件静态阅读，每条结论附 `file:line` 与原文摘录。

---

## 1. 架构

### 1.1 厂商后端矩阵

`src/flag_gems/runtime/backend/` 下共 16 个厂商目录：

```
_aipu  _amd  _arm  _ascend  _cambricon  _enflame  _hygon  _iluvatar
_kunlunxin  _metax  _mthreads  _nvidia  _spacemit  _sunrise  _thead  _tsingmicro
```

**天数和沐曦都在支持列表里**（`_iluvatar`、`_metax`），不是「靠 cuda 兼容硬跑」。

### 1.2 天数后端声明

`src/flag_gems/runtime/backend/_iluvatar/__init__.py:18-36`
```python
def get_triton_extra_name():
    try:
        import triton
        from packaging import version

        if version.parse(triton.__version__) >= version.parse("3.6.0"):
            return "corex"
        return "cuda"
    except Exception:
        return "cuda"


vendor_info = VendorDescriptor(
    vendor_name="iluvatar",
    device_name="cuda",
    device_query_cmd="ixsmi",
    triton_extra_name=get_triton_extra_name(),
    fp64_enabled=False,
)
```

* 设备查询走 **`ixsmi`**（不是 nvidia-smi）
* `fp64_enabled=False` —— **没有 FP64**。任何依赖 FP64 的数值路径会静默降精度或失败
* triton 后端名随版本切换 `cuda` / `corex`

### 1.3 算子分发与回退

FlagGems 通过 `runtime/op_registrar.py` 注册，`runtime/configs_loader.py` 读配置，
`runtime/common.py` 处理厂商差异。未覆盖的算子直接落到 PyTorch/ATen。

---

## 2. 天数 vs 沐曦 的算子覆盖差异（关键）

### 2.1 天数的 ops 目录（`_iluvatar/ops/`）

共 **54 个算子文件**，覆盖：
`addmm`, `mm`, `linear`, `matmul_bf16`, `matmul_int8`, `div`, `avg_pool3d`,
`conv_depthwise2d`, `cholesky_solve`, `linalg_*`, `special_*`, `scatter_add`,
`repeat`, `tile`, `permute_copy`, `median`, `histc`, `hadamard_transform` …

**缺失（对 LLM 推理致命的）**：
* ❌ 无 `attention.py`（无 FlashAttention）
* ❌ 无 `rms_norm`（无 RMSNorm）
* ❌ 无 `rotary_embedding`（无 RoPE）
* ❌ 无 `softmax.py`
* ❌ 无 `silu_and_mul` / 激活融合
* ❌ 无 `layernorm.py`

### 2.2 沐曦的 ops 目录（`_metax/ops/`）

共 **88 个算子文件**，比天数多出：
`bmm`, `batch_norm`, `groupnorm`, `layernorm`, `log_softmax`, `sigmoid`, `tanh`,
`masked_fill`, `index`, `index_put`, `index_select`, `softmax` 相关,
`outer`, `nonzero`, `unique`, `min`, `prod`, `amax`, `arange`, `full`, `ones` …

**同样缺失**：
* ❌ 无 `attention.py`
* ❌ 无 `rms_norm`（有 `layernorm.py`，但不是 RMSNorm）
* ❌ 无 `rotary_embedding`

### 2.3 fused 目录差异

| 文件 | 天数 | 沐曦 |
|---|---|---|
| `fused/sparse_attention.py` | ✅ 6.5 KB | ✅ 18.7 KB |
| `fused/top_k_per_row_prefill.py` | ✅ 7.5 KB | ✅ 7.5 KB |
| `fused/flash_mla.py` | ❌ | ✅ 6.6 KB |
| `fused/matmul_bias_activation.py` | ✅ 5.7 KB | ❌ |

**沐曦的 `sparse_attention.py` 是天数的 2.9 倍大**，说明沐曦侧的
attention 相关融合优化投入远大于天数。

### 2.4 算子黑名单

**天数 `_iluvatar/op_black_list.yaml:20-26`**
```yaml
unsupported_operators:
  - name: grouped_topk
    reason: All dtypes failed

  - name: topk_softmax
    reason: vllm not supported
```
→ 只有 2 个算子被禁，**且都与 MoE 路由有关**。MiniCPM5-2B 是 dense 模型，
**这两个黑名单对我们完全无影响**。

**沐曦 `_metax/op_black_list.yaml`**（1.9 KB，比天数大）

---

## 3. MiniCPM5-2B 热路径算子覆盖表

模型结构（来自 ModelScope `config.json`）：
```
LlamaForCausalLM, hidden 2048, 42 layers, 16 heads, 2 KV heads (GQA 8:1),
head_dim 128, intermediate 6144, vocab 130560, rope_theta 5e6,
max_pos 131072, bf16, tie_word_embeddings=false, SiLU
```

| 算子 | 每 token 调用次数 | FlagGems 通用实现 | 天数 | 沐曦 | 备注 |
|---|---|---|---|---|---|
| **RMSNorm** ×2/layer | 84 | `ops/rms_norm.py` | ❌ 无覆盖 | ❌ 无覆盖 | 走 torch 或 vLLM OOT |
| **RoPE** | 42 | `ops/rotary_embedding.py` | ❌ 无覆盖 | ❌ 无覆盖 | 同上 |
| **Attention prefill** | 42 | `ops/attention.py:1284` `flash_attn_varlen_func` | ⚠️ 通用 Triton | ⚠️ 通用 Triton | **天数实际不用它** |
| **Attention decode** | 42/token | 同上（paged） | ⚠️ | ⚠️ | 沐曦走 MACA 原生 |
| **QKV Linear** | 42 | `ops/mm.py` / `addmm.py` | ✅ `mm.py` 8.3KB | ✅ `mm.py` 33.9KB | 沐曦实现更大 |
| **O Linear** | 42 | 同上 | ✅ | ✅ | |
| **gate/up Linear** | 42 | 同上 | ✅ | ✅ | |
| **down Linear** | 42 | 同上 | ✅ | ✅ | |
| **SwiGLU** (`silu_and_mul`) | 42 | 通用 | ⚠️ | ⚠️ | 走插件 OOT |
| **Residual add** | 42 | `ops/add.py` | ✅ | ✅ | |
| **Embedding** | 1 | `ops/embedding.py` | ✅ | ✅ | |
| **lm_head** (2048×130560) | 1 | `ops/mm.py` | ✅ | ✅ | **267M 参数，2.5B 的 10.7%** |
| **Sampling** (top-k/top-p) | 1/token | `ops/topk.py`, `ops/softmax.py` | ⚠️ topk_softmax 被禁 | ✅ | 天数采样走 torch |

### 3.1 关键发现：FlagGems **有** FA2，但没人用

`src/flag_gems/ops/attention.py` 里是**完整的 FlashAttention-2 实现**：

```
L35   @triton.jit  def _attn_fwd_inner(...)
L176  @triton.jit  def _attn_fwd(...)
L368  @triton.jit  def _attn_bwd_preprocess(...)
L412  @triton.jit  def _attn_bwd_dkdv(...)
L512  @triton.jit  def _attn_bwd_dq(...)
L586  @triton.jit  def _attn_bwd(...)
L1284 def flash_attn_varlen_func(...)
```

`flash_attn_varlen_func` 的 docstring（`L1317-1368`）明确支持 GQA/MQA：
```
Supports multi-query and grouped-query attention (MQA/GQA) by passing in K, V
with fewer heads than Q. Note that the number of heads in Q must be divisible
by the number of heads in KV.
```

**MiniCPM5-2B 正是 16 Q heads / 2 KV heads = 8:1 GQA，16 % 2 == 0，完全满足。**

执行路径（`L1369-1462`）：
```python
if fa_version != 2:
    raise RuntimeError("Only FA2 is implemented.")
if num_splits > 0:
    raise RuntimeError("num_splits > 0 is not implemented in GEMS.")
if use_c_extension:
    ... torch.ops.flag_gems.flash_attn_varlen_func(...)
else:
    ...
    out, q, k, v, softmax_lse, *_ = mha_varlan_fwd(
        q, k, v, out, cu_seqlens_q, ...,
        block_table,          # ← 支持 paged KV
        alibi_slopes,
        max_seqlen_q, max_seqlen_k,
        dropout_p, softmax_scale, False, causal,
        real_window_size[0], real_window_size[1],
        softcap, ...)
```

`mha_varlan_fwd` 定义在 `src/flag_gems/ops/flash_api.py:250`。
**`block_table` 参数存在 = 支持 paged KV cache = 可以同时服务 prefill 和 decode。**

### 3.2 Paged KV 写入也有

`src/flag_gems/fused/reshape_and_cache_flash.py:77`
```python
def reshape_and_cache_flash(
    key,  # [num_tokens, num_heads, head_size]
    value,  # [num_tokens, num_heads, head_size]
    key_cache,  # [num_blocks, block_size, num_heads, head_size]
    value_cache,  # [num_blocks, block_size, num_heads, head_size]
    slot_mapping,  # [num_tokens]
    kv_cache_dtype,
    ...
```
对应 kernel `reshape_and_cache_flash_kernel`（`L32`）。

**所以 FlagGems 侧的 attention 栈是完整的**：
`flash_attn_varlen_func`（FA2 前向 + paged）+ `reshape_and_cache_flash`（KV 写入）。

而 vllm-plugin-FL 里 `AttentionFLBackend` 正是用这两个函数搭起来的：
`vllm_fl/dispatch/backends/flaggems/impl/attention.py:43`
```python
from flag_gems import flash_attn_varlen_func, reshape_and_cache_flash
```
`L460`
```python
self.vllm_flash_attn_version = 2  # FlagGems only supports FA2
```
`L488` 调 `reshape_and_cache_flash`，`L593/665/763` 调 `flash_attn_varlen_func`。

> ⭐ **结论：一条完整的、基于 FlagGems 的 FA2 attention 通路已经写好了，
> 只是默认关闭（`VLLM_FL_USE_FLAGGEMS_ATTN=0`）。
> 这是本赛题最大的一处「现成但未启用」的性能资产。**

---

## 4. Autotune 配置差异

### 天数 `_iluvatar/tune_configs.yaml:15-40`
```yaml
attention:
- gen: true
  param_map:
    META:
      BLOCK_M: block_m
      BLOCK_N: block_n
      PRE_LOAD_V: pre_load_v
    num_warps: warps
    num_stages: stages
  block_m: [64, 128]
  block_n: [32, 64, 128]
  pre_load_v: [true, false]
  warps: [4, 8]
  stages: [1, 2, 3]
```

### 沐曦 `_metax/tune_configs.yaml:15-38`
```yaml
attention:
- gen: true
  param_map:
    META:
      BLOCK_M: block_m
      BLOCK_N: block_n
      PRE_LOAD_V: pre_load_v
    num_warps: warps
    num_stages: stages
  block_m: [64, 128]
  block_n: [32, 64, 128]
  pre_load_v: [true, false]
  warps: [4, 8]
  stages: [1]
```

**差异**：天数多一档 `stages: 3`，沐曦只有 `stages: 1`。

搜索空间大小：
* 天数：2 × 3 × 2 × 2 × 3 = **72 种配置**
* 沐曦：2 × 3 × 2 × 2 × 1 = **24 种配置**

> ⚠️ 但这套 autotune 只在 **FlagGems 的 attention 被实际调用时**才起作用。
> 当前天数的 attention 走 `TRITON_ATTN`（vLLM 自己的 kernel），
> **FlagGems 的这份 autotune 配置根本没被用到**。
> 一旦我们把 attention 切到 FlagGems FA2，这份 72 点的搜索空间才真正生效 ——
> 而且这正是「算子库优化」加分项的直接抓手。

---

## 5. 融合机会

### 已有
* `fused/reshape_and_cache_flash.py` —— KV 写入融合
* `fused/sparse_attention.py` —— 稀疏 attention
* `fused/top_k_per_row_prefill.py` —— prefill 期 top-k 路由
* 天数独有：`fused/matmul_bias_activation.py` —— **matmul + bias + 激活融合**
* 沐曦独有：`fused/flash_mla.py`

### 缺失（Llama 类模型会受益）
| 融合 | 现状 | 收益 |
|---|---|---|
| `fused_add_rms_norm`（residual + RMSNorm） | ❌ 无 | 42 层 × 2 次 = 84 次 kernel 合并成 42 次 |
| `rotary_embedding` + KV cache write 融合 | ❌ 无 | 42 次 kernel 合并 |
| `silu_and_mul` 融合（SwiGLU） | ⚠️ 靠插件 OOT，非 FlagGems | 42 次 |
| `qkv` 三合一 matmul | ❌ 无 | 42 次 |

> **`fused_add_rms_norm` 是最值钱的一个**：Llama 每层有
> `h = x + attn(rms_norm(x))` 和 `out = h + mlp(rms_norm(h))`，
> 两个 RMSNorm 都可以和 residual add 融合。42 层 → 省 42 次 kernel launch +
> 42 次 hidden state 往返显存（2048 × bf16 × batch）。

---

## 6. 调优旋钮

| 旋钮 | 位置 | 作用 |
|---|---|---|
| `tune_configs.yaml` per-vendor | 各厂商目录 | autotune 搜索空间 |
| `heuristics_config_utils.py` per-vendor | 各厂商目录 | 无 autotune 时的启发式块大小 |
| `op_black_list.yaml` per-vendor | 各厂商目录 | 禁用算子 |
| `mm_metax_expand.yaml` / `general_ops_metax_expand.yaml` | `_metax/` | 沐曦 matmul 展开配置 |
| triton cache dir | 环境变量 `TRITON_CACHE_DIR` | 冷编译开销（影响可复现性） |

天数 `_iluvatar/heuristics_config_utils.py:19-21`
```python
def simple_elementwise_blocksize_heur(args):
    return 1024
```
→ 元素级算子固定 1024 块大小。**BI-V150 的 SM 数量决定这个值是否合理**，
上卡后应实测。

---

## 7. 测试与基准位置

* 精度测试：`tests/`（FlagGems 仓库根），与 torch 参考对拍
* 性能基准：`benchmark/` 目录
* 具体命令：**UNVERIFIED** —— 未在源码树中逐一确认，上卡时用
  `ls upstream/FlagGems/tests upstream/FlagGems/benchmark` 确认

> 对我们更重要的是**赛事方给的**评测：
> `evalscope` + `math_500` Level 3（精度），`benchmark_throughput_serve.py`（吞吐）。

---

## 8. 排序后的 kernel 级机会

| # | 算子 | 位置 | 现状弱点 | 建议改动 | 工作量 | 预期影响 |
|---|---|---|---|---|---|---|
| **1** | attention（让 FlagGems FA2 真正生效） | `ops/attention.py:1284` + `_iluvatar/tune_configs.yaml:15` | 天数根本不走它 | 切后端 + 用 BI-V150 实测重新 autotune | M | **极大** |
| **2** | `fused_add_rms_norm` | 新增 | 完全缺失 | 写 Triton kernel，residual+norm 融合 | M | 大（84→42 kernel） |
| **3** | RoPE + KV write 融合 | 新增 | 两次 kernel | 合并成一个 Triton kernel | M | 中 |
| **4** | 天数 `mm` 实现 | `_iluvatar/ops/mm.py` 8.3KB vs 沐曦 33.9KB | 天数实现明显更薄 | 对照沐曦实现补 tiling / 分块 | M | 中 |
| **5** | `simple_elementwise_blocksize_heur` = 1024 | `_iluvatar/heuristics_config_utils.py:19-21` | 硬编码 | 按 BI-V150 SM 数实测调 | S | 小-中 |
| **6** | attention autotune 扩展 | `_iluvatar/tune_configs.yaml` | 只有 72 点，且 perf model 用常量估算 | 加 BLOCK_K、加 stage 组合、用实测 profile | M | 中 |
| **7** | 天数 `topk_softmax` 被禁 | `_iluvatar/op_black_list.yaml:24-26` | `vllm not supported` | 修好它，采样路径回 FlagGems | M | 小-中 |
| **8** | `matmul_bias_activation` 用起来 | `_iluvatar/fused/matmul_bias_activation.py` | 存在但未接入 vLLM | 在 dispatch 注册 | S-M | 中 |
| **9** | `sparse_attention` 移植沐曦版 | 天数 6.5KB vs 沐曦 18.7KB | 天数版本薄 | 对照补全 | L | 中 |
| **10** | lm_head matmul（2048×130560） | `ops/mm.py` | 单次大 matmul，vocab 130560 | 分块 + 融合采样 | L | 小-中 |

---

## 9. 未验证项

| 项 | 需要什么 |
|---|---|
| 各算子在 BI-V150 / C500 上的实际耗时占比 | 上卡 profiler |
| FlagGems `mm` 在天数上是否真的快于 cuBLAS 兼容库 | 上卡 benchmark |
| 天数 triton 版本到底是 3.2 还是 3.6 | 上卡 `triton.__version__` |
| `tests/` 与 `benchmark/` 的确切命令 | 上卡 `ls` |
| 沐曦 `op_black_list.yaml` 的具体内容 | 已读但未在本文展开，见源文件 |