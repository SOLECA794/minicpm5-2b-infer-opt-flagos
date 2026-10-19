# vllm-plugin-FL 优化面测绘

> 源码：`upstream/vllm-plugin-FL` @ branch `flagos-2026-s2`（commit `13eb9be`，基于 vLLM `0.24.0`）
> 方法：逐文件静态阅读。**每条结论都附 `file:line` 与原文摘录**。
> 未能从源码验证的，标注 `UNVERIFIED` 并写明需要什么才能验证。

---

## 0. 结论先行：一句话

**天数的 attention 走的是 vLLM 通用 `TRITON_ATTN`，沐曦走的是 MACA 原生 `FLASH_ATTN`。
这就是 16k 场景 7.68× 差距的根因。** 详见 [`root-cause-tianshu-16k.md`](root-cause-tianshu-16k.md)。

---

## 1. 插件入口与装配

### 1.1 注册方式

`VLLM_PLUGINS=fl` 通过 vLLM 的插件机制加载。插件把 `PlatformFL` 注册为 OOT platform：

`vllm_fl/platform.py:55-57`
```python
class PlatformFL(Platform):
    _enum = PlatformEnum.OOT
    device_info = DeviceInfo()
    vendor_name = device_info.vendor_name
```

Worker 类被整体替换：

`vllm_fl/platform.py:180`
```python
parallel_config.worker_cls = "vllm_fl.worker.worker.WorkerFL"
```

`WorkerFL` 指向 `vllm_fl/worker/model_runner.py`（**6866 行**，仓库最大文件）。
这是被替换的 vLLM 核心推理循环。

### 1.2 厂商归一化

`vllm_fl/utils.py:35-54`
```python
VENDOR_DEVICE_MAP: dict[str, dict[str, str]] = {
    ...
    # Registered backend: vendor/iluvatar
    "iluvatar": {"device_type": "cuda", "device_name": "cuda"},
    # Registered backend: vendor/metax
    "metax": {"device_type": "cuda", "device_name": "metax"},
    ...
}
```

关键差异：**天数 `device_name="cuda"`，沐曦 `device_name="metax"`**。
这个字符串会一路影响后续所有「像不像 NVIDIA」的判断。

`vllm_fl/platform.py:61-67`
```python
# cuda_alike (nvidia/metax): device_name = vendor_name (not used in torch.device)
# non-cuda_alike (iluvatar/ascend): device_name = device_type (used in torch.device)
device_name = device_info.vendor_name if (
    device_info.device_type == "cuda"
    and device_info.vendor_name not in ("iluvatar", "hygon")
) else device_info.device_type
```

`vllm_fl/platform.py:81-89`
```python
def is_cuda_alike(self) -> bool:
    """Stateless version of [torch.cuda.is_available][]."""
    if self.vendor_name == "iluvatar":
        return False
    ...
```

**天数被显式排除在 `is_cuda_alike` 之外。** 凡上游代码写
`if current_platform.is_cuda_alike(): <fast path>` 的地方，天数都会走 else 分支。

---

## 2. 环境变量清单（我们的合法调优旋钮）

serve 参数被官方锁定，但**环境变量不在锁定范围内**——不过注意：评测方是自己
`export VLLM_PLUGINS=fl` 后启动的，我们无法给他们加 env。所以：

> **env 只能用于我们自己的实验；真正生效的必须是改代码把默认值改对。**

| 变量 | 默认 | 作用 | 出处 |
|---|---|---|---|
| `VLLM_FL_PREFER_ENABLED` | `"True"` | 插件总开关，false 则完全退回原生 vLLM | `utils.py:109`、`envs.py:9-11` |
| `VLLM_FL_PREFER` | `""` | 若非空且不为 `flagos`，禁用全部 FlagGems | `utils.py:111-113` |
| `USE_FLAGGEMS` | `True` | FlagGems 总开关 | `utils.py:114-117`、`envs.py:15` |
| `VLLM_FL_FLAGOS_WHITELIST` | 未设 | **只有**列出的算子用 FlagGems | `utils.py:144-161` |
| `VLLM_FL_FLAGOS_BLACKLIST` | 未设 | 列出的算子**不**用 FlagGems | `utils.py:164-165` |
| `VLLM_FL_FLAGOS_BLACKLIST_APPEND` | 未设 | 追加到平台 config 黑名单 | `utils.py:179-187` |
| `VLLM_FL_OP_CONFIG` | 未设 | 外部 JSON 覆盖算子配置 | `utils.py:223-247` |
| `VLLM_FL_OOT_WHITELIST` | 未设 | 只注册列出的 OOT 算子 | `utils.py:311-324` |
| `VLLM_FL_OOT_BLACKLIST` | 未设 | 不注册列出的 OOT 算子 | `utils.py:327-361` |
| `VLLM_FL_OOT_ENABLED` | `"1"` | OOT 注册总开关 | `utils.py:364-377` |
| `VLLM_FL_USE_FLAGGEMS_ATTN` | **`"0"`** | ⭐ 开启 FlagGems 的 FA2 attention backend | `flaggems.py:179-185` |
| `FLAGGEMS_ENABLE_OPLIST_PATH` | `/tmp/flaggems_enable_oplist.txt` | FlagGems 算子表落盘路径 | `envs.py:12-14` |
| `FLAGOS_DEVICE_CONTROL_ENV_VAR` | 未设 | 覆盖设备可见性变量名 | `utils.py:100-105` |

> ⚠️ `VLLM_FL_FLAGOS_WHITELIST` 与 `VLLM_FL_FLAGOS_BLACKLIST` **不可同时设置**，
> 同时设会 `raise ValueError`（`utils.py:149-153`）。这是启动即崩的坑。

### 2.1 OOT 算子清单（插件自己重写的算子）

`vllm_fl/utils.py:301-308`
```python
OOT_OP_NAMES = [
    "silu_and_mul",
    "gelu_and_mul",
    "rms_norm",
    "rotary_embedding",
    "fused_moe",
    "unquantized_fused_moe_method",
]
```

这 6 个算子是插件直接替换 vLLM 原生实现的地方 —— **MiniCPM5-2B 用到其中 4 个**
（`silu_and_mul`、`rms_norm`、`rotary_embedding`；`fused_moe` 不适用于 dense 模型）。

---

## 3. 算子分发层（`vllm_fl/dispatch/`）

### 3.1 分发优先级

`vllm_fl/platform.py:264-290` —— attention backend 也是走分发层的：
```python
@classmethod
def get_attn_backend_cls(
    cls,
    selected_backend: "AttentionBackendEnum | None",
    attn_selector_config: "AttentionSelectorConfig",
    num_heads: int | None = None,
) -> str:
    """Get the attention backend class path using the dispatch mechanism."""
    from vllm_fl.dispatch import call_op

    use_mla = attn_selector_config.use_mla
    use_sparse = attn_selector_config.use_sparse

    backend_path = call_op("attention_backend", use_mla=use_mla, use_sparse=use_sparse)
```

**这意味着 attention 后端是「可配置的算子」，不是硬编码的。**
→ 我们可以在配置层直接改，这是被框架设计者预留的扩展点。

### 3.2 平台配置文件

`vllm_fl/dispatch/config/` 下按厂商自动加载。

**天数 `iluvatar.yaml:30-47`**
```yaml
op_backends:
  # All operators: prioritize FlagGems (Triton), fallback to vendor, then reference
  attention_backend:
    - flagos
    - vendor:iluvatar
    - reference
  rms_norm:
    - flagos
    - vendor:iluvatar
    - reference
  silu_and_mul:
    - flagos
    - vendor:iluvatar
    - reference
  rotary_embedding:
    - flagos
    - vendor:iluvatar
    - reference

flagos_blacklist: []
```

**沐曦 `metax.yaml:28-44`**
```yaml
op_backends:
  # attention_backend: prioritize vendor (MACA attention)
  attention_backend:
    - vendor:metax
  # All other ops: prefer FlagGems (Triton), fallback to reference (PyTorch native)
  silu_and_mul:
    - flagos
    - reference
  rms_norm:
    - flagos
    - reference
  rotary_embedding:
    - flagos
    - reference
  topk_softmax:
    - flagos
    - reference

flagos_blacklist:
  - masked_fill
  - masked_fill_
```

**对照表**

| 算子 | 天数候选顺序 | 沐曦候选顺序 |
|---|---|---|
| `attention_backend` | `flagos` → `vendor:iluvatar` → `reference` | **`vendor:metax`（唯一）** |
| `rms_norm` | `flagos` → `vendor:iluvatar` → `reference` | `flagos` → `reference` |
| `silu_and_mul` | `flagos` → `vendor:iluvatar` → `reference` | `flagos` → `reference` |
| `rotary_embedding` | `flagos` → `vendor:iluvatar` → `reference` | `flagos` → `reference` |

**注意**：天数把 `vendor:iluvatar` 放在 `flagos` 之后作为**兜底**，沐曦把
`vendor:metax` 作为 attention 的**唯一**来源。

### 3.3 各方 attention_backend 的实际返回值

**FlagGems 后端 `vllm_fl/dispatch/backends/flaggems/flaggems.py:153-187`**
```python
def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # TritonAttentionBackend requires CUDA, check if available
    if not torch.cuda.is_available():
        raise RuntimeError(
            "TritonAttentionBackend requires CUDA but CUDA is not available. "
            "Falling back to vendor implementation."
        )
    ...
    use_flaggems_attn = os.environ.get(
        "VLLM_FL_USE_FLAGGEMS_ATTN", "0"
    ).lower() in ("1", "true", "yes")

    if use_flaggems_attn:
        print("Using FlagGems attention backend.")
        return "vllm_fl.dispatch.backends.flaggems.impl.attention.AttentionFLBackend"

    return AttentionBackendEnum.TRITON_ATTN.get_path()
```

> ⭐ **`VLLM_FL_USE_FLAGGEMS_ATTN` 默认 `"0"`。**
> FlagGems 已经实现了一个完整的 FA2 风格 attention backend
> （`AttentionFLBackend`，857 行，改编自 vLLM `flash_attn.py`），
> **但它默认是关的**。默认路径会落到 `TRITON_ATTN`。

**天数厂商后端 `vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py:462-486`**
```python
def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if use_mla:
        if use_sparse:
            return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
        return AttentionBackendEnum.FLASHMLA.get_path()

    # flash_attn is not available on iluvatar. Use TRITON_ATTN (the vllm
    # default). The tl.make_tensor_descriptor stub and perf model patches
    # are already applied at module level (above), so triton JIT's
    # DependencyFinder can hash the kernel without AttributeError on
    # triton < 3.3.  The kernel itself uses USE_TD=False at runtime, so
    # the stub is never called.
    return AttentionBackendEnum.TRITON_ATTN.get_path()
```

原文注释直接承认：**`flash_attn is not available on iluvatar`**，
所以退回 `TRITON_ATTN`。

**沐曦厂商后端 `vllm_fl/dispatch/backends/vendor/metax/metax.py:21-29, 135-151`**
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
```python
def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    # register before selection
    register_attention_backends()

    if use_mla:
        if use_sparse:
            return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
        return AttentionBackendEnum.FLASHMLA.get_path()

    # Default to FLASH_ATTN
    return AttentionBackendEnum.FLASH_ATTN.get_path()
```

沐曦有**真正的 FA2 后端实现**：
* `vendor/metax/impl/attention/flash_attn.py`（1085 行）
* `vendor/metax/impl/attention/ops/triton_unified_attention.py`（906 行）
* `vendor/metax/impl/attention/ops/triton_decode_attention.py`（637 行）

而天数的 `vendor/iluvatar/` 目录下**只有 3 个 impl 文件**：
```
vendor/iluvatar/impl/activation.py
vendor/iluvatar/impl/normalization.py
vendor/iluvatar/impl/rotary.py
```
**没有 attention。**

### 3.4 最终判定表

| | 天数 BI-V150 | 沐曦 C500 |
|---|---|---|
| 配置指定的候选 | `flagos` → `vendor:iluvatar` → `reference` | `vendor:metax` |
| `flagos` 返回什么 | `TRITON_ATTN`（`VLLM_FL_USE_FLAGGEMS_ATTN=0`） | — |
| `vendor:*` 返回什么 | `TRITON_ATTN`（无 FA） | `FLASH_ATTN`（MACA 原生 FA2） |
| **实际后端** | **`TRITON_ATTN`** | **`FLASH_ATTN`** |
| 硬件上有无 flash-attn 库 | ❌ 无（无 NVML，无 FA） | ✅ MACA 自带 |

---

## 4. 天数专属 workaround 清单（全是性能风险点）

天数后端文件 `vendor/iluvatar/iluvatar.py` 里塞了大量**降级/兼容补丁**，
每一个都对应一处性能或功能损失。

### 4.1 Triton 版本落后导致的语言层 patch

`iluvatar.py:67-156` `patch_triton_chained_or_for_iluvatar()`
```python
"""Rewrite chained boolean 'or' chains in vllm triton kernels.

Iluvatar's triton version (3.2.x) raises UnsupportedLanguageConstruct for
chained boolean operators like ``A or B or C`` inside @triton.jit functions.
Rewrites the affected source file in-place (idempotent).
"""
```
→ **天数上是 Triton 3.2.x**（沐曦/现代 NVIDIA 是 3.3+）。
它靠**正则改写 vLLM 源码文件**来绕过语言限制，脆弱且只对
`vllm.v1.attention.ops.triton_attention_helpers` 一个文件生效。

`iluvatar.py:22-64` `patch_triton_language_for_iluvatar()`
```python
"""Add make_tensor_descriptor stub to triton.language for triton < 3.3.
...
TODO: Remove once minimum supported triton version is >= 3.3.
"""
```
→ TMA（Tensor Memory Accelerator）描述符**不可用**，`USE_TD=False`。
**这是长序列 attention 性能的直接损失**：没有 TMA，prefill 的
K/V 访存只能走普通 global load。

### 4.2 Inductor / torch.compile 被削弱

`iluvatar.py:238-332` `patch_torch_inductor_for_iluvatar()`
```python
"""Patch torch._inductor GPUTarget to use the correct triton backend name.

torch._inductor always passes device_type='cuda' to GPUTarget, but
flagtree triton only registers an 'iluvatar' (3.6.x) or 'corex' (3.2.x)
backend -- never 'cuda'.
"""
```

`iluvatar.py:210-235` `patch_sampler_compile_for_iluvatar()`
```python
def patch_sampler_compile_for_iluvatar() -> None:
    # Disable torch.compile on vllm sampler ops for Iluvatar.
    # flagtree triton only supports Iluvatar backend, not cuda target.
    # TODO: Remove once flagtree triton supports cuda inductor target.
    ...
        _orig = _tts.compiled_random_sample
        _tts.compiled_random_sample = getattr(_orig, '__wrapped__', _orig)
```

→ **采样器的 `torch.compile` 被强制解包（关闭）**。
decode 阶段每步都要跑采样，这是 per-token 的固定开销。
在 4k 场景（256×1024 tokens 输出）这条路径被调用 262144 次。

### 4.3 没有 NVML

`vllm_fl/platform.py:412-417`
```python
if cls.device_type == "cuda" and cls.vendor_name == "iluvatar":
    # Iluvatar is CUDA-compatible but has no NVML library.
    # Return a synthetic UUID based on device index.
    return f"ILUVATAR-{device_id}"
```

`iluvatar.py:159-203` `patch_triton_perf_model_for_iluvatar()`
```python
"""...On iluvatar, neither `ixsmi` nor `libnvidia-ml.so` is available, so the
default implementation crashes. We replace it with a fixed value that
matches typical iluvatar BI-V150 SM clock (1500 MHz = 1500000 kHz)."""
```
→ **BI-V150 SM 主频 1500 MHz**（这是官方数字，可用于 roofline 估算）。
且 triton 的 autotune 性能模型用的是**常量估算**，不是真实 profil——→
**autotune 的配置选择在天数上可能是次优的**，这是 kernel 调参的切入口。

### 4.4 static graph 支持

`vllm_fl/platform.py:343-351`
```python
@classmethod
def get_static_graph_wrapper_cls(cls) -> str:
    return "vllm_fl.compilation.graph.GraphWrapper"

@classmethod
def support_static_graph_mode(cls) -> bool:
    if cls.vendor_name in ["nvidia", "ascend", "metax", "hygon", "mthreads", "iluvatar"]:
        return True
    return False
```

天数和沐曦都声明支持 static graph。但**只有天数在 serve 时带了**
`--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`。
沐曦没带这个参数。

`vllm_fl/platform.py:381-384`
```python
### NOTE(lms): will effect compile result
@classmethod
def opaque_attention_op(cls) -> bool:
    return True
```
→ attention 算子对编译器不透明，**不能被融合进 cudagraph 的图内**。
这是 decode 阶段 kernel launch 开销无法消除的直接原因。

### 4.5 KV cache block size

`vllm_fl/platform.py:183-193`
```python
cache_config = vllm_config.cache_config
if cache_config and cache_config.block_size is None:
    if cls.device_type == "npu":
        cache_config.block_size = 128
    elif cls.device_type == "musa":
        cache_config.block_size = 64
    else:
        cache_config.block_size = 16
```
→ 天数和沐曦都落进 `else`，**`block_size = 16`**。
16 是最小的 block，KV cache 碎片最少但 block table 最长、索引开销最高。
对 131072 max-model-len 这种配置，是否应该用 32/64 值得实验
（注意：这是**框架侧代码**，可以改，但要保证不改 serve 参数语义）。

### 4.6 沐曦的特殊处理

`vllm_fl/platform.py:145-159` —— 沐曦会 import `mcoplib._C`、`mcoplib._moe_C`
和 MACA patches。

`vllm_fl/platform.py:254-261`
```python
# maca specific config updates
if cls.vendor_name == "metax":
    if model_config is not None:
        model_config.disable_cascade_attn = True
    if attention_config := vllm_config.attention_config:
        attention_config.use_cudnn_prefill = False
        attention_config.use_trtllm_ragged_deepseek_prefill = False
        attention_config.use_trtllm_attention = False
        attention_config.disable_flashinfer_prefill = True
```
→ 沐曦关掉了**级联 attention**（`disable_cascade_attn=True`）。
如果我们能让沐曦的级联/prefill 路径复活，是沐曦侧的增量点。

---

## 5. 编译 / 图捕获

* `PlatformFL.get_static_graph_wrapper_cls()` → `vllm_fl.compilation.graph.GraphWrapper`（`platform.py:343-345`）
* `vllm_fl/compilation/graph.py` 与 `break_graph.py` 是图捕获的接管实现
* `opaque_attention_op() == True`（`platform.py:381-384`）→ attention 不进图

**天数 serve 带了 `cudagraph_mode=FULL_DECODE_ONLY`**，
意味着只对 decode 做整图捕获；prefill 不捕获（prefill 形状变化大，正常）。
但 attention 是 opaque 的，所以 **decode 图里每步仍然有一次 attention 的
独立 kernel launch 逃逸在图外**。

---

## 6. 排序后的优化目标

| # | 目标 | 位置 | 为什么慢 | 工作量 | 风险 | 预期指标 |
|---|---|---|---|---|---|---|
| **1** | 天数 attention 换掉 `TRITON_ATTN` | `flaggems.py:179-187` / `iluvatar.yaml:32-35` | 通用 Triton unified attention 在长 prefill 上是 O(S²) 慢路径，且无 TMA | M | 中（需数值对齐验证） | **天数 16k tok/s +100%~200%，TTFT ↓ 90%** |
| **2** | 开启并用好 FlagGems FA2（`AttentionFLBackend`） | `flaggems/impl/attention.py`（857 行，默认关） | 默认 `VLLM_FL_USE_FLAGGEMS_ATTN=0`，现成实现被搁置 | M | 中 | 同上，且**算子库侧加分** |
| **3** | 天数 sampler 的 compile 恢复 | `iluvatar.py:210-235` | 被强制解包，每 token 走 Python | S | 低 | 天数 4k decode 吞吐 |
| **4** | KV cache block_size 调优 | `platform.py:183-193` | 固定 16，block table 过长 | S | 低 | 两卡通用 |
| **5** | 天数 attention autotune 配置 | `FlagGems/_iluvatar/tune_configs.yaml:15-40` | 只有 `stages 1/2/3`，且 perf model 用常量 1500MHz 估算 | M | 低 | prefill kernel 直接变快 |
| **6** | 沐曦 `disable_cascade_attn` 评估 | `platform.py:254-256` | 被硬关，可能过度保守 | S | 中 | 沐曦 16k |
| **7** | `opaque_attention_op` 改为 False | `platform.py:381-384` | attention 无法进图 | L | 高 | decode 全图化 |

> ⚠️ 第 7 项风险高：`opaque_attention_op=True` 是有意为之，
> 改成 False 可能直接编译失败或产出错误结果。**优先级放最后。**

---

## 7. 未验证项（诚实登记）

| 项 | 需要什么才能验证 |
|---|---|
| `TRITON_ATTN` 在 BI-V150 上的实际 kernel 时间占比 | 上卡 + profiler |
| `AttentionFLBackend` 在天数上的数值正确性 | 上卡 + 与 baseline 对拍 |
| `VLLM_FL_USE_FLAGGEMS_ATTN=1` 是否有隐藏 bug | 上卡端到端 |
| 天数 triton 实际版本（源码只写「3.2.x」） | 上卡 `python -c "import triton; print(triton.__version__)"` |
| `block_size` 是否能被 vLLM 0.24 安全接受为 32/64 | 上卡启动测试 |
| 沐曦不带 cudagraph 配置时的实际图捕获行为 | 上卡看启动日志 |