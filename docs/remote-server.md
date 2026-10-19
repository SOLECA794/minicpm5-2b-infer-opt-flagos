# 远程实验机接入说明

赛题提供了沐曦曦云 C500 的实验机。本文记录接入方式、实测规格与已验证的可用/不可用清单，
供后续实验直接复用。

---

## 1. 接入方式

机器位于 **SSHPiper** 网关之后（`140.207.205.81:32222`），只支持**密码认证**，且
scp/sftp 数据通道被网关拦截。因此本仓库自带一个基于 `pexpect` 的小工具
[`scripts/remote/sshx.py`](../scripts/remote/sshx.py) 承担登录与文件传输。

### 1.1 凭据

凭据**不入库**，从下列任一来源读取（优先级从上到下）：

| 来源 | 说明 |
|---|---|
| 环境变量 | `SSHX_HOST` / `SSHX_PORT` / `SSHX_USER` / `SSHX_PASS` |
| `~/.ssh/sshx.json` | WSL 内的 home，即 `/home/<user>/.ssh/sshx.json` |

`sshx.json` 结构：

```json
{
  "password": "<统一密码>",
  "default": "<user>@<gateway>",
  "hosts": {
    "metax-a": { "host": "140.207.205.81", "port": 32222, "user": "root+vm-XXXXXXXX" },
    "metax-b": { "host": "140.207.205.81", "port": 32222, "user": "root+vm-YYYYYYYY" }
  }
}
```

所有实验机使用同一密码；`user` 形如 `root+vm-<实例ID>`，实例 ID 与机器一一对应。

### 1.2 运行环境

`sshx.py` 依赖 `pexpect`（已验证 4.9.0），在 **WSL 的 python3** 下运行。
`paramiko` 对该网关不可用（握手阶段即被拒），故不采用。

```bash
# WSL 内安装依赖
pip3 install pexpect

# 语法
python3 scripts/remote/sshx.py exec '<remote command>'          [--host metax-a]
python3 scripts/remote/sshx.py put  <local> <remote>            [--host metax-a]
python3 scripts/remote/sshx.py get  <remote> <local>            [--host metax-a]
```

从 Windows 侧调用的完整姿势：

```powershell
wsl bash -lc "python3 '/mnt/d/Desktop/minicpm5‑2b‑infer‑opt‑flagos/scripts/remote/sshx.py' exec 'mx-smi'"
```

> 路径中的连字符是 U+2011（非断字连字符），不是 ASCII `-`。

### 1.3 必须携带的 SSH 选项

网关对以下默认值敏感，`sshx.py` 已内置，手工 `ssh` 时也需照抄：

```
KexAlgorithms=diffie-hellman-group14-sha256
StrictHostKeyChecking=no
UserKnownHostsFile=/dev/null
PubkeyAuthentication=no
PreferredAuthentications=password
NumberOfPasswordPrompts=1
ControlMaster=no
ControlPath=none
```

* `PubkeyAuthentication=no` —— 公钥尝试会消耗一次往返，且可能触发网关限流。
* `ControlMaster=no` —— 复用连接会在网关侧报 `getsockname failed: Not a socket`。

### 1.4 文件传输的实现方式

scp/sftp 不可用，因此 `put` / `get` 走 **base64 分块 + 单条 `exec`** 的方式：

* 单块原始大小 `CHUNK_RAW = 90 KB`（base64 后约 120 KB，低于单参数上限 131071 字节）；
* 每块附带 md5 校验，失败重试 5 次并指数退避；
* `get` 方向使用远端 `base64 -w0` 编码后回传。

这条路径适合传脚本、配置与结果 CSV，**不适合传模型权重**。大文件应在远端直接下载。

---

## 2. 已实测的机器规格

以下数据由 `mx-smi`、`/proc/cpuinfo`、`torch.cuda` 实测得到，非配置文档推断。

| 项目 | 实测值 |
|---|---|
| 容器 hostname | `50b964409cc5` |
| 内核 | Linux 5.15.0-58-generic x86_64 |
| CPU | 128 核 |
| 内存 | 32 GB |
| 系统盘 `/` | overlay 30 GB |
| 数据盘 `/data` | 98 GB（可用 93 GB） |
| GPU | **1 × MetaX C500** |
| `mx-smi` | 2.3.1 |
| 驱动 | 3.8.30 |
| MACA | 3.7.1.5 |
| BIOS | 1.33.5.0 |
| 功耗上限 | 350 W |

### 2.1 关键限制：当前是 sGPU 切片，不是整卡

```
Minor 006 / GPU 0 / sGPU-Id 3 / Compute 25% / Vram Quota 0/16000 MiB
torch.cuda.get_device_properties(0).total_memory = 15.22 GiB
```

即 **1/4 算力 + 16 GB 显存**，而赛题基线与评测要求的是完整 64 GB 的 C500。
因此这台机器可以承担功能验证与 kernel 级微基准，**无法承担 16k 场景的基线复现**
（16k × 64 并发 + 131072 max-model-len 的 KV cache 远超 16 GB）。

### 2.2 软件栈

| 组件 | 版本 / 位置 |
|---|---|
| Python | `/opt/conda/bin/python` = 3.12.11 |
| torch | `2.8.0+metax3.7.1.3`（`torch.cuda.is_available() = True`） |
| triton | `3.0.0+metax3.7.1.3` |
| MACA 安装 | `/opt/maca`（另有 `maca-20260921`、`maca-3.7.1`） |
| 驱动 | `/opt/mxdriver` |
| Profiler | `/opt/mcProfiler-ubuntu18.04` |
| TileLang | `/opt/tilelang-metax-v0.1.10` |

已预装的沐曦定制包：`flash_attn 2.6.3`、`flash_mla 1.0.1`、`flashinfer 0.2.6`、
`sageattention 2.0.1`、`mamba_ssm 2.2.4`、`xformers 0.0.22`、`transformers 5.8.0`。

**未安装**：`vllm`、`FlagGems`、`vllm-plugin-FL`。这三者是赛题框架，需自行安装。

### 2.3 必设环境变量

```bash
export MACA_PATH=/opt/maca
```

不设该变量时 `import triton` 直接失败：

```
TypeError: expected str, bytes or os.PathLike object, not NoneType
```

原因是沐曦的 triton 在导入期读取 `MACA_PATH` 定位 `libmaca`。所有远端命令都应前置
这一行，建议写进 `/root/.bashrc`。

---

## 3. 已验证可用 / 不可用清单

| 能力 | 状态 | 备注 |
|---|---|---|
| SSH 登录 + 单条 `exec` | ✅ | 见 §1.2 |
| base64 分块传文件 | ✅ | 见 §1.4 |
| `mx-smi` 查询 | ✅ | 2.3.1 |
| `torch.cuda` 识别设备 | ✅ | 需 `MACA_PATH` |
| `import triton` | ✅ | 需 `MACA_PATH` |
| `scp` / `sftp` | ❌ | 网关拦截数据通道 |
| `paramiko` | ❌ | 握手被拒 |
| 整卡 64 GB 显存 | ❌ | 当前为 16 GB 切片 |
| 16k 场景基线复现 | ❌ | 显存不足，需整卡 |

---

## 4. 下一步：在这台机器上能做什么

在拿到整卡之前，这台 16 GB 切片上仍可推进以下工作，且都不依赖 16k 场景：

1. **框架可安装性验证** —— 装 `vllm`（沐曦适配版）、`FlagGems v5.3.5`、
   `vllm-plugin-FL@flagos-2026-s2`，确认能 import、能起服务。
2. **算子功能验证** —— 用 4k 以内的短序列跑 `AttentionFLBackend`
   （即 `VLLM_FL_USE_FLAGGEMS_ATTN=1`），先确认 FlagGems FA2 在沐曦上数值正确，
   再谈天数。这是技术方案 P0 的最小可行前置。
3. **kernel 级微基准** —— FlagGems FA2 的 `flash_attn_varlen_func` 与
   `reshape_and_cache_flash` 单独计时，对照 MACA 原生 `flash_attn 2.6.3`，
   量出差距后再决定是否改 kernel。
4. **`fused_add_rms_norm` 原型** —— P2 的融合算子只需单卡小 batch 即可验证
   数值与带宽收益，与显存容量无关。
5. **profiling 工具链打通** —— `/opt/mcProfiler-ubuntu18.04` 的采集与解析流程。

以上 1–3 项对应技术方案 §4.1 的 M0 前置工作，可在正式算力到位前完成大半。

---

## 5. 待确认事项

* 完整 64 GB 的 C500 实例如何申请、是否与当前 16 GB 切片同网关 —— 需向平台确认。
* 是否可同时开 2–3 台（赛题两卡并行调试会显著缩短 M2 阶段）。
* sGPU 切片的 `Compute 25%` 是固定配额还是可调 —— 若可调到 100%，
  即使显存仍为 16 GB，prefill 相关的 kernel 计时也更有参考价值。
