# HANDOFF — 交接说明

> 最后更新：2026-10-20 23:05 (UTC+8)
> 仓库：`origin/main`（`git log -1` 取最新提交；本文随该分支推送）
> 一句话状态：**材料与工具链都已就绪（M0 已可一键跑）；服务器仍关机；M2 的合规前提今晚被重新审读，方案的表述必须改写。**

本文写给「下一次接手的人（或下一次的自己）」。读完后应当能在 10 分钟内知道：
现在在哪、什么已验证、什么被卡住、下一步具体做什么。

---

## 1. 三十秒速览

| 项目 | 状态 | 位置 |
|---|---|---|
| 赛事情报（规则/基线/赛程） | ✅ 已本地化 | `docs/competition/` |
| 源码调研（含 file:line 证据） | ✅ 已完成 | `docs/research/` |
| 根因定位（天数 16k 的 599 s TTFT） | ✅ 已定位 | `docs/research/root-cause-tianshu-16k.md` |
| 技术方案正文 | ✅ 正式书面体 | `proposals/01-initial-technical-plan.md` |
| 团队信息 | ✅ 已填全 | `proposals/team.yaml` |
| 算力申请 PDF | ✅ 已生成，待投递 | `proposals/算力申请-Casprian.pdf`（10 页 / 0.49 MB，gitignored） |
| 远程实验机接入 | ⏸ 工具已验证，**实例已关机** | `scripts/remote/sshx.py`、`docs/remote-server.md` |
| 远程编排 / 安装脚本 | ✅ **已写好并入库**（6421748） | `scripts/remote/run_m0.py`、`probe_env.sh`、`install_stack.sh` |
| 框架安装 / 环境复现 | ⬜ 未开始（等开机） | — |
| **合规红线复核** | ⚠️ **新发现，见 §4.3** | `docs/competition/rules-s2-track2.md:81` |

**当前的两个阻塞**：

1. 人为动作：把 PDF 上传到 flagos.io 的算力申请入口（§3.1）。
2. 实例关机：远程机没开机，任何实机工作都开不了（§3.2）。

---

## 2. 时间线（今天 2026-10-20）

| 节点 | 日期 | 说明 |
|---|---|---|
| 算力申请开放 | 09-14 ~ **11-10** | 先到先得，审核 5 个工作日 |
| 开发提交窗口 | 09-21 ~ **11-20 23:59** | 提交物为 `参赛队伍名.zip` |
| 评审 | 12 月初 | — |

原计划 **10-21 前投出**算力申请，才能在 11-20 前留出两轮完整实验的时间。
今天已是 10-20，**这条线明天必须走掉**。

> 注：赛程日期来自 `docs/competition/raw/race_539vlt2p.json`。若平台页面有更新，
> `python scripts/fetch_race.py --diff` 会以 exit 1 报警。

### 2.1 算力预算（已测算，详见 `docs/compute-budget.md`）

用户持有约 **1000 元**算力券（模力方舟）：64 GB 整卡 **4 元/h**、16 GB 切片 **1.5 元/h**。

| 项 | 结论 |
|---|---|
| 需要几台 | **2 台**：1 × 64 GB 整卡（测量机）+ 1 × 16 GB 切片（开发机） |
| 机器时间 | 64 GB **90 h** + 16 GB **40 h** |
| 费用 | **420 元**（占 42%），余 580 元作缓冲 |
| 日历 | 10-20 → 11-20，关键分叉点 **11-03**（M2 假设证实/证伪） |

**显存硬约束（由 `config.json` 算出，非估算）**：KV 开销 42.0 KiB/token，权重 4.69 GiB。
标称容量按 mx-smi 的 MiB 口径折算，并用实测切片（16000 MiB → torch 15.22 GiB）校准。

| 档位 | 0.85 额度 | 4k 需 17.82 GiB | 16k 需 49.32 GiB |
|---|---|---|---|
| 16 GB | 12.94 GiB | **否**（并发须降到 32） | **否** |
| 32 GB | 25.87 GiB | 是 | **否**（并发 32 就超） |
| 64 GB | 51.75 GiB | 是 | 是（余量 2.43 GiB / 4.7%） |

三条必须记住的结论：

1. **现有 16 GB 切片跑不了任何官方场景**，只能做算子功能验证与 kernel 微基准。
2. **32 GB 是陷阱** —— 能跑 4k 但跑不了 16k，而 16k 正是差距最大（7.68×）的场景。
3. **64 GB 跑 16k 只剩约 2.4 GiB 余量（额度 4.7%）** —— 任何增加显存的改动都会 OOM，
   每次改动前先算一遍峰值。这条要进 M0 之后的改动检查清单。

单次全场景测量（4k + 16k）约 12–15 分钟 ≈ **0.84–0.97 元**；成本不在跑，在**改代码时的空转**。
64 GB 空转一晚 = 96 元 = 预算的 23%。**每次会话结束立刻关机。**

---

## 3. 立即要做的三件事

### 3.1 投递算力申请（人为动作，只有你能做）

1. 登录 <https://flagos.io>，进入赛季二 · 赛道二（rid `539vlt2p`）的算力申请入口。
2. 上传 `proposals/算力申请-Casprian.pdf`。
3. 平台要求文件名形如 `算力申请-[队名].pdf` —— 当前队名 **Casprian**，文件名已匹配。

投递后把回执/工单号记到本文 §8 的变更日志里。

> PDF 含手机号与邮箱，已被 `.gitignore` 排除，**不在仓库里**。
> 若本机文件丢失，重新生成即可：
>
> ```powershell
> python scripts/build_proposal.py     # → proposals/算力申请-Casprian.pdf
> ```
>
> 前提是 `proposals/team.yaml` 与 `proposals/01-initial-technical-plan.md` 仍在（两者已入库）。

### 3.2 开机后第一件事：跑 preflight

用户重新开机后，**一条命令同时回答三个问题**（是不是整卡 / 显存够不够跑官方场景 / 框架装了没）：

```powershell
wsl bash -lc "cd '/mnt/d/Desktop/minicpm5‑2b‑infer‑opt‑flagos' && python3 scripts/remote/run_m0.py preflight"
```

它输出 remote identity、mx-smi 前 25 行、`classify()` 的判定 JSON、五个包的 import 结果，
最后给一句 verdict：`torch usable ~X GiB -> official scenarios that fit: 4k, 16k / NONE`。

判定依据见 `docs/remote-server.md` §2.1：`Vram Quota 0/64000 MiB` 且 `Compute 100%` 才是整卡；
`0/16000 MiB` / `Compute 25%` 仍是切片。

* 整卡 → 装栈（§3.4）→ 进 §4 的 M0。
* 仍是切片 → 只能做 §5 的「切片可做项」，同时向平台申请整卡。

### 3.3 验证本机工具链仍然可用

```powershell
cd 'D:\Desktop\minicpm5‑2b‑infer‑opt‑flagos'
python scripts/score.py --demo          # 期望 self-check passed: 9/9
python scripts/build_proposal.py --check # 期望 exit 0
```

### 3.4 装栈（只在开机后做一次）

远端机**没有** `vllm` / `FlagGems` / `vllm-plugin-FL`，必须自己装：

```powershell
# 上传（脚本很小，base64 分块几秒传完）
wsl bash -lc "python3 '/mnt/d/Desktop/minicpm5‑2b‑infer‑opt‑flagos/scripts/remote/sshx.py' put '/mnt/d/Desktop/minicpm5‑2b‑infer‑opt‑flagos/scripts/remote/install_stack.sh' /data/work/install_stack.sh"
wsl bash -lc "python3 '/mnt/d/Desktop/minicpm5‑2b‑infer‑opt‑flagos/scripts/remote/sshx.py' put '/mnt/d/Desktop/minicpm5‑2b‑infer‑opt‑flagos/scripts/remote/probe_env.sh' /data/work/probe_env.sh"

# 后台跑（约 20–40 分钟，主要耗在 vllm 的 C++ 编译）
wsl bash -lc "python3 '.../sshx.py' exec 'cd /data/work && mkdir -p logs && nohup bash install_stack.sh all > logs/install.log 2>&1 & echo started'"
wsl bash -lc "python3 '.../sshx.py' exec 'tail -40 /data/work/logs/install.log'"
```

脚本分阶段幂等（`base|vllm|flaggems|plugin|verify|all`），失败可单阶段重跑。
**关键约束：`flaggems` 必须先于 `plugin` 装** —— FlagGems 要 `setuptools<77`，
plugin 要 `setuptools>=77.0.3`，顺序反了会互相覆盖。脚本已内置这个顺序。
**绝不让 pip 动 torch**：FlagGems 的 metax extra 只认 MACA 3.7.2.0/3.8.1.0，
与本机 3.7.1.5 不匹配，装了就毁掉 `2.8.0+metax3.7.1.3`。脚本里已不装任何 extra。

---

## 4. 拿到整卡后的执行顺序（对应技术方案 §4.1）

```
M0 环境复现（3 日）
  └─ 两款芯片各复现一次官方 baseline，吞吐与 TTFT 相对偏差 ≤1%
       ↓ 门禁：python scripts/score.py --verify-baseline <csv> --vendor <tianshu|metax>
M1 归因确认（2 日）
  └─ profiler 分离 prefill / decode 的 kernel 占比，确认或修正 §2.4 的结论
       ↓
M2 P0 打通（5 日）
  └─ 见 §4.3 / §4.4：优化 FlagGems attention kernel 本身，而不是「把开关打开」
       ↓ 验收：16k tok/s 提升 ≥30%，math_500 ≥0.95，TTFT ≤ 基线×1.01
M3 P1 autotune（5 日）→ M4 P2 融合算子（5 日）→ M5 P3 收尾（3 日）
       ↓
M6 报告与提交（11-18 ~ 11-20）
```

### 4.1 M0 用一条命令跑完

```powershell
# 全流程：preflight → serve → 等健康 → benchmark → 关机前 stop → 取回 CSV → 本地门禁
wsl bash -lc "cd '/mnt/d/Desktop/minicpm5‑2b‑infer‑opt‑flagos' && python3 scripts/remote/run_m0.py all"
```

分步版（排障时用）：`preflight` / `serve` / `status` / `stop` / `bench` / `fetch` / `verify`。

* `serve` 用的是官方参数（`--port 9031 --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072`），
  并强制 `MACA_PATH=/opt/maca`、`VLLM_PLUGINS=fl`。
* `bench` 用官方 test-cases `[[4096,1024,64,256],[16384,1024,64,128]]`。
  **耗时说明**：脚本内 `RUNS=4`、`SKIP_FIRST=1`（`benchmark_throughput_serve.py:47,50`），
  每个 case 跑 4 次丢第 1 次 → 2 个 case 共 **8 次 `vllm bench serve`**，12–15 分钟。
* benchmark 的 CSV 目录是**相对路径** `benchmark_results`（`:438`），所以必须先 `cd /workspace`。

### 4.2 M0 为什么必须先做

评分是**相对提升**，分母就是官方 baseline。若实验机本身偏差 20%（共享卡、降频、
`VLLM_PLUGINS` 没设、漏了天数的 `--compilation-config`、框架 commit 不符），
之后每一次「提升」都是错误分母的产物。`score.py --verify-baseline` 就是这道门禁。

### 4.3 ⚠️ 合规红线（2026-10-20 晚复核，**优先级最高**）

`docs/competition/rules-s2-track2.md` 的三条直接约束：

| 行 | 原文 | 对我们的影响 |
|---|---|---|
| 76 | 允许的手段须是「**需要在框架有开发量**」的：显存管理、算子融合、内核重写、自研量化 | 提交物里必须有我们自己写的代码 |
| 78 | 「要求 `vllm serve` 的参数和基线一致」 | **不能**靠调 `--gpu-memory-utilization` / `--max-num-seqs` / `--enforce-eager` 拿分 |
| **81** | 「**禁止没有算子优化的情况下切换算子**，删掉框架主要算子选择逻辑等方式来进行模型性能优化」 | **原 M2 表述直接撞线** |

**问题**：此前 M2 被写成「把天数的 attention 从 `TRITON_ATTN` 切到 FlagGems FA2」。
这句话的动作就是**切换算子**，而它本身**不包含任何算子优化**。按 81 行，这是被禁止的手段，
技术报告里这样写会被判「使用禁止的优化策略」→ 成绩无效。

**改写的方向**（M2 仍然是方案赌注，只是价值来源要换）：

* 优化对象必须是 **FlagGems 的 attention kernel 本体**：`src/flag_gems/ops/attention.py:1284 flash_attn_varlen_func`
  在沐曦/天数上的 tile 尺寸、`num_warps`/`num_stages`、GQA 分组映射、split-KV 切分、
  mask 分支、以及 decode 路径的 `reshape_and_cache_flash`。
* 产出必须是 **FlagGems 侧的代码改动**（提交物里 `FlagGems/` 目录就是干这个的），
  技术报告要能指着某几行说「我们把这个 kernel 的 X 改成了 Y，原因是 Z」。
* 「切换」只能作为**接入手段**存在（把优化后的 kernel 接进服务），不能当作卖点本身。
* 天数侧：`iluvatar.yaml:30-35` 的 `attention_backend` 顺序是 `[flagos, vendor:iluvatar, reference]`，
  而 `flaggems.py:179-187` 里 `VLLM_FL_USE_FLAGGEMS_ATTN` 默认 `"0"` → 落到 `TRITON_ATTN`。
* 沐曦侧：`metax.yaml:28-31` 的 `attention_backend` **只有** `[vendor:metax]`，
  即 FlagGems FA2 根本没进候选，要接进来必须改这个 yaml —— 但只有在 kernel 真的被我们优化过之后才成立。

> 这一条要在写报告前再核一遍原文。规则里 76 行的「泛化能力高」也在暗示：
> 只在某一款芯片上 work 的 trick 分数不高。

### 4.4 M2 的核心假设（证据已二次核实）

天数 BI-V150 与沐曦 C500 的 attention 后端不同：

| | 天数 BI-V150 | 沐曦 C500 |
|---|---|---|
| 分发配置 | `iluvatar.yaml:30-35` → `[flagos, vendor:iluvatar, reference]` | `metax.yaml:28-31` → `[vendor:metax]` |
| 实际后端 | **vLLM 通用 `TRITON_ATTN`** | **MACA 原生 `FLASH_ATTN`** |

依据（均已在本仓库源码里核对过行号）：

* `vllm_fl/dispatch/backends/flaggems/flaggems.py:179-187` —— `VLLM_FL_USE_FLAGGEMS_ATTN` 默认 `"0"`，走 `TRITON_ATTN`
* `vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py:462-486` —— 注释原文 `# flash_attn is not available on iluvatar.`
* `vllm_fl/platform.py:276` —— `call_op("attention_backend", use_mla=..., use_sparse=...)` 是唯一入口
* `vllm_fl/dispatch/backends/flaggems/impl/attention.py`（857 行，类 `AttentionFLBackend`）**接入代码已写好**
* FlagGems v5.3.5 已有完整 FA2：`src/flag_gems/ops/attention.py:1284 flash_attn_varlen_func`

**该假设尚未被实机证实。** 现有证据只有：基线数据的算术自洽（`docs/competition/baselines.md`）
+ 源码静态阅读。M1 第一件事就是拿 profiler 证伪或证实它。
若被证伪，转向 P1 / P2 的框架层收益，不要在错误方向上继续投入。

---

## 5. 切片（16 GB）状态下能做的五件事

即使拿不到整卡，以下工作仍可推进。**但注意：16 GB 跑不了任何官方测试场景**
（4k 需 17.82 GiB > 12.67 GiB 额度），以下全部只产出「对不对」而非「快不快」：

1. **框架可安装性验证** —— 用 `install_stack.sh` 装完，确认五个包能 import、`entry_points` 里有 `fl`。
2. **算子功能验证** —— 用 ≤4k 短序列跑 `VLLM_FL_USE_FLAGGEMS_ATTN=1`，
   先确认 FlagGems FA2 在沐曦上数值正确（这一步是**功能验证**，不是性能优化，不构成 §4.3 的违规）。
3. **kernel 微基准** —— `flash_attn_varlen_func` / `reshape_and_cache_flash`
   单独计时，对照 MACA 原生 `flash_attn 2.6.3`。**这是 M2 真正需要的数据**，
   切片上就能做，不必等整卡。
4. **`fused_add_rms_norm` 原型** —— P2 的融合算子只需单卡小 batch 即可验证数值与带宽收益。
5. **profiler 工具链打通** —— `/opt/mcProfiler-ubuntu18.04` 的采集与解析。

---

## 6. 环境与踩坑（省你几个小时）

### 6.1 远程机（沐曦 C500）

完整说明见 [`docs/remote-server.md`](docs/remote-server.md)，这里只列最关键的：

* 网关 `140.207.205.81:32222`，**只能密码认证**，`scp`/`sftp`/`paramiko` 全部不可用。
* 用 `scripts/remote/sshx.py`（WSL 内 `python3`，依赖 `pexpect`），支持 `exec` / `put` / `get`。
* **凭据不入库**：读 `~/.ssh/sshx.json` 或 `SSHX_*` 环境变量。
  Windows 侧副本 `C:\Users\34511\.ssh\sshx.json`；另一份说明在
  `D:\Desktop\gitee服务器使用工具包.md` §1.1（该文件在仓库外）。
* **必设** `export MACA_PATH=/opt/maca`，否则 `import triton` 直接报
  `TypeError: expected str, bytes or os.PathLike object, not NoneType`。
* 实例规格（2026-10-20 实测）：1× C500 **sGPU 切片**，`Compute 25%`，
  `Vram Quota 0/16000 MiB`（torch 报 15.22 GiB）；128 核 / 32 GB / `/data` 93 GB 可用。
* 软件栈：`/opt/conda/bin/python` 3.12.11、torch `2.8.0+metax3.7.1.3`、
  triton `3.0.0+metax3.7.1.3`、MACA 3.7.1.5、驱动 3.8.30。
  **未装** `vllm` / `FlagGems` / `vllm-plugin-FL`。
* 远端网络：pypi 与阿里云镜像可达（pip 已配 `mirrors.aliyun.com`）、modelscope 可达；
  `huggingface.co` 超时。`github.com` 探测结果存疑，**装框架前先复核**，
  必要时用 gitee 镜像或本地打包上传。
* 本机（Windows）到 pypi 已验证可达：`https://pypi.org/pypi/vllm/0.20.2/json` → 200，
  `https://mirrors.aliyun.com/pypi/simple/vllm/` → 200。所以 vLLM 0.20.2 的 sdist 走 pip 下载没问题。

### 6.2 本机（Windows）

* 工作目录 `D:\Desktop\minicpm5‑2b‑infer‑opt‑flagos`，连字符是 **U+2011**，
  不是 ASCII `-`。手打会「找不到路径」。
* 用 Python 改文件后**必须把 CRLF 归一为 LF**，否则 `git diff` 会把整文件算作重写
  （曾出现 1820 行的假 diff）。归一脚本思路：`p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))`。
* `git commit -m "多行中文带引号"` 会被解析成 pathspec 报错。
  **正确做法**：消息写进文件，`git commit -F <file>`。
* PowerShell 里 `$out = python ... | Select-Object -First N` 会把 `$LASTEXITCODE`
  污染成 0。**先整体捕获，再截断。**
* 用 Python 覆盖文件时，若断言失败会先把目标截断为 0 字节（`score.py` 曾因此变空）。
  **先算好内容再写临时文件 + `os.replace`。**

### 6.3 Git / GitHub

* `github.com:443` 在本机不通，故 `origin` 走 **SSH**：
  `git@github.com:SOLECA794/minicpm5-2b-infer-opt-flagos.git`
* 已设 `core.sshCommand` 带 `-o ControlMaster=no -o ControlPath=none`，
  否则报 `getsockname failed: Not a socket`。
* `git fetch` 偶发 `Recv failure: Connection was reset`，重试即可。

### 6.4 PDF 流水线

`scripts/build_proposal.py` 把 `proposals/01-initial-technical-plan.md` 与
`proposals/team.yaml` 合成 PDF。已知约束：

* `team.yaml` 有任何 `TODO` 时**拒绝生成**（`--check` 可单独校验）。
* 分页用 reportlab 原生 `keepWithNext=1`（加在 `h1`–`h4` 样式上），
  **不要**用 `KeepTogether` 包标题 —— `KeepTogether` 是 `_ContainerSpace` 子类，
  会让 `_ktAllow()` 返回 False，**静默禁用整条标题粘合链**。
* 短表格也不要包 `KeepTogether`，靠 `repeatRows=1` 让表头在续页重复。
* 验收指标：`overflow 0.00 pt`、`kinsoku violations` 仅代码块内的 1–2 处、逐页目视。

### 6.5 本次新增的三个远端脚本

| 脚本 | 作用 | 关键点 |
|---|---|---|
| `scripts/remote/run_m0.py` | M0 全流程编排 | 子命令 `preflight/serve/status/stop/bench/fetch/verify/all`；官方参数只写一次 |
| `scripts/remote/probe_env.sh` | 一次性环境+网络侦察 | 永远 `exit 0`，可反复跑；段落化输出 |
| `scripts/remote/install_stack.sh` | 推理栈安装 | 分阶段幂等；`flaggems` 必须先于 `plugin`；不碰 torch |

两个踩过的坑：

* `--compilation-config` 的 JSON **必须带引号**，否则 bash 会拆成两个参数。已在 `run_m0.py` 里写成
  `--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'`，并用 `bash -n` 验过。
* `benchmark_throughput_serve.py` 的输出目录是相对路径，**必须先 `cd /workspace`**，
  否则 CSV 落在别处、`fetch` 找不到。

---

## 7. 关键数字（避免重新翻资料）

官方 baseline（来自赛事页截图，本地存 `docs/competition/*_baseline.png`）：

| 场景 | 天数 tok/s | 天数 TTFT | 沐曦 tok/s | 沐曦 TTFT |
|---|---:|---:|---:|---:|
| 4k `[4096,1024,64,256]` | 2028.01 | 11573.53 ms | 5089.65 | 3199.44 ms |
| 16k `[16384,1024,64,128]` | 915.15 | **599262.03 ms** | 7029.68 | 27197.14 ms |

护栏：精度 `math_500` Level 3 `acc ≥ 0.95`（基线 0.962）；`Mean TTFT ≤ 基线 × 1.01`。

评分：`总分 = 性能分(天数) + 性能分(沐曦) + 创新分`，
`性能分 = 100 × 提升比例 × 0.70`，`创新分 = 专家评分 × 0.30`，单场景提升 <1% 不计分。

算力申请：天数 BI-V150 **17** 卡日 + 沐曦 C500 **6** 卡日 = **23** 卡日（含约 30% 冗余）。

---

## 8. 变更日志

| 日期 | 事件 |
|---|---|
| 2026-10-19 | 服务器接入打通（`sshx.py`）；实测规格入库 `docs/remote-server.md` |
| 2026-10-19 | 技术方案改写为正式书面体；队名改 `Casprian`；生成 PDF（10 页 / 0.49 MB） |
| 2026-10-20 | 推送 `b9ebc95`；更新 `D:\Desktop\gitee服务器使用工具包.md`（新增 §1.1 多实例、§3.1 sshx.py、§6.15、§8 GPU 规格） |
| 2026-10-20 | **服务器关机**（用户操作）；写本文 |
| 2026-10-20 | 算力预算测算入库 `docs/compute-budget.md`：2 台 / 90+40 h / 420 元；证明 16 GB 与 32 GB 均无法跑官方场景 |
| 2026-10-20 | 提交 `6421748`：新增 `run_m0.py` / `probe_env.sh` / `install_stack.sh`；三者均在本机做过语法与单元校验 |
| 2026-10-20 | **合规复核**：`rules-s2-track2.md:81` 禁止「没有算子优化就切换算子」→ M2 表述改写为「优化 FlagGems attention kernel 本体」，见 §4.3 |

**待填**：

| 日期 | 事件 |
|---|---|
| | 算力申请已投递（工单号：______） |
| | 服务器重新开机，实例规格：______（整卡 64 GB / 切片 16 GB） |
| | M0 baseline 复现结果：天数 ______，沐曦 ______ |
