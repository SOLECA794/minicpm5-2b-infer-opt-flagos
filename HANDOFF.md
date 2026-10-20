# HANDOFF — 交接说明

> 最后更新：2026-10-20 14:00 (UTC+8)
> 仓库：`origin/main`（`git log -1` 取最新提交；本文随该分支推送）
> 一句话状态：**材料已就绪，卡在「向平台投递算力申请」这一步；代码侧无未提交改动。**

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
| **算力申请 PDF** | ✅ **已生成，待投递** | `proposals/算力申请-Casprian.pdf`（10 页 / 0.49 MB，gitignored） |
| 远程实验机接入 | ⏸ 工具已验证，**实例已关机** | `scripts/remote/sshx.py`、`docs/remote-server.md` |
| 框架安装 / 环境复现 | ⬜ 未开始 | — |

**当前唯一的硬阻塞是人为动作**：把 PDF 上传到 flagos.io 的算力申请入口。技术侧没有阻塞。

---

## 2. 时间线（今天 2026-10-20）

| 节点 | 日期 | 说明 |
|---|---|---|
| 算力申请开放 | 09-14 ~ **11-10** | 先到先得，审核 5 个工作日 |
| 开发提交窗口 | 09-21 ~ **11-20 23:59** | 提交物为 `参赛队伍名.zip` |
| 评审 | 12 月初 | — |

原计划 **10-21 前投出**算力申请，才能在 11-20 前留出两轮完整实验的时间。
今天已是 10-20，**这条线今天/明天必须走掉**。

> 注：赛程日期来自 `docs/competition/raw/race_539vlt2p.json`。若平台页面有更新，
> `python scripts/fetch_race.py --diff` 会以 exit 1 报警。

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

### 3.2 重新开机并确认实例规格

用户已关机。重新开机后第一件事是确认这次拿到的是**整卡 64 GB** 还是仍是 16 GB 切片：

```powershell
wsl bash -lc "python3 '/mnt/d/Desktop/minicpm5‑2b‑infer‑opt‑flagos/scripts/remote/sshx.py' exec 'mx-smi'"
```

判定依据见 `docs/remote-server.md` §2.1：`Vram Quota` 若是 `0/64000 MiB` 且
`Compute 100%`，才是整卡；若是 `0/16000 MiB` / `Compute 25%`，仍是切片。

* 整卡 → 直接进 §4 的 M0。
* 仍是切片 → 只能做 §5 的「切片可做项」，同时向平台申请整卡。

### 3.3 验证本机工具链仍然可用

```powershell
cd 'D:\Desktop\minicpm5‑2b‑infer‑opt‑flagos'
python scripts/score.py --demo          # 期望 self-check passed: 9/9
python scripts/build_proposal.py --check # 期望 exit 0
```

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
  └─ 天数侧 attention 从 TRITON_ATTN 切到 FlagGems FA2（AttentionFLBackend）
       ↓ 验收：16k tok/s 提升 ≥30%，math_500 ≥0.95，TTFT ≤ 基线×1.01
M3 P1 autotune（5 日）→ M4 P2 融合算子（5 日）→ M5 P3 收尾（3 日）
       ↓
M6 报告与提交（11-18 ~ 11-20）
```

### M0 为什么必须先做

评分是**相对提升**，分母就是官方 baseline。若实验机本身偏差 20%（共享卡、降频、
`VLLM_PLUGINS` 没设、漏了天数的 `--compilation-config`、框架 commit 不符），
之后每一次「提升」都是错误分母的产物。`score.py --verify-baseline` 就是这道门禁。

### M2 的核心假设（整个方案的赌注）

天数 BI-V150 与沐曦 C500 的 attention 后端不同：

| | 天数 BI-V150 | 沐曦 C500 |
|---|---|---|
| 分发配置 | `iluvatar.yaml`：`[flagos, vendor:iluvatar, reference]` | `metax.yaml`：`[vendor:metax]` |
| 实际后端 | **vLLM 通用 `TRITON_ATTN`** | **MACA 原生 `FLASH_ATTN`** |

依据：
* `vllm_fl/dispatch/backends/flaggems/flaggems.py:179-187` —— `VLLM_FL_USE_FLAGGEMS_ATTN` 默认 `"0"`，走 `TRITON_ATTN`
* `vllm_fl/dispatch/backends/vendor/iluvatar/iluvatar.py:480-486` —— 注释原文 `# flash_attn is not available on iluvatar.`
* 而 `vllm_fl/dispatch/backends/flaggems/impl/attention.py`（857 行，类 `AttentionFLBackend`）**已经写好了接入代码**
* FlagGems v5.3.5 已有完整 FA2：`src/flag_gems/ops/attention.py:1284 flash_attn_varlen_func`

即：**组件全都在，只是默认值把它关掉了。**

**该假设尚未被实机证实。** 现有证据只有：基线数据的算术自洽（`docs/competition/baselines.md`）
+ 源码静态阅读。M1 阶段的第一件事就是拿 profiler 证伪或证实它。
若被证伪，转向 P1 / P2 的框架层收益，不要在错误方向上继续投入。

---

## 5. 切片（16 GB）状态下能做的五件事

即使拿不到整卡，以下工作仍可推进，且都不依赖 16k 场景：

1. **框架可安装性验证** —— 装 `vllm`（沐曦适配版）、`FlagGems v5.3.5`、
   `vllm-plugin-FL@flagos-2026-s2`，确认能 import、能起服务。
2. **算子功能验证** —— 用 ≤4k 短序列跑 `VLLM_FL_USE_FLAGGEMS_ATTN=1`，
   先确认 FlagGems FA2 在沐曦上数值正确。
3. **kernel 微基准** —— `flash_attn_varlen_func` / `reshape_and_cache_flash`
   单独计时，对照 MACA 原生 `flash_attn 2.6.3`。
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

### 6.2 本机（Windows）

* 工作目录 `D:\Desktop\minicpm5‑2b‑infer‑opt‑flagos`，连字符是 **U+2011**，
  不是 ASCII `-`。手打会「找不到路径」。
* `write` / `edit` 工具在该目录报 `SetFileSecurityW EACCES (Win32 5)`。
  **绕法**：先写到 `C:\Users\34511\AppData\Local\Temp\`，再 `Copy-Item` 覆盖。
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

**待填**：

| 日期 | 事件 |
|---|---|
| | 算力申请已投递（工单号：______） |
| | 服务器重新开机，实例规格：______（整卡 64 GB / 切片 16 GB） |
| | M0 baseline 复现结果：天数 ______，沐曦 ______ |
