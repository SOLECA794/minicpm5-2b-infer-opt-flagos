#!/usr/bin/env bash
# Remote environment probe for the MetaX C500 experiment box.
#
# Run this FIRST on every fresh boot. It answers, in one shot, the four
# questions that decide what the session can do:
#   1. which instance did we get this time (full 64 GB card vs 16 GB sGPU slice)
#   2. what is already installed (vllm / flag_gems / vllm_fl)
#   3. what the network can reach (pypi mirror, gitee, github, modelscope)
#   4. how much room is on /data
#
# Usage:  bash /data/work/probe_env.sh
# Exit code is always 0: this is a report, not a gate.

set -u

hr() { printf '\n===== %s =====\n' "$1"; }

hr "identity"
hostname
date -Is
uname -srm
echo "container: $(cat /proc/1/cgroup 2>/dev/null | head -1)"

hr "cpu / mem"
echo "cores: $(nproc)"
awk '/MemTotal/ {printf "mem total: %.1f GiB\n", $2/1048576}' /proc/meminfo
awk '/MemAvailable/ {printf "mem avail: %.1f GiB\n", $2/1048576}' /proc/meminfo

hr "disk"
df -h / /data 2>/dev/null | sed 's/  */ /g'

hr "maca / gpu"
echo "MACA_PATH=${MACA_PATH:-<unset>}"
for p in /opt/maca /opt/maca-*; do [ -d "$p" ] && echo "found: $p"; done
MXSMI="$(command -v mx-smi || true)"
[ -z "$MXSMI" ] && [ -x /opt/maca/bin/mx-smi ] && MXSMI=/opt/maca/bin/mx-smi
if [ -n "$MXSMI" ]; then
  echo "mx-smi: $MXSMI"
  "$MXSMI" 2>&1 | head -40
else
  echo "mx-smi: NOT FOUND"
fi

hr "python / torch"
PY="$(command -v python || true)"
[ -z "$PY" ] && [ -x /opt/conda/bin/python ] && PY=/opt/conda/bin/python
echo "python: ${PY:-NOT FOUND}"
if [ -n "$PY" ]; then
  "$PY" - <<'PYEOF'
import importlib, os, sys
print("sys.version:", sys.version.split()[0], "|", sys.executable)
def show(name, attr="__version__"):
    try:
        m = importlib.import_module(name)
    except Exception as exc:
        print(f"{name}: MISSING ({type(exc).__name__}: {str(exc)[:120]})")
        return None
    print(f"{name}: {getattr(m, attr, 'unknown')}")
    return m
torch = show("torch")
show("triton")
show("vllm")
show("flag_gems")
show("vllm_fl")
if torch is not None:
    try:
        print("torch.cuda.is_available:", torch.cuda.is_available())
        print("torch.cuda.device_count:", torch.cuda.device_count())
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            print("device:", p.name, "| total_mem: %.2f GiB" % (p.total_memory / 2**30))
            print("capability:", torch.cuda.get_device_capability(0))
    except Exception as exc:
        print("torch.cuda probe failed:", type(exc).__name__, exc)
PYEOF
fi

hr "pip packages of interest"
"$PY" -m pip list 2>/dev/null | grep -Ei '^(vllm|flag|torch|triton|transformers|numpy|modelscope|openai|evalscope|flash)' || echo "(none)"

hr "env"
for v in VLLM_PLUGINS GEMS_VENDOR VLLM_USE_V1 VLLM_FL_USE_FLAGGEMS_ATTN LD_LIBRARY_PATH; do
  echo "$v=${$v:-<unset>}"
done

hr "network (8s timeout each)"
probe() {
  local url="$1" label="$2"
  local code
  code="$(curl -s -o /dev/null -w '%{http_code}' --connect-timeout 4 --max-time 8 "$url" 2>/dev/null || echo ERR)"
  printf '%-14s %-46s %s\n' "$label" "$url" "$code"
}
probe https://mirrors.aliyun.com/pypi/simple/ aliyun-pypi
probe https://pypi.org/simple/ pypi
probe https://gitee.com gitee
probe https://github.com github
probe https://codeload.github.com codeload
probe https://modelscope.cn modelscope
probe https://huggingface.co hf

hr "pip index config"
"$PY" -m pip config list 2>/dev/null || true
cat /etc/pip.conf 2>/dev/null || true
cat ~/.pip/pip.conf 2>/dev/null || true
cat ~/.config/pip/pip.conf 2>/dev/null || true

hr "workspace"
for d in /workspace /data /root; do
  [ -d "$d" ] && echo "$d: $(ls -A "$d" 2>/dev/null | head -8 | tr '\n' ' ')"
done

hr "profiler"
ls -d /opt/mcProfiler* 2>/dev/null || echo "no /opt/mcProfiler*"

echo
echo "probe done"
