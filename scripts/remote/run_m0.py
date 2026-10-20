#!/usr/bin/env python3
"""M0 orchestrator: boot the MetaX box -> official baseline CSV -> local gate.

Run from **WSL** (pexpect lives there), same as sshx.py:

    wsl bash -lc "cd /mnt/d/Desktop/minicpm5-2b-infer-opt-flagos && python3 scripts/remote/run_m0.py preflight"

Subcommands
-----------
    preflight   classify the GPU (whole card vs slice) + check the stack
    serve       start vllm serve with the official flags
    status      one-screen snapshot: /v1/models, mx-smi, tail of the server log
    stop        kill the server (and anything holding the port)
    bench       run the official benchmark, print the summary CSV path
    fetch       download the newest summary CSV into reports/
    verify      run scripts/score.py --verify-baseline on the fetched CSV
    all         preflight -> serve -> bench -> stop -> fetch -> verify

Everything is idempotent: serve refuses to double-start, stop is safe when
nothing runs, fetch always takes the newest CSV on the remote side.

Why this exists: the official benchmark is 2 cases x (RUNS=4, SKIP_FIRST=1) =
8 "vllm bench serve" invocations, i.e. 12-15 min of billed GPU time. Getting
the flags wrong (missing VLLM_PLUGINS, missing MACA_PATH, wrong
--compilation-config) silently produces a wrong *denominator*, and the scoring
is relative to that denominator. So the flags live here, once, exactly as the
rules state them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sshx  # noqa: E402  (same directory)

# --- official numbers (docs/competition/rules-s2-track2.md) -------------------
PORT = 9031
MODEL = "/workspace/MiniCPM5-2B"
SERVED_NAME = "minicpm"
GPU_UTIL = "0.85"
MAX_LEN = "131072"
TEST_CASES = "[[4096,1024,64,256],[16384,1024,64,128]]"
WORKSPACE = "/workspace"
BENCH_DIR = WORKSPACE + "/benchmark_results"
LOG = "/data/work/logs/serve.log"
PYTHON3 = "/opt/conda/bin/python3"

# the one and only difference between the two vendors
COMPILATION_CONFIG = {
    "tianshu": " --compilation-config '" + '{"cudagraph_mode": "FULL_DECODE_ONLY"}' + "'",
}

# VRAM the two official scenarios need (docs/compute-budget.md, GiB)
NEED_GIB = {"4k": 17.82, "16k": 49.32}

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def say(msg: str = "") -> None:
    print(msg, flush=True)


def remote(cmd: str, timeout: float = 120.0) -> tuple[int, str]:
    return sshx.run(sshx.load_config(), cmd, timeout)


def must(cmd: str, timeout: float = 120.0, what: str = "") -> str:
    rc, out = remote(cmd, timeout)
    if rc != 0:
        say("!! " + (what or cmd) + " failed (rc=" + str(rc) + ")")
        say(out[-2000:])
        raise SystemExit(1)
    return out


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------
def classify(out: str) -> dict:
    """Pull quota / compute / driver facts out of an mx-smi dump."""
    info: dict = {}
    m = re.search(r"Vram Quota\s*:?\s*([0-9]+)\s*/\s*([0-9]+)\s*MiB", out, re.I)
    if m:
        info["quota_used_mib"] = int(m.group(1))
        info["quota_total_mib"] = int(m.group(2))
    m = re.search(r"Compute\s*:?\s*([0-9]+)\s*%", out, re.I)
    if m:
        info["compute_pct"] = int(m.group(1))
    m = re.search(r"Driver Version\s*:?\s*([0-9][0-9A-Za-z.\-]*)", out, re.I)
    if m:
        info["driver"] = m.group(1)
    total = info.get("quota_total_mib")
    if total is not None:
        whole = total >= 60000 and info.get("compute_pct") == 100
        info["kind"] = "whole-card" if whole else "slice"
        # calibrated on the real slice: 16000 MiB quota -> torch sees 15.22 GiB
        info["quota_usable_gib"] = round(total * 0.95 / 1024.0, 2)
    return info


def torch_mem_gib(probe_out):
    """Read the device memory the torch probe printed, if any."""
    m = re.search(r"mem_gib\s+([0-9.]+)", probe_out)
    return float(m.group(1)) if m else None


PROBE = r'''import importlib, sys
print("python", sys.version.split()[0])
for m in ("torch", "triton", "vllm", "flag_gems", "vllm_fl"):
    try:
        mod = importlib.import_module(m)
        print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "MISSING", type(e).__name__)
try:
    import torch
    print("cuda_available", torch.cuda.is_available(), "count", torch.cuda.device_count())
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print("device", p.name, "mem_gib", round(p.total_memory / 2**30, 2))
except Exception as e:
    print("torch probe failed", type(e).__name__, e)
'''


def cmd_preflight(args: argparse.Namespace) -> int:
    say("== remote identity ==")
    say(must("hostname; date -Is; nproc; free -g | head -2", what="identity").strip())

    say()
    say("== mx-smi ==")
    rc, smi = remote("mx-smi")
    if rc != 0:
        say("mx-smi rc=" + str(rc) + " -- the box may still be powered off")
        say(smi[-800:])
        return 1
    say("\n".join(smi.strip().splitlines()[:25]))

    info = classify(smi)
    say()
    say("== classification ==")
    say(json.dumps(info, ensure_ascii=False, indent=2))

    say()
    say("== stack ==")
    rc, out = remote(
        "export MACA_PATH=/opt/maca; /opt/conda/bin/python -c " + shlex.quote(PROBE),
        timeout=240,
    )
    say(out.strip())

    say()
    say("== verdict ==")
    usable = torch_mem_gib(out) or info.get("quota_usable_gib")
    if usable is not None and info.get("kind") is None:
        # no sGPU quota line -> an unsliced card; name it by what torch reports
        info["kind"] = "whole-card" if usable >= 55 else "slice"
    say("instance: " + info.get("kind", "unknown"))
    if usable is not None:
        fits = [k for k in ("4k", "16k") if NEED_GIB[k] <= usable]
        say("torch usable ~" + str(usable) + " GiB -> official scenarios that fit: "
            + (", ".join(fits) or "NONE"))
        if not fits:
            say("=> slice: NO official scenario fits. Do install / operator work only (HANDOFF section 5).")
    return 0


# ---------------------------------------------------------------------------
# serve / status / stop
# ---------------------------------------------------------------------------
def serve_cmd(vendor: str) -> str:
    extra = COMPILATION_CONFIG.get(vendor, "")
    return (
        "cd " + WORKSPACE + " && "
        "export MACA_PATH=/opt/maca && export VLLM_PLUGINS=fl && "
        "mkdir -p " + os.path.dirname(LOG) + " && "
        "nohup vllm serve " + MODEL
        + " --port " + str(PORT)
        + " --served-model-name " + SERVED_NAME
        + " --gpu-memory-utilization " + GPU_UTIL
        + " --max-model-len " + MAX_LEN
        + extra
        + " > " + LOG + " 2>&1 & echo started_pid=$!"
    )


def healthy() -> bool:
    rc, out = remote("curl -s -m 3 http://127.0.0.1:" + str(PORT) + "/v1/models")
    return rc == 0 and SERVED_NAME in out


def cmd_serve(args: argparse.Namespace) -> int:
    rc, out = remote("curl -s -m 3 http://127.0.0.1:" + str(PORT) + "/v1/models")
    if rc == 0 and SERVED_NAME in out:
        say("server already healthy on :" + str(PORT) + " -- not restarting")
        return 0
    if rc == 0 and out.strip():
        say("port " + str(PORT) + " answers but not with " + SERVED_NAME + "; run stop first")
        return 1

    cmd = serve_cmd(args.vendor)
    say("$ " + cmd)
    must(cmd, what="start server")
    say("server launching; weight load takes minutes. Poll with: run_m0.py status")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    rc, out = remote("curl -s -m 5 http://127.0.0.1:" + str(PORT) + "/v1/models")
    say("== /v1/models ==")
    say(out.strip()[:600] if out.strip() else "(no answer, rc=" + str(rc) + ")")
    say()
    say("== mx-smi (memory) ==")
    _, smi = remote("mx-smi | head -20")
    say(smi.strip())
    say()
    say("== tail " + LOG + " ==")
    _, tail = remote("tail -25 " + LOG + " 2>/dev/null || echo '(no log yet)'")
    say(tail.strip())
    return 0 if healthy() else 1


def cmd_stop(args: argparse.Namespace) -> int:
    say(must(
        "pkill -f 'vllm serve' >/dev/null 2>&1; sleep 2; "
        "pkill -9 -f 'vllm serve' >/dev/null 2>&1; sleep 1; "
        "mx-smi | grep -i comput || true; echo stopped",
        what="stop server",
    ).strip())
    return 0


# ---------------------------------------------------------------------------
# bench / fetch / verify
# ---------------------------------------------------------------------------
def cmd_bench(args: argparse.Namespace) -> int:
    if not healthy():
        say("server is not healthy; run serve first")
        return 1
    cases = args.test_cases or TEST_CASES
    cmd = (
        "cd " + WORKSPACE + " && mkdir -p " + BENCH_DIR + " && "
        + "/opt/conda/bin/python3 " + WORKSPACE + "/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py "
        "--model " + MODEL + " --served-model-name " + SERVED_NAME
        + " --port " + str(PORT) + " --test-cases '" + cases + "'"
    )
    say("$ " + cmd)
    say("expect ~12-15 min (2 cases x RUNS=4, first run discarded)...")
    rc, out = remote(cmd, timeout=args.timeout)
    say(out[-4000:])
    if rc != 0:
        say("benchmark exited rc=" + str(rc))
        return rc
    found = re.findall(r"Summary CSV:\s*(\S+)", out)
    if found:
        say("remote summary CSV: " + found[-1])
        REPORTS.mkdir(parents=True, exist_ok=True)
        (REPORTS / "last_summary_path.txt").write_text(found[-1] + "\n", encoding="utf-8")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    hint = REPORTS / "last_summary_path.txt"
    remote_path = ""
    if hint.exists():
        remote_path = hint.read_text(encoding="utf-8").strip()
    if not remote_path:
        rc, out = remote("ls -t " + BENCH_DIR + "/summary_*.csv 2>/dev/null | head -1")
        if rc != 0 or not out.strip():
            say("no summary CSV on the remote box yet")
            return 1
        remote_path = out.strip().splitlines()[0]
    REPORTS.mkdir(parents=True, exist_ok=True)
    local = REPORTS / Path(remote_path).name
    sshx.get(sshx.load_config(), remote_path, local)
    say("local copy: " + str(local))
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    cands = sorted(REPORTS.glob("summary_*.csv"))
    if not cands:
        say("no local summary CSV; run fetch first")
        return 1
    csv_path = args.csv or str(cands[-1])
    cmd = [sys.executable, str(ROOT / "scripts" / "score.py"),
           "--verify-baseline", csv_path, "--vendor", args.vendor]
    say("$ " + " ".join(shlex.quote(c) for c in cmd))
    return os.spawnv(os.P_WAIT, sys.executable, cmd)


# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------
def cmd_all(args: argparse.Namespace) -> int:
    if cmd_preflight(args) != 0:
        return 1
    if cmd_serve(args) != 0:
        return 1
    say()
    say("waiting for the server to become healthy (up to " + str(args.wait) + "s)...")
    started = time.time()
    ok = False
    for _ in range(int(args.wait // 15) + 1):
        if healthy():
            say("healthy after " + str(int(time.time() - started)) + "s")
            ok = True
            break
        time.sleep(15)
    if not ok:
        say("server did not become healthy in time; run status to see why")
        return 1
    rc = cmd_bench(args)
    try:
        if not args.keep:
            cmd_stop(args)
    finally:
        cmd_fetch(args)
        if rc == 0:
            return cmd_verify(args)
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vendor", default="metax", choices=("metax", "tianshu"))
    sub = ap.add_subparsers(dest="action", required=True)

    sub.add_parser("preflight").set_defaults(func=cmd_preflight)

    for name, fn in (("serve", cmd_serve), ("status", cmd_status), ("stop", cmd_stop)):
        p = sub.add_parser(name)
        p.set_defaults(func=fn)

    p_bench = sub.add_parser("bench")
    p_bench.add_argument("--test-cases", help="JSON list; default is the official two")
    p_bench.add_argument("--timeout", type=float, default=2400.0)
    p_bench.set_defaults(func=cmd_bench)

    sub.add_parser("fetch").set_defaults(func=cmd_fetch)

    p_verify = sub.add_parser("verify")
    p_verify.add_argument("--csv")
    p_verify.set_defaults(func=cmd_verify)

    p_all = sub.add_parser("all")
    p_all.add_argument("--test-cases")
    p_all.add_argument("--timeout", type=float, default=2400.0)
    p_all.add_argument("--wait", type=float, default=900.0)
    p_all.add_argument("--keep", action="store_true", help="leave the server running")
    p_all.set_defaults(func=cmd_all)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
