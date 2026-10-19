#!/usr/bin/env python3
"""SSHPiper-friendly remote exec / transfer helper.

The experiment boxes sit behind an **SSHPiper** gateway at 140.207.205.81:32222.
The gateway routes by *username suffix* (`root+vm-XXXX`), kills the default
KEX, and stalls any scp/sftp data channel. This helper wraps the three things
that actually work: the command channel, and base64-over-argv for file moves.

Run it from WSL (`python3` there has pexpect; paramiko does NOT work against
this gateway -- auth is reset after KEX):

    wsl bash -lc "python3 /mnt/d/.../scripts/remote/sshx.py exec 'nvidia-smi'"

Credentials come from `~/.ssh/sshx.json` (see `sshx.example.json`) or from the
`SSHX_HOST` / `SSHX_PORT` / `SSHX_USER` / `SSHX_PASS` environment variables.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shlex
import sys
import time
from pathlib import Path

try:
    import pexpect
except ImportError:  # pragma: no cover - the message is the whole point
    sys.exit(
        "error: pexpect is missing. Run this from WSL, not Windows Python:\n"
        "  wsl bash -lc \"python3 scripts/remote/sshx.py ...\""
    )

CONFIG_PATH = Path.home() / ".ssh" / "sshx.json"

# Gateway quirks (see docs/remote-server.md):
#   - default curve25519 KEX is reset mid-handshake
#   - the host key is regenerated per container, so it must not be pinned
#   - password auth only; a pubkey attempt costs a round trip and can trip
#     the rate limiter
SSH_OPTS = [
    "-o", "KexAlgorithms=diffie-hellman-group14-sha256",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "PubkeyAuthentication=no",
    "-o", "PreferredAuthentications=password",
    "-o", "NumberOfPasswordPrompts=1",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=15",
    "-o", "ServerAliveCountMax=4",
    "-o", "ControlMaster=no",
    "-o", "ControlPath=none",
]

# A single argv element is capped at 131071 bytes by MAX_ARG_STRLEN, and the
# command prefix eats into that. 90 KiB of raw bytes -> ~120 KiB base64, safe.
CHUNK_RAW = 90 * 1024


class Config:
    def __init__(self, host: str, port: int, user: str, password: str):
        self.host = host
        self.port = port
        self.user = user
        self.password = password

    @property
    def target(self) -> str:
        return f"{self.user}@{self.host}"


def load_config(name: str | None = None) -> Config:
    env = {
        "host": os.environ.get("SSHX_HOST"),
        "port": os.environ.get("SSHX_PORT"),
        "user": os.environ.get("SSHX_USER"),
        "password": os.environ.get("SSHX_PASS"),
    }
    if env["host"] and env["user"] and env["password"]:
        return Config(env["host"], int(env["port"] or 32222), env["user"], env["password"])

    if not CONFIG_PATH.exists():
        sys.exit(
            f"error: no credentials. Create {CONFIG_PATH} or set SSHX_HOST/"
            "SSHX_PORT/SSHX_USER/SSHX_PASS."
        )
    data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if name:
        if name not in data.get("hosts", {}):
            sys.exit(f"error: host '{name}' not in {CONFIG_PATH}")
        entry = dict(data["hosts"][name])
        entry.setdefault("password", data.get("password", ""))
    else:
        entry = dict(data.get("default", {}))
        entry.setdefault("password", data.get("password", ""))
    if not entry.get("host") or not entry.get("user"):
        sys.exit(f"error: {CONFIG_PATH} is missing host/user")
    return Config(entry["host"], int(entry.get("port", 32222)), entry["user"], entry["password"])


def _spawn(cfg: Config, argv: list[str], timeout: float) -> pexpect.spawn:
    full = ["ssh", "-p", str(cfg.port), *SSH_OPTS, cfg.target, *argv]
    child = pexpect.spawn(full[0], full[1:], timeout=timeout, encoding="utf-8",
                          codec_errors="replace")
    idx = child.expect([r"[Pp]assword:", pexpect.EOF, pexpect.TIMEOUT], timeout=45)
    if idx == 0:
        child.sendline(cfg.password)
    elif idx == 2:
        child.close(force=True)
        raise TimeoutError("timed out waiting for the password prompt (gateway rate limit?)")
    return child


def run(cfg: Config, command: str, timeout: float = 120.0) -> tuple[int, str]:
    """Run one remote command through the gateway's command channel."""
    child = _spawn(cfg, [command], timeout)
    child.expect(pexpect.EOF, timeout=timeout)
    output = child.before or ""
    child.close()
    rc = child.exitstatus if child.exitstatus is not None else 255
    # The gateway echoes a banner on some containers; strip a trailing CR noise.
    return rc, output.replace("\r\n", "\n").replace("\r", "\n")


def put(cfg: Config, local: Path, remote: str, timeout: float = 300.0) -> None:
    """Upload via base64-over-argv, chunked. scp is not usable here."""
    raw = local.read_bytes()
    digest = hashlib.md5(raw).hexdigest()
    tmp = f"/tmp/.sshx_{digest}.b64"
    run(cfg, f"rm -f {tmp} && mkdir -p $(dirname {shlex.quote(remote)})", timeout=60)
    total = (len(raw) + CHUNK_RAW - 1) // CHUNK_RAW
    for i in range(0, len(raw), CHUNK_RAW):
        blob = base64.b64encode(raw[i:i + CHUNK_RAW]).decode("ascii")
        n = i // CHUNK_RAW + 1
        rc, out = _retry(cfg, f"printf %s {blob} | base64 -d >> {tmp} && echo CHUNK-OK", timeout)
        if rc != 0 or "CHUNK-OK" not in out:
            sys.exit(f"error: chunk {n}/{total} failed (rc={rc})\n{out[-500:]}")
        print(f"  chunk {n}/{total}", file=sys.stderr)
    rc, out = run(cfg, f"md5sum {tmp}", timeout=60)
    if digest not in out:
        sys.exit(f"error: checksum mismatch after upload: local {digest}, remote {out.strip()}")
    rc, out = run(cfg, f"mv {tmp} {shlex.quote(remote)} && ls -l {shlex.quote(remote)}", timeout=60)
    if rc != 0:
        sys.exit(f"error: could not install {remote}\n{out}")
    print(f"uploaded {local} -> {remote} ({len(raw)} bytes, md5 {digest})", file=sys.stderr)


def get(cfg: Config, remote: str, local: Path, timeout: float = 300.0) -> None:
    """Download via base64-over-stdout."""
    rc, out = run(cfg, f"base64 -w0 {shlex.quote(remote)}", timeout=timeout)
    if rc != 0 or not out.strip():
        sys.exit(f"error: could not read {remote} (rc={rc})\n{out[-500:]}")
    data = base64.b64decode("".join(out.split()))
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_bytes(data)
    print(f"downloaded {remote} -> {local} ({len(data)} bytes)", file=sys.stderr)


def _retry(cfg: Config, command: str, timeout: float, attempts: int = 5) -> tuple[int, str]:
    """The gateway rate-limits bursts of connections; back off and retry."""
    last: tuple[int, str] = (255, "")
    for i in range(attempts):
        try:
            rc, out = run(cfg, command, timeout)
            if rc == 0:
                return rc, out
            last = (rc, out)
        except (TimeoutError, pexpect.ExceptionPexpect) as exc:
            last = (255, str(exc))
        time.sleep(1.0 + i)
    return last


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", help="named host from ~/.ssh/sshx.json")
    sub = ap.add_subparsers(dest="action", required=True)

    p_exec = sub.add_parser("exec", help="run a remote command")
    p_exec.add_argument("command")
    p_exec.add_argument("--timeout", type=float, default=120.0)

    p_put = sub.add_parser("put", help="upload a file (base64 chunked)")
    p_put.add_argument("local", type=Path)
    p_put.add_argument("remote")

    p_get = sub.add_parser("get", help="download a file")
    p_get.add_argument("remote")
    p_get.add_argument("local", type=Path)

    args = ap.parse_args(argv)
    cfg = load_config(args.host)

    if args.action == "exec":
        rc, out = run(cfg, args.command, args.timeout)
        sys.stdout.write(out)
        if out and not out.endswith("\n"):
            sys.stdout.write("\n")
        return rc
    if args.action == "put":
        put(cfg, args.local, args.remote)
        return 0
    if args.action == "get":
        get(cfg, args.remote, args.local)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
