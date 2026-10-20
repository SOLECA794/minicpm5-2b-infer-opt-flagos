#!/usr/bin/env bash
# Install the FlagOS S2 Track 2 inference stack on a MetaX C500 box.
#
# Recipe derived from upstream/vllm-plugin-FL/docker/metax/Dockerfile (the
# vendor's own CI image), adapted to what this box actually is:
#
#   * MACA 3.7.1.5 / torch 2.8.0+metax3.7.1.3 -- FlagGems v5.3.5 has no
#     matching optional-dependency group for that torch (only metax-maca3720
#     wants torch 2.8.0+metax3.7.2.0), so we install WITHOUT the [metax-*]
#     extras and never let pip touch torch.
#   * vLLM is installed in "empty mode" (VLLM_TARGET_DEVICE=empty): the MACA
#     kernels and torch come from the vendor image, only the Python package
#     comes from upstream vLLM.
#   * setuptools conflict: FlagGems pins setuptools<77 (PEP 639 metadata),
#     vllm-plugin-FL pins >=77.0.3. Hence the install ORDER matters:
#     flaggems first with setuptools<77, then bump to >=77 for the plugin.
#
# Every stage is idempotent and re-runnable. Long stages must be launched with
# nohup from the gateway (a single exec call has a hard timeout):
#
#   sshx.py exec 'cd /data/work && nohup bash install_stack.sh all > logs/install.log 2>&1 & echo started'
#   sshx.py exec 'tail -40 /data/work/logs/install.log'
#
# Stages: base | vllm | flaggems | plugin | verify | all
#
# Pins (competition-mandated):
#   vllm-plugin-FL  branch flagos-2026-s2
#   FlagGems        tag v5.3.5
#   vllm            0.20.2 (upstream, empty mode)

set -uo pipefail

WORK="${WORK:-/data/work}"
DIST="$WORK/dist"
SRC="$WORK/src"
LOG="$WORK/logs"
VLLM_VERSION="${VLLM_VERSION:-0.20.2}"
FLAGGEMS_TAG="${FLAGGEMS_TAG:-v5.3.5}"
PLUGIN_BRANCH="${PLUGIN_BRANCH:-flagos-2026-s2}"

mkdir -p "$WORK" "$DIST" "$SRC" "$LOG"

say() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

python_bin() {
  if [ -n "${PYTHON:-}" ]; then echo "$PYTHON"; return; fi
  if [ -x /opt/conda/bin/python ]; then echo /opt/conda/bin/python; return; fi
  command -v python3 || command -v python || die "no python"
}

PY="$(python_bin)"
pip_() { "$PY" -m pip "$@"; }

# ---------------------------------------------------------------- stages ----

stage_base() {
  say "base: build tooling, MACA env, workdirs"
  # MACA_PATH is required or 'import triton' dies with
  # "TypeError: expected str, bytes or os.PathLike object, not NoneType".
  export MACA_PATH="${MACA_PATH:-/opt/maca}"
  say "MACA_PATH=$MACA_PATH"

  pip_ install --upgrade --no-cache-dir \
    wheel "setuptools>=64,<77" "setuptools-scm>=8,<10" \
    ninja cmake pybind11 packaging || die "build tooling install failed"

  "$PY" - <<'PYEOF' || die "python sanity check failed"
import sys
print("python", sys.version.split()[0])
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
print("device", torch.cuda.get_device_name(0))
PYEOF
}

fetch() { # fetch <url> <out>
  local url="$1" out="$2"
  say "download $url"
  curl -fL --retry 3 --retry-all-errors --connect-timeout 20 --max-time 1800 \
    -o "$out" "$url" && return 0
  rm -f "$out"; return 1
}

stage_vllm() {
  say "vllm $VLLM_VERSION (empty mode)"
  export VLLM_TARGET_DEVICE=empty

  if "$PY" -c "import vllm, sys; sys.exit(0 if vllm.__version__.startswith('$VLLM_VERSION') else 1)" 2>/dev/null; then
    say "vllm already at $VLLM_VERSION -- skipping"
    return 0
  fi

  local sdist="$DIST/vllm-$VLLM_VERSION.tar.gz"
  if [ ! -f "$sdist" ]; then
    # The sdist is the route that survives a blocked github.com: it comes from
    # the pypi mirror and carries the full source tree (csrc included).
    say "pip download vllm sdist"
    pip_ download --no-deps --no-binary :all: "vllm==$VLLM_VERSION" -d "$DIST" \
      || fetch "https://github.com/vllm-project/vllm/archive/refs/tags/v$VLLM_VERSION.tar.gz" "$sdist" \
      || die "could not obtain vllm source (pypi sdist and github both failed)"
    ls "$DIST"/vllm-*.tar.gz >/dev/null 2>&1 || die "vllm sdist missing in $DIST"
    sdist="$(ls -1 "$DIST"/vllm-*.tar.gz | head -1)"
  fi

  rm -rf "$SRC/vllm"
  mkdir -p "$SRC/vllm"
  tar -xf "$sdist" -C "$SRC/vllm" --strip-components=1 || die "cannot unpack $sdist"

  # Remove any vendor-packaged vllm first, exactly like the upstream Dockerfile.
  pip_ uninstall -y vllm >/dev/null 2>&1 || true

  say "building vllm (the long one, 10-25 min)"
  ( cd "$SRC/vllm" && VLLM_TARGET_DEVICE=empty pip_ install --no-build-isolation --no-deps . ) \
    || die "vllm empty-mode install failed"
  "$PY" -c "import vllm; print('vllm', vllm.__version__)" || die "vllm import failed"
}

stage_flaggems() {
  say "FlagGems $FLAGGEMS_TAG (GEMS_VENDOR=metax)"
  if [ -n "${FLAGGEMS_TARBALL:-}" ]; then
    say "using pre-uploaded tarball $FLAGGEMS_TARBALL"
    cp -f "$FLAGGEMS_TARBALL" "$DIST/FlagGems.tar.gz"
  elif [ ! -f "$DIST/FlagGems.tar.gz" ]; then
    fetch "https://codeload.github.com/FlagOpen/FlagGems/tar.gz/refs/tags/$FLAGGEMS_TAG" \
      "$DIST/FlagGems.tar.gz" \
      || fetch "https://github.com/flagos-ai/FlagGems/archive/refs/tags/$FLAGGEMS_TAG.tar.gz" \
           "$DIST/FlagGems.tar.gz" \
      || die "FlagGems tarball download failed (upload one: sshx.py put FlagGems.tar.gz $DIST/)"
  fi

  rm -rf "$SRC/FlagGems"
  mkdir -p "$SRC/FlagGems"
  tar -xzf "$DIST/FlagGems.tar.gz" -C "$SRC/FlagGems" --strip-components=1 || die "cannot unpack FlagGems"

  # Build deps for FlagGems' own pyproject; --no-build-isolation means pip will
  # NOT fetch them, and setuptools>=77 breaks its metadata (see pyproject note).
  pip_ install --no-cache-dir "setuptools>=64,<77" "setuptools-scm>=8,<10" wheel==0.46.2 \
    || die "FlagGems build deps failed"

  ( cd "$SRC/FlagGems" && GEMS_VENDOR=metax pip_ install --no-build-isolation . ) \
    || die "FlagGems install failed"

  GEMS_VENDOR=metax "$PY" - <<'PYEOF' || die "FlagGems import failed"
import flag_gems
print("flag_gems:", flag_gems.__file__)
print("version:", getattr(flag_gems, "__version__", "unknown"))
PYEOF
}

stage_plugin() {
  say "vllm-plugin-FL @ $PLUGIN_BRANCH"
  if [ -n "${PLUGIN_SRC:-}" ]; then
    say "using pre-uploaded tree $PLUGIN_SRC"
    rm -rf "$SRC/vllm-plugin-FL"; cp -a "$PLUGIN_SRC" "$SRC/vllm-plugin-FL"
  elif [ ! -d "$SRC/vllm-plugin-FL/.git" ]; then
    say "clone from github"
    GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$PLUGIN_BRANCH" \
      https://github.com/flagos-ai/vllm-plugin-FL.git "$SRC/vllm-plugin-FL" \
      || die "clone failed (github blocked?). Upload a tree and set PLUGIN_SRC=..."
  fi

  # The plugin needs setuptools>=77.0.3 -- bump only AFTER FlagGems is built.
  pip_ install --no-cache-dir "setuptools>=77.0.3,<81" "setuptools-scm>=8" \
    scikit-build-core==0.11 pybind11 ninja cmake || die "plugin build deps failed"

  ( cd "$SRC/vllm-plugin-FL" && pip_ install --no-build-isolation . ) \
    || die "vllm-plugin-FL install failed"
}

stage_verify() {
  say "verify: imports, entry points, torch untouched"
  export MACA_PATH="${MACA_PATH:-/opt/maca}"
  "$PY" - <<'PYEOF'
import importlib
fail = []
for name in ("torch", "triton", "vllm", "flag_gems", "vllm_fl"):
    try:
        m = importlib.import_module(name)
        print(f"{name:10s} {getattr(m, '__version__', 'ok')}")
    except Exception as exc:
        fail.append(name)
        print(f"{name:10s} FAILED: {type(exc).__name__}: {exc}")

import torch
print("torch", torch.__version__, "(must stay 2.8.0+metax3.7.1.3)")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device mem: %.2f GiB" % (torch.cuda.get_device_properties(0).total_memory / 2**30))

from importlib.metadata import entry_points
eps = entry_points()
for group in ("vllm.platform_plugins", "vllm.general_plugins"):
    got = [e.name for e in (eps.select(group=group) if hasattr(eps, "select") else eps.get(group, []))]
    print(f"{group}: {got}")
    if "fl" not in got:
        fail.append(group)

raise SystemExit(1 if fail else 0)
PYEOF
  local rc=$?
  say "verify rc=$rc"
  return $rc
}

# ------------------------------------------------------------------ main ----

case "${1:-all}" in
  base)     stage_base ;;
  vllm)     stage_base; stage_vllm ;;
  flaggems) stage_base; stage_flaggems ;;
  plugin)   stage_base; stage_plugin ;;
  verify)   stage_verify ;;
  all)
    stage_base
    stage_vllm
    stage_flaggems
    stage_plugin
    stage_verify || say "VERIFY FAILED -- see above"
    ;;
  *) die "unknown stage '$1' (base|vllm|flaggems|plugin|verify|all)" ;;
esac
