#!/usr/bin/env python3
"""Score calculator + baseline guard for FlagOS S2 Track 2.

Implements the official scoring formula from the competition page:

    总分 = 性能分(天数) + 性能分(沐曦) + 创新分
    性能分 = 100 * 提升比例 * 0.70
    创新分 = 专家评分 * 0.30

and the two hard guards:

    精度:  math_500 Level 3  acc >= 0.95      (baseline 0.962)
    TTFT:  mean_ttft <= baseline_ttft * 1.01

A scenario only scores if its improvement exceeds 1% (sub-1% is noise).

Usage
-----
    python scripts/score.py                     # show baseline table + scoring rules
    python scripts/score.py --demo              # run the built-in self-check
    python scripts/score.py --before b.csv --after a.csv
    python scripts/score.py --before b.csv --after a.csv --acc 0.958 --expert 8.5

CSV format: whatever `benchmark_throughput_serve.py` writes to
`benchmark_results/summary_*.csv` (columns: Prefill, Decode, Conc, Num Prompts,
Benchmark Duration (s), Output tok/s, Total tok/s, Mean TTFT (ms), ...).
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------
# Official baselines (2026-10-19, from the competition page screenshots).
# key: (prefill_len, decode_len, concurrency, num_prompts)
# --------------------------------------------------------------------------
SCENARIOS = {
    "4k": (4096, 1024, 64, 256),
    "16k": (16384, 1024, 64, 128),
}

BASELINE = {
    "tianshu": {
        "4k": {"total_tok_s": 2028.01, "ttft_ms": 11573.53, "output_tok_s": 405.6},
        "16k": {"total_tok_s": 915.15, "ttft_ms": 599262.03, "output_tok_s": 53.83},
    },
    "metax": {
        "4k": {"total_tok_s": 5089.645, "ttft_ms": 3199.435, "output_tok_s": 1017.93},
        "16k": {"total_tok_s": 7029.675, "ttft_ms": 27197.135, "output_tok_s": 413.51},
    },
}

ACC_BASELINE = 0.962
ACC_FLOOR = 0.95
TTFT_SLACK = 1.01          # TTFT must stay within +1% of baseline
NOISE_FLOOR = 0.01         # <1% improvement is treated as noise, scores 0
PERF_WEIGHT = 0.70
INNOV_WEIGHT = 0.30


@dataclass
class Verdict:
    vendor: str
    scenario: str
    before: float
    after: float
    ttft_before: float
    ttft_after: float

    @property
    def improvement(self) -> float:
        return (self.after - self.before) / self.before

    @property
    def ttft_ratio(self) -> float:
        return self.ttft_after / self.ttft_before

    @property
    def ttft_ok(self) -> bool:
        return self.ttft_ratio <= TTFT_SLACK

    @property
    def counts(self) -> bool:
        """Sub-1% improvements do not score."""
        return self.improvement > NOISE_FLOOR

    @property
    def perf_score(self) -> float:
        if not self.counts or not self.ttft_ok:
            return 0.0
        return 100.0 * self.improvement * PERF_WEIGHT


def read_summary(path: Path) -> dict[str, dict[str, float]]:
    """Parse a benchmark summary CSV into {scenario: {total_tok_s, ttft_ms, ...}}."""
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            key = (
                int(float(row["Prefill"])),
                int(float(row["Decode"])),
                int(float(row["Conc"])),
                int(float(row["Num Prompts"])),
            )
            name = next((n for n, s in SCENARIOS.items() if s == key), None)
            if name is None:
                continue
            out[name] = {
                "total_tok_s": float(row["Total tok/s"]),
                "ttft_ms": float(row["Mean TTFT (ms)"]),
                "output_tok_s": float(row.get("Output tok/s") or 0.0),
            }
    return out


def print_baselines() -> None:
    print("Official baselines — FlagOS S2 Track 2 (MiniCPM5-2B)\n")
    hdr = f"{'vendor':<9} {'scenario':<9} {'total tok/s':>12} {'TTFT (ms)':>13} {'output tok/s':>13}"
    print(hdr)
    print("-" * len(hdr))
    for vendor, scens in BASELINE.items():
        for scen, v in scens.items():
            print(f"{vendor:<9} {scen:<9} {v['total_tok_s']:>12.2f} {v['ttft_ms']:>13.2f} {v['output_tok_s']:>13.2f}")
    print()
    print("Guards")
    print(f"  accuracy  >= {ACC_FLOOR}   (baseline {ACC_BASELINE})")
    print(f"  TTFT      <= baseline x {TTFT_SLACK}")
    print(f"  noise     improvement must exceed {NOISE_FLOOR:.0%}")
    print()
    print("Scoring")
    print(f"  perf score  = 100 * improvement * {PERF_WEIGHT}")
    print(f"  innov score = expert_score(0-100) * {INNOV_WEIGHT}")
    print( "  total       = perf(tianshu) + perf(metax) + innov")
    print( "  per-vendor perf = mean(perf(4k), perf(16k))")


def evaluate(before: dict, after: dict, acc: float | None, expert: float | None) -> int:
    verdicts: list[Verdict] = []
    for vendor in ("tianshu", "metax"):
        for scen in ("4k", "16k"):
            b = before.get(vendor, {}).get(scen)
            a = after.get(vendor, {}).get(scen)
            if not b or not a:
                print(f"  ! missing data for {vendor}/{scen}, skipped")
                continue
            verdicts.append(
                Verdict(
                    vendor=vendor,
                    scenario=scen,
                    before=b["total_tok_s"],
                    after=a["total_tok_s"],
                    ttft_before=b["ttft_ms"],
                    ttft_after=a["ttft_ms"],
                )
            )

    print(f"\n{'vendor':<9} {'scen':<5} {'before':>10} {'after':>10} {'delta':>9} "
          f"{'TTFT x':>8} {'ttft':<6} {'score':>7}")
    print("-" * 74)
    for v in verdicts:
        flag = "OK" if v.ttft_ok else "FAIL"
        print(f"{v.vendor:<9} {v.scenario:<5} {v.before:>10.1f} {v.after:>10.1f} "
              f"{v.improvement:>+8.1%} {v.ttft_ratio:>8.3f} {flag:<6} {v.perf_score:>7.2f}")

    print()
    total = 0.0
    for vendor in ("tianshu", "metax"):
        scores = [v.perf_score for v in verdicts if v.vendor == vendor]
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        total += mean
        print(f"perf score [{vendor:<8}] = {mean:6.2f}")

    if expert is not None:
        innov = expert * INNOV_WEIGHT
        total += innov
        print(f"innov score [expert={expert:.1f}] = {innov:6.2f}")

    print("-" * 30)
    print(f"TOTAL = {total:.2f}")

    rc = 0
    if acc is not None:
        ok = acc >= ACC_FLOOR
        print(f"\naccuracy {acc:.3f} vs floor {ACC_FLOOR}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            rc = 2
    bad_ttft = [f"{v.vendor}/{v.scenario}" for v in verdicts if not v.ttft_ok]
    if bad_ttft:
        print(f"TTFT guard violated: {', '.join(bad_ttft)} -> result would be DISQUALIFIED")
        rc = 3
    return rc


def _selfcheck() -> int:
    """Sanity-check the arithmetic against the example in the official rules."""
    # Rules example: baseline 200 -> 210 is +5%, perf score = 100*5%*70% = 3.5
    v = Verdict("tianshu", "4k", 200.0, 210.0, 100.0, 100.0)
    assert abs(v.improvement - 0.05) < 1e-12, v.improvement
    assert abs(v.perf_score - 3.5) < 1e-9, v.perf_score
    assert v.ttft_ok

    # Sub-1% is noise -> 0
    assert Verdict("tianshu", "4k", 1000.0, 1005.0, 1.0, 1.0).perf_score == 0.0

    # TTFT +2% violates the guard
    assert not Verdict("tianshu", "4k", 1000.0, 1200.0, 100.0, 102.0).ttft_ok

    # Exactly +1% TTFT is still allowed (boundary is inclusive)
    assert Verdict("tianshu", "4k", 1000.0, 1200.0, 100.0, 101.0).ttft_ok

    # Doubling throughput on the official tianshu 16k baseline:
    # +100% improvement -> 100 * 1.00 * 0.70 = 70.0
    v2 = Verdict("tianshu", "16k", 915.15, 1830.30, 599262.03, 500000.0)
    assert abs(v2.improvement - 1.0) < 1e-9, v2.improvement
    assert abs(v2.perf_score - 70.0) < 1e-6, v2.perf_score

    print("self-check passed: 6/6 assertions")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", type=Path, help="baseline summary CSV")
    ap.add_argument("--after", type=Path, help="optimized summary CSV")
    ap.add_argument("--vendor", choices=["tianshu", "metax"], default="tianshu",
                    help="which vendor's baseline to compare against (default tianshu)")
    ap.add_argument("--acc", type=float, help="measured math_500 Level3 accuracy")
    ap.add_argument("--expert", type=float, help="expected expert score 0-100 for the report")
    ap.add_argument("--demo", action="store_true", help="run the self-check")
    args = ap.parse_args(argv)

    if args.demo or not (args.before and args.after):
        print_baselines()
        if args.demo:
            print()
            return _selfcheck()
        return 0

    base = {args.vendor: read_summary(args.before)}
    new = {args.vendor: read_summary(args.after)}
    if not base[args.vendor]:
        print(f"error: no matching scenarios in {args.before}", file=sys.stderr)
        return 1
    if not new[args.vendor]:
        print(f"error: no matching scenarios in {args.after}", file=sys.stderr)
        return 1
    return evaluate(base, new, args.acc, args.expert)


if __name__ == "__main__":
    raise SystemExit(main())
