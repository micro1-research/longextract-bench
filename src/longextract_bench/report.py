"""Aggregate `<run-dir>/scores.json` into a leaderboard plus a failures block and a
latency block.

Accuracy is computed ONLY over successful runs, so each system is scored on the
documents it actually completed; the leaderboard reports how many runs each system
completed (its `successful` count) alongside the metrics, and the failures block
accounts for the rest. The reconciliation line proves success + failure == #docs.

Consumes the precomputed `scores.json` (it does not re-grade), so importing this module
has no side effects.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

from .providers import PROVIDERS


def _quantile(values: list[float], frac: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    return s[min(len(s) - 1, int(len(s) * frac))]


def aggregate(docs: dict) -> dict:
    """Per-provider aggregate over a scores.json `docs` map."""
    n_docs = len(docs)
    out: dict[str, dict] = {}
    for prov in PROVIDERS:
        states: Counter = Counter()
        fail_reasons: Counter = Counter()
        P = R = L = 0.0
        scored = 0
        lats: list[float] = []
        for _slug, entry in docs.items():
            rec = (entry.get("providers", {}) or {}).get(prov) or {}
            st = rec.get("state", "failure")
            states[st] += 1
            if st == "success":
                if isinstance(rec.get("prec"), (int, float)):
                    P += rec["prec"]; R += rec["rec"]; L += rec["leaf"]; scored += 1
                if isinstance(rec.get("lat"), (int, float)):
                    lats.append(float(rec["lat"]))
            else:
                fail_reasons[rec.get("reason") or "Other API error"] += 1
        p = P / scored if scored else 0.0
        r = R / scored if scored else 0.0
        leaf = L / scored if scored else 0.0
        out[prov] = {
            "n_docs": n_docs,
            "states": dict(states),
            "successful": states.get("success", 0),
            "scored": scored,
            "precision": p, "recall": r, "leaf": leaf,
            "fail_reasons": fail_reasons,
            "lat_n": len(lats),
            "lat_mean": statistics.mean(lats) if lats else 0.0,
            "lat_median": statistics.median(lats) if lats else 0.0,
            "lat_p95": _quantile(lats, 0.95),
            "lat_max": max(lats) if lats else 0.0,
        }
    return out


def render(rows: dict, n_docs: int) -> str:
    L: list[str] = []
    # Rank by leaf accuracy (overall field-level correctness over completed docs).
    order = sorted(PROVIDERS, key=lambda p: rows[p]["leaf"], reverse=True)

    L.append("=" * 78)
    L.append(f"LongExtractBench — {n_docs} documents")
    L.append("=" * 78)

    L.append("\n## 1. Leaderboard (accuracy over COMPLETED documents only)\n")
    L.append(f"{'system':<14}{'successful':>11}{'precision':>11}{'recall':>9}{'leaf':>8}")
    L.append("-" * 53)
    for p in order:
        r = rows[p]
        L.append(f"{p:<14}{r['successful']:>11}{r['precision']:>10.1%}{r['recall']:>9.1%}"
                  f"{r['leaf']:>7.1f}%")

    L.append("\n## 2. Failures (run did not produce a usable result)\n")
    L.append(f"{'system':<14}{'failures':>9}   reasons")
    L.append("-" * 60)
    for p in order:
        r = rows[p]
        nfail = r["states"].get("failure", 0)
        reasons = ", ".join(f"{v}× {k}" for k, v in r["fail_reasons"].most_common()) or "—"
        L.append(f"{p:<14}{nfail:>9}   {reasons}")

    L.append("\n## 3. Latency (over completed documents)\n")
    L.append(f"{'system':<14}{'n':>5}{'mean_s':>9}{'median_s':>10}{'p95_s':>9}{'max_s':>9}")
    L.append("-" * 55)
    for p in order:
        r = rows[p]
        L.append(f"{p:<14}{r['lat_n']:>5}{r['lat_mean']:>9.0f}{r['lat_median']:>10.0f}"
                 f"{r['lat_p95']:>9.0f}{r['lat_max']:>9.0f}")

    L.append("\n## Reconciliation (success + failure == #docs)\n")
    L.append(f"{'system':<14}{'success':>9}{'failure':>9}{'sum':>6}")
    L.append("-" * 38)
    for p in order:
        s = rows[p]["states"]
        success, fail = s.get("success", 0), s.get("failure", 0)
        total = success + fail
        flag = "" if total == n_docs else "  <-- MISMATCH"
        L.append(f"{p:<14}{success:>9}{fail:>9}{total:>6}{flag}")

    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Aggregate scores.json -> leaderboard + failures + latency")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--json", action="store_true", help="emit the aggregate as JSON instead of a table")
    args = ap.parse_args(argv)

    scores_path = args.run_dir / "scores.json"
    if not scores_path.exists():
        raise SystemExit(f"no scores.json in {args.run_dir} — run `lxbench-grade` first")
    docs = json.loads(scores_path.read_text()).get("docs", {})
    rows = aggregate(docs)
    if args.json:
        print(json.dumps({"n_docs": len(docs), "rows": rows}, indent=2, default=str))
    else:
        print(render(rows, len(docs)))


if __name__ == "__main__":
    main()
