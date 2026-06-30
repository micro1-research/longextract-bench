"""Console entry points: lxbench-run / lxbench-grade / lxbench-report.

Generate extractions, then grade and report a run:

    lxbench-run    --run-dir runs/latest
    lxbench-grade  --run-dir runs/latest
    lxbench-report --run-dir runs/latest
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .providers import PROVIDERS


def _load_env() -> None:
    """Load a local `.env` into the environment if python-dotenv is installed.
    Optional — providers also read keys straight from the environment."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def run_main(argv: list[str] | None = None) -> None:
    """Fan the providers out over the dataset -> runs/<run-dir>/<slug>/<provider>.json."""
    _load_env()
    from .runner import run_benchmark

    ap = argparse.ArgumentParser(description="Run providers over the LongExtractBench corpus")
    ap.add_argument("--run-dir", required=True, type=Path, help="output dir for this run")
    ap.add_argument("--providers", default=",".join(PROVIDERS),
                    help=f"comma-separated subset (default: all of {','.join(PROVIDERS)})")
    ap.add_argument("--concurrency", "-j", type=int, default=8)
    ap.add_argument("--retries", type=int, default=3,
                    help="re-attempts per job; re-runs RESUME the persistent job (no re-submit)")
    ap.add_argument("--timeout", type=int, default=3600, help="per-job timeout (s)")
    ap.add_argument("--force", action="store_true", help="re-run even if an output exists")
    ap.add_argument("--repo-id", default=None, help="override HF dataset repo id")
    ap.add_argument("--revision", default=None, help="override HF dataset revision/SHA")
    args = ap.parse_args(argv)

    provs = [p.strip() for p in args.providers.split(",") if p.strip()]
    run_benchmark(
        run_dir=args.run_dir,
        providers=provs,
        repo_id=args.repo_id,
        revision=args.revision,
        concurrency=args.concurrency,
        retries=args.retries,
        timeout=args.timeout,
        force=args.force,
    )


def grade_main(argv: list[str] | None = None) -> None:
    """Classify each run success/failure + grade the successful ones -> runs/<run-dir>/scores.json."""
    _load_env()
    from .score import main as score_main
    score_main(argv)


def report_main(argv: list[str] | None = None) -> None:
    """Aggregate scores.json -> leaderboard + failures + latency blocks."""
    from .report import main as report_main_
    report_main_(argv)
