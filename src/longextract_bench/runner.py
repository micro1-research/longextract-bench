"""Fan out the providers over the dataset and write one envelope per (doc, provider).

Resolves the corpus via `dataset.py` (downloads + caches the HF snapshot), then runs
the chosen providers on every document in parallel at the requested concurrency.
Each failed attempt is re-tried up to `retries` times (default 3, set with
`--retries`). Resumable: a (doc, provider) pair is skipped when its output already
exists — including a persisted failure marker (use --force to redo, or delete the
marker to re-attempt just one). Each provider writes the `{result, _meta:{latency_s,
usage}}` envelope the grader reads.

Outputs go to a gitignored `runs/<run-dir>/<slug>/<provider>.json` tree (the HF
snapshot is read-only); the layout is defined in `dataset.run_output_path`.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import dataset as DS
from .providers import PROVIDERS


def write_failure_marker(out: Path, prov: str, err: str, attempts: int) -> None:
    """Persist a failed attempt as a result envelope so the resume logic skips it on
    re-runs and the grader scores it as the documented real 0. NOT written for
    timeouts — those leave no file so the next run can resume (sidecar) or re-attempt.
    Shape mirrors the provider envelope (null result + _meta) so the grader/scorer
    read it uniformly."""
    import json
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "result": None,
            "_meta": {
                "provider": prov,
                "status": "failed",
                "error": err[:500],
                "attempts": attempts,
                "latency_s": None,
                "usage": None,
            },
        }, indent=2))
    except OSError:
        pass  # best-effort; a missing marker just means the pair is re-attempted later


def run_one(
    doc: Path, prov: str, run_dir: Path, timeout: int, retries: int
) -> tuple[str, str, str, float]:
    slug = DS.slug_of(doc)
    out = DS.run_output_path(run_dir, slug, prov)
    schema = DS.schema_path(doc)
    pdf = DS.pdf_path(doc)
    # Reuse the current interpreter; invoke the provider as a package module so it
    # stays standalone-runnable (`python -m longextract_bench.providers.<prov>`).
    cmd = [
        sys.executable, "-m", f"longextract_bench.providers.{prov}",
        "--pdf", str(pdf), "--schema", str(schema), "--out", str(out),
    ]
    env = os.environ.copy()
    # Stagger starts so we don't open many TLS connections in the same instant.
    time.sleep(random.uniform(0, 8))
    last = ""
    timed_out = False
    attempts_made = 0
    for attempt in range(1, retries + 1):
        attempts_made = attempt
        t0 = time.time()
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
            dt = time.time() - t0
            if r.returncode == 0 and out.exists():
                return (slug, prov, "ok" if attempt == 1 else f"ok(retry {attempt})", dt)
            full_err = r.stderr.strip()
            last = f"rc={r.returncode}: {full_err[-140:]}"
            timed_out = False
        except subprocess.TimeoutExpired:
            # The job keeps running server-side; a retry resumes it via the sidecar
            # (poll, no re-submit), so retrying a timeout does not re-bill generation.
            last = "TIMEOUT"
            timed_out = True
        if attempt < retries:
            time.sleep(min(45, 5 * 2 ** attempt) + random.uniform(0, 5))
    # Retries re-run the provider, which RESUMES the persistent job (poll, no re-submit).
    # A non-timeout failure that exhausted its retries → persist a marker so re-runs skip
    # it and the grader counts the documented 0. A final timeout leaves no file so the next
    # run can resume/re-attempt the still-pending server-side job.
    if not timed_out:
        write_failure_marker(out, prov, last, attempts_made)
    return (slug, prov, f"FAIL after {attempts_made}x: {last}", 0.0)


def run_benchmark(
    *,
    run_dir: Path,
    providers: list[str],
    repo_id: str | None = None,
    revision: str | None = None,
    concurrency: int = 8,
    retries: int = 3,
    timeout: int = 3600,
    force: bool = False,
) -> None:
    """Resolve the dataset, build the (doc, provider) job list once (skipping existing
    outputs unless --force), and fan out at the requested concurrency."""
    bad = [p for p in providers if p not in PROVIDERS]
    if bad:
        raise SystemExit(f"unknown providers: {bad} (valid: {list(PROVIDERS)})")

    root = DS.dataset_root(repo_id=repo_id, revision=revision)
    docs = DS.doc_dirs(root)
    run_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (d, p) for d in docs for p in providers
        if force or not DS.run_output_path(run_dir, DS.slug_of(d), p).exists()
    ]
    skipped = len(docs) * len(providers) - len(jobs)

    def log(m: str) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    log(f"START run_dir={run_dir} docs={len(docs)} providers={providers} "
        f"jobs={len(jobs)} skipped(existing)={skipped} conc={concurrency} retries={retries}")
    ok = fail = 0
    by_prov: dict[str, list[int]] = {p: [0, 0] for p in providers}  # [ok, fail]
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = {
            ex.submit(run_one, d, p, run_dir, timeout, retries): (d, p)
            for d, p in jobs
        }
        for i, f in enumerate(as_completed(futs), 1):
            name, prov, status, dt = f.result()
            if status.startswith("ok"):
                ok += 1; by_prov[prov][0] += 1
            else:
                fail += 1; by_prov[prov][1] += 1
            log(f"[{i}/{len(jobs)}] {prov:13} {name[:44]:44} {status[:60]} ({dt:.0f}s)")
    log(f"DONE ok={ok} fail={fail} skipped={skipped}")
    log("per-provider: " + " | ".join(f"{p} {v[0]}ok/{v[1]}fail" for p, v in by_prov.items()))
