"""Score a run: per (document, system) record success/failure and, for successful
runs, the deterministic grader's precision / recall / leaf accuracy.

For every provider x every document it records, in `<run-dir>/scores.json`:
  - state: success | failure
  - reason: captured error text for failures (uncategorized)
  - prec, rec  (0-1, from the grader) and leaf (0-100), for successful runs
  - lat (s)
Plus pages/fields per doc (from the Reducto envelope usage, when present).

Compute-once: each (doc, provider) is graded a single time and cached. Re-runs skip
anything already cached UNLESS the output envelope's mtime is newer than the cached
entry (so re-runs refresh automatically) or --force is given. Flushes after every
document, so an interruption never loses completed work.

Reconciliation: for each provider, success + failure == number of documents. The
invariant is asserted and surfaced in the run metadata.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from . import dataset as DS
from . import grading as G
from .classify import classify
from .providers import PROVIDERS


def score_run(
    run_dir: Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    force: bool = False,
) -> dict:
    """Grade everything in `run_dir` against the dataset, writing `<run-dir>/scores.json`."""
    cache_path = run_dir / "scores.json"
    cache: dict = {}
    if cache_path.exists() and not force:
        try:
            cache = json.loads(cache_path.read_text()).get("docs", {})
        except Exception:
            cache = {}

    root = DS.dataset_root(repo_id=repo_id, revision=revision)
    docs = DS.doc_dirs(root)
    t0 = time.time()
    n_graded = n_skip = 0

    def flush() -> None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"meta": {"updated": time.strftime("%Y-%m-%d %H:%M:%S"),
                      "providers": list(PROVIDERS)},
             "docs": cache}, indent=1))

    for i, doc in enumerate(docs, 1):
        slug = DS.slug_of(doc)
        gtp = DS.ground_truth_path(doc)
        schp = DS.schema_path(doc)
        if not (gtp and schp):
            continue
        entry = cache.setdefault(slug, {"pages": None, "fields": None, "providers": {}})
        gt = schema = None  # lazy-load only when first needed
        for prov in PROVIDERS:
            envp = DS.run_output_path(run_dir, slug, prov)
            cached = entry["providers"].get(prov)
            if not envp.exists():
                entry["providers"][prov] = {"state": "failure", "reason": "no output file", "mtime": 0}
                continue
            mtime = envp.stat().st_mtime
            if cached and cached.get("mtime") == mtime and not force:
                n_skip += 1
                continue
            try:
                env = json.loads(envp.read_text())
            except Exception:
                entry["providers"][prov] = {"state": "failure", "reason": "unreadable output file", "mtime": mtime}
                continue
            state, reason = classify(env)
            m = env.get("_meta") or {}
            rec = {"state": state, "reason": reason, "mtime": mtime,
                   "lat": (m.get("latency_s") if isinstance(m.get("latency_s"), (int, float)) and m.get("latency_s") > 0 else None)}
            # capture pages/fields from the reducto usage block when present
            if prov == "reducto":
                u = m.get("usage") or {}
                if isinstance(u.get("num_pages"), (int, float)):
                    entry["pages"] = u["num_pages"]
                if isinstance(u.get("num_fields"), (int, float)):
                    entry["fields"] = u["num_fields"]
            if state == "success":
                if gt is None:
                    gt = G._load_gt(gtp)
                    schema = json.loads(schp.read_text())
                try:
                    pred = G._load_doc(envp, prov)
                except Exception:
                    pred = None
                if pred is None:
                    rec["state"] = "failure"
                    rec["reason"] = "load-none"
                else:
                    g = G.grade(pred, gt, schema)
                    rec["prec"] = round(g["precision"], 6)
                    rec["rec"] = round(g["recall"], 6)
                    rec["leaf"] = round(g["leaf_accuracy"], 4)
                    n_graded += 1
            entry["providers"][prov] = rec
        flush()
        print(f"[{i}/{len(docs)}] {slug[:44]:44} graded_so_far={n_graded} "
              f"skipped={n_skip} ({time.time()-t0:.0f}s)", flush=True)

    reconcile(cache)
    print(f"DONE graded={n_graded} skipped={n_skip} elapsed={time.time()-t0:.0f}s -> {cache_path}",
          flush=True)
    return cache


def reconcile(docs: dict) -> dict[str, Counter]:
    """Assert success + failure == #docs for every provider."""
    n = len(docs)
    per_prov: dict[str, Counter] = {p: Counter() for p in PROVIDERS}
    for _slug, entry in docs.items():
        for prov in PROVIDERS:
            st = (entry.get("providers", {}).get(prov) or {}).get("state", "failure")
            per_prov[prov][st] += 1
    for prov, c in per_prov.items():
        total = c["success"] + c["failure"]
        assert total == n, (
            f"reconciliation failed for {prov}: success+failure="
            f"{total} != {n} docs ({dict(c)})"
        )
    return per_prov


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Grade a LongExtractBench run -> scores.json")
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--repo-id", default=None, help="override HF dataset repo id")
    ap.add_argument("--revision", default=None, help="override HF dataset revision/SHA")
    ap.add_argument("--force", action="store_true", help="re-grade everything")
    args = ap.parse_args(argv)
    score_run(args.run_dir, repo_id=args.repo_id, revision=args.revision, force=args.force)


if __name__ == "__main__":
    main()
