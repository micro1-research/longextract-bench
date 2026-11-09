"""Check that the grader scores the same extraction the same way however it is laid out.

For every document in the dataset this grades a few predictions built from the answer
key itself, so the right score is known without calling any provider:

  self      the answer key, unchanged                     -> P = R = 1, leaf = 100
  c30       one wrong value in 30% of the rows             -> leaf near the injected rate
  c30-shuf  the same rows in a random order                -> must equal c30
  c30-miss  the same rows with the first row left out      -> must stay near c30
  junk      every row replaced by unrelated values         -> nothing matches

Usage (grader and dataset only, no API keys):
  python scripts/grader_invariance.py [--data-dir DIR] [--only SUBSTRING]
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path

from longextract_bench import dataset as DS
from longextract_bench import grading as G


def corrupt(gt: dict, arrays: list[str], rate: float, seed: int) -> dict:
    rnd = random.Random(seed)
    pred = copy.deepcopy(gt)
    for a in arrays:
        for row in pred.get(a) or []:
            if isinstance(row, dict) and rnd.random() < rate:
                fields = [k for k, v in row.items() if not isinstance(v, (dict, list)) and v not in (None, "")]
                if fields:
                    row[rnd.choice(fields)] = f"zz-wrong-{rnd.random()}"
    return pred


def reshape(pred: dict, arrays: list[str], how: str, seed: int = 0) -> dict:
    out = copy.deepcopy(pred)
    for a in arrays:
        rows = out.get(a)
        if not isinstance(rows, list) or not rows:
            continue
        if how == "shuffle":
            random.Random(seed).shuffle(rows)
        elif how == "miss":
            del rows[0]
        elif how == "junk":
            out[a] = [{k: "zz-junk" for k in r} if isinstance(r, dict) else r for r in rows]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, help="local dataset folder (default: download/cache)")
    ap.add_argument("--only", help="grade only documents whose name contains this")
    args = ap.parse_args()
    root = args.data_dir or DS.dataset_root()
    print("| document | rows | self | c30 | c30-shuf | c30-miss | junk P/R | seconds |")
    print("|---|---|---|---|---|---|---|---|")
    total = 0.0
    # only the answer key and schema are needed, so PDFs may be absent
    docs = sorted(d for d in root.iterdir() if d.is_dir() and DS.schema_path(d) and DS.ground_truth_path(d))
    for doc in docs:
        if args.only and args.only not in doc.name:
            continue
        gt = G.load_ground_truth(doc)
        schema = json.loads(DS.schema_path(doc).read_text())
        arrays = G._arrays(schema)
        c30 = corrupt(gt, arrays, 0.3, seed=1)
        cases = {
            "self": gt,
            "c30": c30,
            "c30-shuf": reshape(c30, arrays, "shuffle", seed=2),
            "c30-miss": reshape(c30, arrays, "miss"),
            "junk": reshape(gt, arrays, "junk"),
        }
        t0 = time.perf_counter()
        r = {k: G.grade(p, gt, schema) for k, p in cases.items()}
        dt = time.perf_counter() - t0
        total += dt

        def cell(k: str, r: dict = r) -> str:
            return f"{r[k]['leaf_accuracy']:.2f} (R {r[k]['recall']:.3f})"

        rows = sum(len(gt.get(a) or []) for a in arrays)
        print(
            f"| {doc.name[:48]} | {rows} | {cell('self')} | {cell('c30')} | {cell('c30-shuf')} | "
            f"{cell('c30-miss')} | {r['junk']['precision']:.2f}/{r['junk']['recall']:.2f} | {dt:.1f} |",
            flush=True,
        )
    print(f"\ntotal grading time: {total:.1f}s")


if __name__ == "__main__":
    main()
