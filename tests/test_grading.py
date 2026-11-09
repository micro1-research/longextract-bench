"""Tests for the deterministic grader's row pairing.

The synthetic tables below are built so that `infer_row_key` finds no key (every
column has only a handful of values, and no combination of up to four columns is
near-unique), which exercises the keyless path, and so that the residual after exact
matching is large enough to go through the indexed pairing rather than the all-pairs scan.
"""

from __future__ import annotations

import copy
import random

from longextract_bench import grading as G
from longextract_bench.grading import canonical, grade, infer_row_key

SCHEMA = {"properties": {"rows": {"type": "array"}, "title": {"type": "string"}}}
FIELDS = ["region", "sex", "age", "outcome", "quarter", "count"]


def keyless_table(n: int, seed: int = 0) -> dict:
    rnd = random.Random(seed)
    rows = [
        {
            "region": rnd.choice(["north", "south", "east", "west"]),
            "sex": rnd.choice(["male", "female", "total"]),
            "age": rnd.choice(["0-17", "18-29", "30-49", "50+"]),
            "outcome": rnd.choice(["grant", "refuse", "withdrawn"]),
            "quarter": rnd.choice(["2022 Q1", "2022 Q2", "2022 Q3", "2022 Q4"]),
            "count": rnd.randint(1, 9),
        }
        for _ in range(n)
    ]
    return {"title": "Synthetic decisions", "rows": rows}


def corrupt_one_cell(doc: dict, rate: float, seed: int) -> tuple[dict, int]:
    """One wrong value in `rate` of the rows. Returns (prediction, cells changed)."""
    rnd = random.Random(seed)
    pred = copy.deepcopy(doc)
    changed = 0
    for r in pred["rows"]:
        if rnd.random() < rate:
            r[rnd.choice(FIELDS)] = f"wrong-{rnd.random()}"
            changed += 1
    return pred, changed


def test_synthetic_table_is_keyless():
    gt = keyless_table(1500)
    assert infer_row_key(gt["rows"], gt["rows"]) == []


def test_identical_prediction_scores_perfect():
    gt = keyless_table(1500)
    r = grade(copy.deepcopy(gt), gt, SCHEMA)
    assert (r["precision"], r["recall"], r["leaf_accuracy"]) == (1.0, 1.0, 100.0)


def test_keyless_scores_do_not_depend_on_row_order():
    gt = keyless_table(1500)
    pred, _ = corrupt_one_cell(gt, 0.3, seed=1)
    base = grade(pred, gt, SCHEMA)
    for seed in range(3):
        shuffled = copy.deepcopy(pred)
        random.Random(seed).shuffle(shuffled["rows"])
        assert grade(shuffled, gt, SCHEMA) == base
    reversed_gt = copy.deepcopy(gt)
    reversed_gt["rows"].reverse()
    assert grade(pred, reversed_gt, SCHEMA) == base


def test_keyless_one_missed_row_costs_one_row_not_a_shifted_table():
    gt = keyless_table(1500)
    pred, changed = corrupt_one_cell(gt, 0.3, seed=1)
    full = grade(pred, gt, SCHEMA)
    del pred["rows"][0]
    r = grade(pred, gt, SCHEMA)
    assert r["matched"] == 1499 and r["gt_rows"] == 1500
    # one wrong cell per corrupted row: leaf accuracy stays at the injected error rate
    # instead of collapsing because every later row is compared with its neighbour.
    # Pairing by content is greedy, so on this deliberately repetitive table it can sit
    # a hair under the true-pairing value; it must not be more than 0.05 points under.
    total = 1500 * len(FIELDS) + 1  # every cell plus the title
    ceiling = 100 * (total - changed) / total
    assert ceiling - 0.05 <= full["leaf_accuracy"] <= ceiling + 1e-9
    assert abs(r["leaf_accuracy"] - full["leaf_accuracy"]) < 0.05


def test_keyless_unrelated_rows_are_not_matches():
    gt = keyless_table(1500)
    junk = {"title": gt["title"], "rows": [{f: "zz" for f in FIELDS} for _ in gt["rows"]]}
    r = grade(junk, gt, SCHEMA)
    assert r["matched"] == 0
    assert r["precision"] == 0.0 and r["recall"] == 0.0


def test_keyless_half_junk_counts_half_the_rows():
    gt = keyless_table(1500)
    pred = copy.deepcopy(gt)
    for row in pred["rows"][750:]:
        for f in FIELDS:
            row[f] = "zz"
    r = grade(pred, gt, SCHEMA)
    assert r["matched"] == 750
    assert r["recall"] == 0.5 and r["precision"] == 0.5


def test_keyless_minimum_overlap_bar_small_table():
    gt = {"title": "t", "rows": [{"a": "x", "b": "y", "c": "z", "d": "w"}] * 3}
    # shares 2 of 4 fields with every GT row -> at the bar, matched
    at_bar = {"title": "t", "rows": [{"a": "x", "b": "y", "c": "q", "d": "q"}]}
    r = grade(at_bar, gt, SCHEMA)
    assert r["matched"] == 1
    # shares 1 of 4 -> below the bar, unmatched
    below = {"title": "t", "rows": [{"a": "x", "b": "q", "c": "q", "d": "q"}]}
    r = grade(below, gt, SCHEMA)
    assert r["matched"] == 0 and r["precision"] == 0.0


def test_keyed_tables_keep_their_matching():
    gt = {
        "title": "t",
        "rows": [{"id": f"r{i}", "v": i, "w": i % 3} for i in range(200)],
    }
    assert infer_row_key(gt["rows"], gt["rows"]) == ["id"]
    pred = copy.deepcopy(gt)
    pred["rows"][5]["v"] = -1  # wrong value on a matched row -> leaf miss, still matched
    pred["rows"][6]["id"] = "nope"  # wrong key -> unmatched
    r = grade(pred, gt, SCHEMA)
    assert r["matched"] == 199
    assert r["leaf_total"] == 199 * 3 + 1 and r["leaf_match"] == 199 * 3 + 1 - 1


def test_large_keyed_bucket_is_order_independent():
    # 250 unique ids plus 150 rows sharing id "dup": the key is still inferred, and the
    # "dup" bucket's residual exceeds the greedy cap, the case that used to fall back to
    # pairing by list position
    rnd = random.Random(3)
    rows = [{"id": f"r{i}", "a": rnd.randint(0, 50), "b": rnd.randint(0, 50)} for i in range(250)]
    rows += [{"id": "dup", "a": rnd.randint(0, 50), "b": rnd.randint(0, 50)} for _ in range(150)]
    gt = {"title": "t", "rows": rows}
    assert infer_row_key(gt["rows"], gt["rows"]) == ["id"]
    pred = copy.deepcopy(gt)
    for r in pred["rows"][250:]:
        r[rnd.choice("ab")] = -1
    base = grade(pred, gt, SCHEMA)
    assert base["matched"] == 400  # keyed bucket: same record by key, pair count kept
    for seed in range(3):
        shuffled = copy.deepcopy(pred)
        random.Random(seed).shuffle(shuffled["rows"])
        assert grade(shuffled, gt, SCHEMA) == base


def test_indexed_and_exhaustive_pairing_agree(monkeypatch):
    gt = keyless_table(600, seed=5)
    pred, _ = corrupt_one_cell(gt, 0.4, seed=6)
    random.Random(7).shuffle(pred["rows"])
    indexed = grade(pred, gt, SCHEMA)
    monkeypatch.setattr(G, "_EXHAUSTIVE_CAP", 10**9)
    exhaustive = grade(pred, gt, SCHEMA)
    assert indexed["matched"] == exhaustive["matched"]
    assert abs(indexed["leaf_accuracy"] - exhaustive["leaf_accuracy"]) < 0.05


def test_canonical_cache_keeps_types_apart():
    assert canonical(1) == "1"
    assert canonical(1.0) == "1"
    assert canonical(True) == "true"
    assert canonical(1) == "1"  # not poisoned by the bool entry (True == 1 hashes alike)
    assert canonical("1,000") == "1000"
    assert canonical(None) == "" and canonical([]) == "" and canonical({}) == ""
    assert canonical(["a"]) == canonical(["a"])
