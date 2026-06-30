# Methodology

LongExtractBench scores how well a system turns a long PDF into structured JSON that
conforms to a supplied JSON Schema, against a human-reviewed answer key. This document
defines the metrics and the success/failure split. The grader is deterministic and
dependency-free (`src/longextract_bench/grading.py`, standard library only).

---

## 1. The unit of evaluation

Each benchmark item is a folder:

```
<doc>/
├── document.pdf        # the input
├── schema.json         # the JSON Schema every system is asked to fill
└── ground_truth.json   # the human-reviewed answer key
```

A document's schema is typically **one or more large arrays of rows** (tables,
line-items, records) plus **a few document-level scalar fields** (a title, a date, a
total). The grader treats those two shapes differently, because "did you find the
rows?" and "is each value right?" are different questions.

---

## 2. The metrics (accuracy over completed documents)

These are computed **only over the documents a system actually completed.** A system is
never scored on a document it failed — those are accounted for in the failures block,
and the count of successful runs is reported alongside the metrics.

### Precision / recall — over array ROWS

For every array field in the schema, the predicted rows are paired against the
ground-truth rows (see §3), then:

```
recall    = matched_rows / ground_truth_rows     "did it find the rows?"
precision = matched_rows / predicted_rows         "did it return only real rows,
                                                    or pad with junk / duplicates?"
```

Document-level scalars are single values, not rows, so they never move precision or
recall.

### Leaf accuracy — over matched rows + document scalars

```
leaf_accuracy = 100 × correct_leaves / total_leaves
```

A *leaf* is an individual scalar value. The leaf pool is **every document-level scalar
field plus every cell inside a row that matched** (rows are paired in §3), counted
recursively: a scalar, a nested-object field, a scalar inside a nested array (compared as
a multiset), or an object inside a nested array (recursed element by element).

Two cases behave differently, by design:

* **Top-level array rows that do not match are *not* in the leaf pool.** A predicted row
  with no ground-truth partner (padding) or a ground-truth row the system never returned
  (omission) is charged to **precision** and **recall** — not to leaf accuracy. Leaf
  accuracy is computed only over the rows the system actually matched.
* **Inside a matched row, nested-array elements that don't line up *do* count as misses**
  (their leaves are added to the total but not the matches), so padding/omission *within*
  a matched row still costs accuracy.

So leaf accuracy answers "of the values on the rows it found, how many are exactly
right?" — which is why it is only meaningful read **alongside** precision and recall.

> A system can post high leaf accuracy on the handful of rows it matched while missing
> most of the table; **recall** is what exposes that. The leaderboard always shows all
> three together for exactly this reason — see §3 on key inference.

---

## 3. How rows are paired

To compare two row lists, the grader infers a matching **key from the data itself** —
no hand-picked keys, so there is no lever to bias the result. The key is inferred
*anchored on the ground truth*:

1. **Single primary key** — the most-unique ground-truth column that is a usable
   identifier (≥80% non-null and ≥50% unique). Wherever a real identifier exists, it is
   the most-unique field, so rows match on the true key.
2. **Composite key** — if no single column qualifies, greedily combine dimension
   columns (e.g. region + gender) until the combination is near-unique on the ground
   truth.
3. **No key** — if neither qualifies, all rows fall into a single bucket and are paired
   by content (below), not by position.

Rows are then bucketed by key (one bucket when there is no key) and paired **by content**
within each bucket: identical rows lock first, then the best-overlap residual is matched
greedily. Only when a bucket's residual is too large to pair greedily does it fall back
to positional order, purely to bound the cost — never as the primary alignment. Each
ground-truth row can be claimed once, so duplicate rows a system emits stay unmatched and
correctly lower its precision.

Because the key is anchored on the ground truth, a system that mis-extracts the key
simply produces unmatched rows (its own precision/recall hit) — it never forces a
positional mis-alignment that would manufacture false per-cell errors for a competitor.

---

## 4. Fairness normalization

Every value — in keys and in leaf comparisons — passes through one shared `canonical()`
normalizer, so purely cosmetic differences never count as errors for **any** system:
case, whitespace (including mid-word OCR spaces), thousands separators, `100.0` vs
`100`, `$`/`%` symbols, smart quotes, accents, leading zeros, and placeholder markers
(`..`, `-`, `n/a`). The same normalizer is applied to every system's output and to the
ground truth.

The grader scores against the ground truth *as given*; it does not judge whether the
ground truth itself is correct. Each system's schema-dialect adapter (the per-API
translation of the shared JSON Schema) is documented in
[`providers.md`](providers.md).

---

## 5. The success/failure split

For every (document, system) pair, the outcome is exactly one of two states:

| State | Meaning |
| --- | --- |
| **success** | The system produced a real, non-empty extraction, which the grader then scores. |
| **failure** | The run did not produce usable output: a persisted failure marker, an empty/null result, an unreadable envelope, or no output file at all. |

Accuracy metrics are computed only over `success` runs, and the report states how many
documents each system completed successfully. Everything else is a failure — there is
no third category. A model that cannot fit a document in its context window, a service
whose extractor cannot handle a schema this complex, a submit-time refusal, a timeout,
a throttle, or an internal error all mean the same thing for the benchmark: the run
produced no usable result.

The reconciliation invariant is asserted for every system:

```
success + failure == number of documents
```

If it ever fails to hold, the run is flagged rather than silently averaged. The
classification lives in one small function, `classify.classify`, which the scorer
applies to every output envelope.

---

## 6. Why the scores are point-in-time

The systems are stochastic and the upstream APIs change over time, so the absolute
numbers are a snapshot, not a constant. What stays fixed is the **methodology and the
grader**:

* the **grader is fully deterministic** — same inputs, same scores, every run;
* key inference is sorted/seed-independent, so grades do not vary across processes;
* the dataset can be **pinned to an exact commit** with `--revision`.

When publishing numbers, record the run date and the resolved model IDs (the runner
logs them). The grader produces the same scores from the same outputs every time; the
live systems do not.
