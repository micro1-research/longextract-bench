# Dataset

LongExtractBench evaluates on **50 document folders**, hosted on Hugging Face and
downloaded + cached at runtime. Nothing is committed to git: the corpus lives at

```
micro1-inc/longextract-bench-50   (repo_type="dataset")
```

and the resolver in `src/longextract_bench/dataset.py` is the single place that knows
how to fetch it and where every file lives.

---

## Folder layout

Each of the 50 items is a folder containing exactly three files:

```
<slug>/
├── document.pdf        # the input PDF
├── schema.json         # the JSON Schema the system is asked to fill
└── ground_truth.json   # the human-reviewed answer key
```

The resolver accepts a couple of alternate filenames for the schema and ground truth
(`grading.SCHEMA_NAMES` / `grading.GT_NAMES`) for robustness, but `schema.json` and
`ground_truth.json` are canonical. Folder slugs are neutral identifiers — they carry no
hint about whether a document is "easy" or "hard."

---

## What the 50 documents represent

The 50 are a **curated representative subset** selected by objective strata so the
benchmark exercises the full range of difficulty, not just the easy middle:

* **Document length** — page-count buckets from short documents to long, multi-hundred-page
  reports, so context-window and page-cap behavior is exercised.
* **Schema complexity** — field-count and nesting buckets, from flat single-table schemas
  to deeply nested multi-array schemas, so schema-complexity ceilings are exercised.
* **Domain** — a spread across document domains (financial, scientific/technical,
  regulatory, tabular data, and others), so no single document style dominates.

A few documents are included specifically because they stress the **failure** paths
(very large inputs, very complex schemas) — that is by design, since how many documents
a system completes is a first-class result, not noise to be filtered out. The dataset
card on Hugging Face describes the *distribution* only; it carries **no per-document
annotations** about which systems succeed or fail on which document.

> Scores are computed over completed documents, with the failures block accounting for
> the rest — so a system is never quietly advantaged by a subset that happens to omit
> the documents it cannot handle. See [`methodology.md`](methodology.md) §5.

---

## How it is downloaded

The first run pulls the snapshot via `huggingface_hub.snapshot_download` and caches it
locally:

```python
from longextract_bench import dataset as DS

root = DS.dataset_root()        # downloads + caches, returns the snapshot root
docs = DS.doc_dirs(root)        # the 50 document folders, sorted
DS.pdf_path(docs[0])            # -> <doc>/document.pdf
DS.schema_path(docs[0])         # -> <doc>/schema.json
DS.ground_truth_path(docs[0])   # -> <doc>/ground_truth.json
```

* **Cache location:** `.hf_cache/` in the working directory by default, or
  `$LXBENCH_CACHE_DIR`. The cache is gitignored.
* **Offline after first pull:** set `HF_HUB_OFFLINE=1` to work entirely from the local
  cache.
* **Pinning a revision:** `DEFAULT_REVISION` in `dataset.py` is unset until the dataset is
  published; pass `--revision <commit-sha>` to `lxbench-run` / `lxbench-grade` to pin the
  exact dataset bytes a run reads.

The snapshot is **read-only**. Provider outputs and grades are written to a separate,
gitignored `runs/<run-dir>/` tree — never into the dataset cache. The output path for
one (document, system) pair is defined once, in `dataset.run_output_path`:

```
runs/<run-dir>/<slug>/<system>.json
```

---

## Licensing & attribution

* The dataset is licensed **CC-BY-4.0** (stated on the Hugging Face dataset card).
* The benchmark code is **MIT** (`LICENSE`).
* The benchmark was **commissioned by Reducto** and **independently audited and validated
  by micro1**. The systems evaluated (Reducto among them) are not affiliated with the audit
  and are all treated as peers.

If you use the dataset, please attribute it per CC-BY-4.0.
