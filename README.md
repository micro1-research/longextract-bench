# LongExtractBench

## Abstract

**LongExtractBench** is a benchmark for structured
data extraction from long documents. It measures how well a document-extraction system
turns a long PDF into structured JSON that conforms to a supplied JSON Schema, scored
against a human-reviewed answer key.

This benchmark was **commissioned by Reducto** and **independently audited and validated
by micro1**: micro1 reviewed the methodology, re-ran and re-graded the systems, and
reconciled the ground-truth answer keys. Reducto is one of the systems evaluated and is
held to the same bar as every other. Each system is treated as a peer, running in its own
highest-accuracy configuration behind the same prompt, the same dataset, and the same
deterministic grader.

Its central design choice is that **completion is a first-class result**. Accuracy is
reported only over the documents a system actually completed, with the count of
successful runs shown right next to it — so a system that quietly fails the hardest
documents and posts a high score on the easy ones cannot hide behind a single blended
number.

- **Code** (this repo, GitHub): the runner, the deterministic grader, the scorer, and
  the reporting tooling.
- **Data** ([`micro1-inc/longextract-bench-50`](https://huggingface.co/datasets/micro1-inc/longextract-bench-50)
  on Hugging Face): the 50 document folders. **Not committed here**; it is downloaded on
  demand (see [Generating extractions](#generating-extractions)).

## What's being tested

- **Extraction fidelity** — pulling the right values out of a long PDF, scored as
  precision, recall, and leaf accuracy against a human-reviewed key.
- **Schema conformance** — filling a supplied JSON Schema (mostly large *arrays* of rows
  plus a few document-level scalar fields), matched row by row.
- **Long-document handling** — multi-hundred-page PDFs that exercise context windows,
  page caps, and output limits.
- **Completion** — whether the system produces a usable extraction at all. How many
  documents a system completes is reported alongside accuracy, never blended into it.

## Dataset structure

The corpus is **50 document folders**, a curated representative subset selected by
objective strata — page-count buckets (short to multi-hundred-page), schema-complexity
buckets (flat single-table to deeply nested multi-array), and document domain (financial,
scientific/technical, regulatory, tabular, and others) — so the benchmark exercises the
full range of difficulty, not just the easy middle.

It is hosted on Hugging Face as [`micro1-inc/longextract-bench-50`](https://huggingface.co/datasets/micro1-inc/longextract-bench-50)
(`repo_type="dataset"`) and downloaded + cached at runtime; nothing is committed to git.
The dataset card describes the *distribution* only — it carries no per-document
annotations about which systems succeed or fail.

Each of the 50 items is a self-contained folder of exactly three files:

```
<slug>/
├── document.pdf        # the input PDF
├── schema.json         # the JSON Schema the system is asked to fill
└── ground_truth.json   # the human-reviewed answer key
```

The resolver in [`src/longextract_bench/dataset.py`](src/longextract_bench/dataset.py)
is the single place that knows how to fetch the snapshot and where every file lives. See
[`docs/dataset.md`](docs/dataset.md) for the strata, the download/cache mechanics, and
licensing.

## Setup

Install the package (Python ≥3.10):

```bash
git clone https://github.com/micro1-research/longextract-bench
cd longextract-bench
pip install -e ".[providers,dotenv]"
```

`pip install -e .` is enough to run the grader and reporter offline. The `providers`
extra adds the two SDK-backed runners (Anthropic, Extend); the other five use `httpx`
only. `dotenv` lets the CLI read a local `.env`.

API keys: copy `.env.template` to `.env` and fill in keys only for the systems you
intend to run. Each system runs in its own highest-accuracy configuration (see
[`docs/providers.md`](docs/providers.md)):

| System | How it is called | Auth (env var) |
| --- | --- | --- |
| Reducto | `/extract_async`, deep extract (v2) | `REDUCTO_API_KEY` |
| Extend | `extract_runs`, performance processor | `EXTEND_API_KEY` |
| LlamaExtract | v2 extract, `agentic` tier | `LLAMA_CLOUD_API_KEY` |
| OpenAI | `responses` API, native `json_schema` | `OPENAI_API_KEY` |
| Anthropic Claude | Messages API, schema-as-tool | `ANTHROPIC_API_KEY` |
| Google Gemini | `generateContent`, native `response_schema` | `GEMINI_API_KEY` |
| Datalab | `/extract`, `balanced` tier | `DATALAB_API_KEY` |

## Generating extractions

The 50 documents published here are a **subset of the full LongExtractBench corpus** — drawn from the same documents the systems were evaluated on — released so the extraction task can be run and inspected independently.

The benchmark data is **not** in this repo; the first run downloads it from
[`micro1-inc/longextract-bench-50`](https://huggingface.co/datasets/micro1-inc/longextract-bench-50)
into a local cache (`.hf_cache/` by default, or `$LXBENCH_CACHE_DIR`). Set
`HF_HUB_OFFLINE=1` afterwards to work fully offline, or pass `--revision <commit-sha>`
to pin an exact dataset commit.

Three commands run the whole pipeline — run, grade, report:

```bash
# 1. Fan the systems out over the corpus -> runs/latest/<slug>/<system>.json
lxbench-run    --run-dir runs/latest

# 2. Classify each result (success/failure) and grade the successful ones -> runs/latest/scores.json
lxbench-grade  --run-dir runs/latest

# 3. Aggregate scores.json into the leaderboard + failures + latency blocks
lxbench-report --run-dir runs/latest
```

`lxbench-run` is **resumable**: a (document, system) pair whose output already exists is
skipped, and providers persist their server-side job id so an interrupted run re-polls
rather than re-submitting (no double-billing). Failed attempts are retried up to
`--retries` times (default 3). Run a single system or a subset with `--providers`, tune
parallelism with `-j/--concurrency`, and re-run everything with `--force`:

```bash
lxbench-run --run-dir runs/latest --providers openai,claude -j 4
```

> **Running `lxbench-run` calls paid third-party APIs.** A full 7-system × 50-document
> sweep is real spend. The grader and reporter are free and offline.

Running these commands **generates a fresh set of extractions** on the public corpus and
then grades and reports them — it produces your own run, not a copy of any previously
published numbers. The scores are **point-in-time**: the systems are stochastic and the
upstream APIs change, so run-to-run deltas are expected. Record the run date and the
resolved model IDs (the runner logs them); the *grader itself is fully deterministic* —
the same outputs always grade to the same scores.

## Metrics

Each (document, system) run is exactly one of two states:

- **success** — the system produced a real, non-empty extraction.
- **failure** — the run produced no usable output (output truncation, context-window
  overflow, page caps, schema-complexity ceilings, timeouts, internal errors, throttles,
  refusals, or no output file).

`lxbench-report` then prints, **separately — never blended into a single number**:

```
## 1. Leaderboard (accuracy over COMPLETED documents only)
system        successful  precision   recall   leaf
...

## 2. Failures (run did not produce a usable result)
## 3. Latency (over completed documents)
## Reconciliation (success + failure == #docs)
```

1. **Accuracy over completed documents** — precision, recall, and leaf accuracy,
   computed **only over the documents a system completed successfully**, alongside the
   count of successful runs.
2. **Failures** — how many documents each system failed, and why.
3. **Latency** — wall-clock seconds over completed documents.

The split is the whole point: a system that quietly fails the hardest 20% of documents
and posts a high accuracy on the easy 80% looks very different once the successful-run
count is visible next to the accuracy column. A reconciliation line proves
`success + failure == #docs` for every system. Comparing two systems means comparing
accuracy *and* how many documents each one actually completed.

The metrics and the success/failure rule are detailed in
[`docs/methodology.md`](docs/methodology.md).

## Documentation

- [Methodology](docs/methodology.md): the metrics, the success/failure rule, and the
  reporting blocks.
- [Providers](docs/providers.md): each system's API mode and highest-accuracy config.
- [Dataset](docs/dataset.md): the 50 documents — strata, download/cache, and licensing.

## Repo layout

```
longextract-bench/
├── README.md
├── pyproject.toml              # deps + console scripts (lxbench-run/grade/report)
├── .env.template               # the 7 provider key names, blank
├── docs/                       # methodology, providers, dataset
└── src/longextract_bench/
    ├── dataset.py              # Hugging Face snapshot resolver + run-output paths
    ├── prompt.py               # the single shared extraction prompt
    ├── envelope.py             # uniform output envelope
    ├── runner.py               # parallel fan-out + bounded retries (resumable)
    ├── grading.py              # deterministic metric engine
    ├── classify.py             # single success/failure classifier
    ├── score.py                # per-(doc,system) status + grade -> scores.json
    ├── report.py               # scores.json -> leaderboard + failures + latency
    ├── cli.py                  # console entry points
    └── providers/              # the 7 standalone runners
```

Each provider is standalone-runnable for one document:

```bash
python -m longextract_bench.providers.openai \
    --pdf doc.pdf --schema schema.json --out /tmp/openai.json
```

## License

Code: MIT (© 2026 Micro1). Dataset: CC-BY-4.0 (stated on the Hugging Face dataset card).
See [LICENSE](LICENSE).
