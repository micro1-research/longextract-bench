"""Dataset resolver for LongExtractBench.

The 50-document corpus (each a folder with `document.pdf`, a schema, and a
ground-truth answer key) is hosted on Hugging Face and downloaded + cached at
runtime — nothing is committed to git. This module is the single place that knows
how to fetch the dataset and where every file lives, including where a run writes
its provider outputs (a separate, gitignored `runs/` tree, since the HF cache is
read-only).

First run downloads the snapshot; set `HF_HUB_OFFLINE=1` afterwards to work fully
offline against the local cache.
"""

from __future__ import annotations

import os
from pathlib import Path

from .grading import GT_NAMES, SCHEMA_NAMES

# Hugging Face dataset coordinates. Pin a revision once the dataset is published so
# every run resolves the exact same bytes; `--revision` (CLI) overrides at runtime.
DEFAULT_REPO_ID = "micro1-inc/longextract-bench-50"
DEFAULT_REVISION: str | None = None  # set to a pinned commit SHA at publish time

# Local cache for the downloaded snapshot (gitignored).
DEFAULT_CACHE_DIR = Path(os.environ.get("LXBENCH_CACHE_DIR", ".hf_cache")).resolve()

PDF_NAME = "document.pdf"


def dataset_root(
    repo_id: str | None = None,
    revision: str | None = None,
    cache_dir: Path | None = None,
) -> Path:
    """Download (or reuse the cache of) the HF dataset snapshot and return its root.

    Requires `huggingface_hub`. The snapshot is read-only; never write into it.
    """
    from huggingface_hub import snapshot_download

    local = snapshot_download(
        repo_id=repo_id or DEFAULT_REPO_ID,
        repo_type="dataset",
        revision=revision if revision is not None else DEFAULT_REVISION,
        cache_dir=str((cache_dir or DEFAULT_CACHE_DIR)),
    )
    return Path(local)


def doc_dirs(root: Path) -> list[Path]:
    """Every document folder in the dataset (one per benchmark item), sorted."""
    return sorted(
        d for d in root.iterdir()
        if d.is_dir() and (d / PDF_NAME).exists() and schema_path(d) is not None
    )


def pdf_path(doc: Path) -> Path:
    return doc / PDF_NAME


def schema_path(doc: Path) -> Path | None:
    """First matching schema filename in a doc folder."""
    return next((doc / n for n in SCHEMA_NAMES if (doc / n).exists()), None)


def ground_truth_path(doc: Path) -> Path | None:
    """First matching ground-truth filename in a doc folder."""
    return next((doc / n for n in GT_NAMES if (doc / n).exists()), None)


def slug_of(doc: Path) -> str:
    """Stable identifier for a document folder (used in the runs/ output tree)."""
    return doc.name


def run_output_path(run_dir: Path, slug: str, provider: str) -> Path:
    """Where one (document, provider) extraction envelope is written.

    Outputs live under a gitignored `runs/<run-dir>/<slug>/<provider>.json` tree —
    separate from the read-only HF snapshot. Centralized here so the runner, scorer,
    and reporter all agree on the layout.
    """
    return run_dir / slug / f"{provider}.json"
