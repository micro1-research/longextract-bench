"""The single source of truth for run status.

Every (document, system) run is exactly one of two states:

  * success — the system produced a real extraction (a non-empty result), which the
              grader then scores.
  * failure — the run did not produce usable output: a persisted failure marker, an
              empty/null result, an unreadable envelope, or no output file at all.

That is the whole taxonomy. Accuracy metrics are computed only over `success` runs,
and the report states how many runs each system completed successfully.
"""

from __future__ import annotations


def classify(env: dict) -> tuple[str, str | None]:
    """Classify a provider output envelope into ("success"|"failure", detail).

    * failure — a persisted failure marker (`_meta.status == "failed"`), or a
                structurally-successful run that produced no extraction (null/{}/[]).
                `detail` is the captured error text, when present.
    * success — a real, non-empty extraction.
    """
    m = env.get("_meta") or {}
    if m.get("status") == "failed":
        return "failure", str(m.get("error") or "")
    r = env.get("result")
    if r in ({}, None, []):
        return "failure", "empty result"
    return "success", None
