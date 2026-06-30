"""The uniform provider output envelope.

Every provider writes the same `{result, _meta}` shape so each run carries the
extraction plus wall-clock latency and the raw vendor usage block. Keeping the
envelope in one place is what lets every system be read uniformly by the grader.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_output(
    out_path: Path,
    *,
    provider: str,
    result: Any,
    latency_s: float,
    usage: Any,
) -> None:
    """Write the uniform provider output envelope so every run carries the
    extraction plus wall-clock latency in the same shape.

    `usage` holds the raw vendor usage/response block (page counts, token counts,
    job metadata) exactly as returned, for transparency and traceability.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "result": result,
                "_meta": {
                    "provider": provider,
                    "latency_s": round(latency_s, 2),
                    "usage": usage,
                },
            },
            indent=2,
            default=str,
        )
    )
