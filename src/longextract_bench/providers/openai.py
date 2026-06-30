#!/usr/bin/env python3
"""Run OpenAI extraction against a PDF + JSON Schema.

Encodes the PDF as base64 and sends it via the responses API with file input and a
native `json_schema` response format. A truncated/incomplete run is a hard failure
rather than a silently-truncated result.

Auth: OPENAI_API_KEY.

Usage:
    python -m longextract_bench.providers.openai \
        --pdf document.pdf --schema schema.json --out /tmp/openai.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx

from longextract_bench.envelope import write_output
from longextract_bench.prompt import build_extract_prompt

DEFAULT_MODEL = "gpt-5.5-pro"
BASE = "https://api.openai.com/v1"


def extract(
    api_key: str, pdf_path: Path, schema: dict, model: str
) -> tuple[dict, dict, float]:
    print(f"Encoding {pdf_path.name} ({pdf_path.stat().st_size // 1024} KB)…")
    pdf_b64 = base64.b64encode(pdf_path.read_bytes()).decode()

    prompt = build_extract_prompt(schema)

    payload = {
        "model": model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_file",
                        "filename": pdf_path.name,
                        "file_data": f"data:application/pdf;base64,{pdf_b64}",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        # Maximize reasoning for this hard long-doc extraction task rather than
        # relying on the model's default effort tier.
        "reasoning": {"effort": "high"},
        # NOTE: `temperature` is intentionally absent — the model rejects the
        # parameter entirely ("Unsupported parameter"), so unlike Gemini it cannot
        # be pinned to 0. An API-inherent constraint, documented for fairness.
        "text": {
            "format": {
                "type": "json_schema",
                "name": "extraction",
                "schema": schema,
                "strict": False,
            }
        },
        # no max_output_tokens cap — let the model use its full default output
        # budget (parity with Gemini); a truncated/incomplete run is caught below.
    }

    print(f"Calling {model}…")
    # LATENCY = the OpenAI API request ONLY. We prefer the server-reported
    # `openai-processing-ms` response header (pure server processing, excludes
    # network) and fall back to a tight client timer around just this one
    # /responses call. No encoding, prompt build, upload, or client queueing.
    start = time.monotonic()
    resp = httpx.post(
        f"{BASE}/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=httpx.Timeout(30, read=3600),
    )
    resp.raise_for_status()
    client_elapsed = time.monotonic() - start
    proc_ms = resp.headers.get("openai-processing-ms")
    elapsed = float(proc_ms) / 1000 if proc_ms else client_elapsed

    data = resp.json()
    # No silent truncation: an incomplete (e.g. token-limited) run is a hard failure.
    if data.get("status") == "incomplete":
        raise RuntimeError(
            f"OpenAI run incomplete: {data.get('incomplete_details')} — result discarded"
        )
    usage = data.get("usage", {})
    print(
        f"  → done in {elapsed:.1f}s  tokens: input={usage.get('input_tokens')} output={usage.get('output_tokens')}"
    )

    # Track the full response metadata (all keys except the bulky `output` result
    # payload) plus the server-side timing header and request id, for transparency.
    details = {k: v for k, v in data.items() if k != "output"}
    details["openai_processing_ms"] = float(proc_ms) if proc_ms else None
    details["client_wall_s"] = round(client_elapsed, 3)
    details["x_request_id"] = resp.headers.get("x-request-id")

    # Find the message output (skip reasoning blocks)
    msg = next((o for o in data["output"] if o.get("type") == "message"), None)
    if not msg:
        raise ValueError(f"No message in output: {data['output']}")
    text = msg["content"][0]["text"].strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

    return json.loads(text), details, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set")

    schema = json.loads(args.schema.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    result, details, latency_s = extract(api_key, args.pdf, schema, args.model)

    for key, v in result.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {key}: {len(v)} rows")

    # `usage` holds the FULL metadata blob (token usage, timestamps, model,
    # service_tier, timing) for transparency and traceability.
    write_output(
        args.out, provider="openai", result=result,
        latency_s=latency_s, usage=details,
    )
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
