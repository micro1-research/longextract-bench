#!/usr/bin/env python3
"""Run Anthropic Claude extraction against a PDF + JSON Schema.

Uploads the PDF via the Files API (beta), then makes ONE Messages call with extended
thinking enabled and the schema bound as a tool — the JSON template
(`emit_extraction.input_schema`) is preserved, so output is still schema-shaped
(parity with OpenAI's `json_schema` / Gemini's `response_schema`). Same shared
extraction prompt as the other prompt-driven providers. A `max_tokens` stop is a loud
failure rather than a silently-truncated result.

Reasoning standardization (max thinking): `thinking.type="adaptive"` is the only
thinking mode this model accepts (the `"enabled"` mode with an explicit
`budget_tokens` is rejected). Adaptive has no budget/effort knob — the model scales
its own thinking up for hard tasks, which is the max-reasoning equivalent here. Two
API-inherent constraints follow and are documented for fairness (NOT a methodology
choice):
  • Thinking forbids a FORCED tool_choice, so we use `tool_choice="auto"`. The model
    still reliably calls `emit_extraction` (stop_reason="tool_use"); a turn that
    returns no tool_use is treated as a loud failure below.
  • `temperature` is deprecated for this model and may only be 1 when thinking is on,
    so it is left UNSET (cannot be pinned to 0 like Gemini).

Auth: ANTHROPIC_API_KEY.

Usage:
    python -m longextract_bench.providers.claude \
        --pdf document.pdf --schema schema.json --out /tmp/claude.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import anthropic

from longextract_bench.envelope import write_output
from longextract_bench.prompt import build_extract_prompt

DEFAULT_MODEL = "claude-opus-4-8"
FILES_BETA = "files-api-2025-04-14"
OUTPUT_128K_BETA = "output-128k-2025-02-19"  # lift the output ceiling for big extractions
MAX_TOKENS = 128000  # raised from 64k so large extractions aren't truncated


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")

    schema = json.loads(args.schema.read_text())
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Uploading {args.pdf.name} ({args.pdf.stat().st_size // 1024} KB)…")
    with open(args.pdf, "rb") as f:
        up = client.beta.files.upload(file=(args.pdf.name, f, "application/pdf"))
    file_id = up.id
    print(f"  → file_id: {file_id}")
    # Persist the file id immediately so a killed/interrupted run keeps its handle.
    try:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".runid.json").write_text(
            json.dumps({"provider": "claude", "file_id": file_id, "model": args.model})
        )
    except OSError as e:
        print(f"  (warn: could not write runid sidecar: {e})")

    # Schema bound as a tool = JSON template (schema-shaped output). With thinking
    # on, tool_choice must be "auto" (a forced tool is rejected alongside thinking);
    # the model still reliably emits the emit_extraction tool_use block.
    tool = {
        "name": "emit_extraction",
        "description": "Return the structured data extracted from the document.",
        "input_schema": schema,
    }
    prompt = build_extract_prompt(schema)

    print(f"Calling {args.model}…")
    start = time.monotonic()
    # Stream (the SDK requires it for requests that may exceed 10 min at this
    # max_tokens); we only need the final assembled message.
    with client.beta.messages.stream(
        model=args.model,
        max_tokens=MAX_TOKENS,
        betas=[FILES_BETA, OUTPUT_128K_BETA],
        thinking={"type": "adaptive"},  # max reasoning; only mode this model accepts
        tools=[tool],
        tool_choice={"type": "auto"},  # forced tool is incompatible with thinking
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "document", "source": {"type": "file", "file_id": file_id}},
                ],
            }
        ],
    ) as stream:
        resp = stream.get_final_message()
    elapsed = time.monotonic() - start

    # No silent truncation: a token-limited response is a hard failure.
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(
            f"Claude hit max_tokens (truncated) after {elapsed:.0f}s — result discarded"
        )

    block = next((b for b in resp.content if b.type == "tool_use"), None)
    if block is None:
        raise RuntimeError(f"No tool_use block in response (stop={resp.stop_reason})")
    result = block.input

    usage = resp.usage.model_dump() if hasattr(resp.usage, "model_dump") else dict(resp.usage)
    usage["model"] = resp.model
    usage["stop_reason"] = resp.stop_reason
    usage["client_wall_s"] = round(elapsed, 3)
    print(
        f"  → done in {elapsed:.1f}s  stop={resp.stop_reason}  "
        f"tokens: in={usage.get('input_tokens')} out={usage.get('output_tokens')}"
    )

    write_output(
        args.out, provider="claude", result=result,
        latency_s=elapsed, usage=usage,
    )
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
