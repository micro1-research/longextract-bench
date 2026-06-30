#!/usr/bin/env python3
"""Run Gemini extraction against a PDF + JSON Schema.

Uploads the PDF via the Gemini Files API, then makes a SINGLE non-streaming
`generateContent` call (1-hour read timeout, mirroring the OpenAI runner) with the
schema passed in Gemini's native `response_schema` slot (constrained decoding —
parity with how OpenAI gets `json_schema` and the schema-native systems get the
schema directly). No max-output-token cap; a `MAX_TOKENS` finish is a loud failure
rather than a silently-truncated result.

Auth: GEMINI_API_KEY.

Usage:
    python -m longextract_bench.providers.gemini \
        --pdf doc.pdf --schema schema.json --out /tmp/gemini.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import httpx

from longextract_bench.envelope import write_output
from longextract_bench.prompt import build_extract_prompt

DEFAULT_MODEL = "gemini-3.1-pro-preview"
BASE = "https://generativelanguage.googleapis.com"


def _adapt_schema_gemini(schema: dict, defs: dict | None = None) -> dict:
    """Convert a generic JSON Schema to Gemini's `response_schema` dialect (OpenAPI subset).

    Gemini rules: no `$ref`/`$defs` (inline), no `$`-prefixed metadata keys
    (`$schema` etc.), no `additionalProperties`, and nullability is expressed as
    `nullable: true` rather than a `["type","null"]` array. This is a pure dialect
    translation — no field, description, enum, or type-category is changed.
    """
    if defs is None:
        defs = schema.get("$defs", {})
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        return _adapt_schema_gemini(
            copy.deepcopy(defs.get(schema["$ref"].split("/")[-1], {})), defs
        )
    # Drop $-prefixed metadata, annotation-only keywords (documentation, not
    # extraction semantics), and array-length bounds. Gemini's OpenAPI subset
    # rejects unknown keys such as "examples" and rejects minItems/maxItems with
    # large values (some schemas bake the exact row count into them). Stripping
    # these changes no field/type/description/enum — it only drops a count
    # constraint Gemini cannot express, so the benchmark stays fair.
    _drop = {
        "examples", "$comment", "default", "readOnly", "writeOnly", "deprecated",
        "minItems", "maxItems",
    }
    node = {k: v for k, v in schema.items() if not k.startswith("$") and k not in _drop}
    node.pop("additionalProperties", None)
    t = node.get("type")
    if isinstance(t, list):
        non_null = [x for x in t if x != "null"]
        node["type"] = non_null[0] if non_null else "string"
        if "null" in t:
            node["nullable"] = True
    if "properties" in node:
        node["properties"] = {
            k: _adapt_schema_gemini(v, defs) for k, v in node["properties"].items()
        }
    if "items" in node:
        node["items"] = _adapt_schema_gemini(node["items"], defs)
    return node


def upload_pdf(api_key: str, pdf_path: Path) -> str:
    """Upload PDF via Gemini Files API, return file URI."""
    size = pdf_path.stat().st_size
    print(f"Uploading {pdf_path.name} ({size // 1024} KB)…")
    resp = httpx.post(
        f"{BASE}/upload/v1beta/files",
        params={"key": api_key},
        headers={
            "X-Goog-Upload-Protocol": "resumable",
            "X-Goog-Upload-Command": "start",
            "X-Goog-Upload-Header-Content-Length": str(size),
            "X-Goog-Upload-Header-Content-Type": "application/pdf",
            "Content-Type": "application/json",
        },
        content=json.dumps({"file": {"display_name": pdf_path.name}}),
        timeout=30,
    )
    resp.raise_for_status()
    upload_url = resp.headers["x-goog-upload-url"]
    resp = httpx.post(
        upload_url,
        headers={
            "Content-Length": str(size),
            "X-Goog-Upload-Offset": "0",
            "X-Goog-Upload-Command": "upload, finalize",
        },
        content=pdf_path.read_bytes(),
        timeout=120,
    )
    resp.raise_for_status()
    file_uri = resp.json()["file"]["uri"]
    print(f"  → uri: {file_uri}")
    return file_uri


def extract(
    api_key: str, file_uri: str, schema: dict, model: str
) -> tuple[dict, dict, float]:
    """Single non-streaming generateContent call with native response_schema."""
    payload = {
        "contents": [
            {
                "parts": [
                    {"file_data": {"mime_type": "application/pdf", "file_uri": file_uri}},
                    {"text": build_extract_prompt(schema)},
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": _adapt_schema_gemini(schema),
            "temperature": 0,
        },
    }

    print(f"Calling {model}…")
    # LATENCY = the generateContent API request ONLY (Gemini exposes no server-side
    # timing field, so this client timer around the single POST is the best
    # available — it excludes file upload, prompt/schema build, and client queueing).
    start = time.monotonic()
    resp = httpx.post(
        f"{BASE}/v1beta/models/{model}:generateContent",
        params={"key": api_key},
        json=payload,
        timeout=httpx.Timeout(30, read=3600),
    )
    resp.raise_for_status()
    elapsed = time.monotonic() - start

    data = resp.json()
    cand = data["candidates"][0]
    finish = cand.get("finishReason")
    # No silent truncation: a token-limited response is a hard failure.
    if finish == "MAX_TOKENS":
        raise RuntimeError(
            f"Gemini hit MAX_TOKENS (truncated) after {elapsed:.0f}s — result discarded"
        )
    usage = data.get("usageMetadata", {})
    print(
        f"  → done in {elapsed:.1f}s  finish={finish}  "
        f"tokens: input={usage.get('promptTokenCount')} output={usage.get('candidatesTokenCount')}"
    )

    result = json.loads(cand["content"]["parts"][0]["text"])
    # Track the full response metadata for transparency and traceability.
    details = {
        "usageMetadata": usage,
        "modelVersion": data.get("modelVersion"),
        "responseId": data.get("responseId"),
        "finishReason": finish,
        "client_wall_s": round(elapsed, 3),
    }
    return result, details, elapsed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        sys.exit("GEMINI_API_KEY not set")

    schema = json.loads(args.schema.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    file_uri = upload_pdf(api_key, args.pdf)
    result, details, latency_s = extract(api_key, file_uri, schema, args.model)

    for key, v in result.items():
        if isinstance(v, list) and len(v) > 5:
            print(f"  {key}: {len(v)} rows")
            break

    # `usage` holds the FULL metadata blob (usageMetadata tokens, model version,
    # response id, finishReason) for transparency and traceability.
    write_output(
        args.out, provider="gemini", result=result,
        latency_s=latency_s, usage=details,
    )
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
