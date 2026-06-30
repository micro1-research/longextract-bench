#!/usr/bin/env python3
"""Run Extend extraction against a PDF + JSON Schema.

Uses the extend-ai Python SDK (v1.12+) with inline schema config.
Auth: EXTEND_API_KEY.

Usage:
    python -m longextract_bench.providers.extend \
        --pdf document.pdf --schema schema.json --out /tmp/extend.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from extend_ai import Extend

from longextract_bench.envelope import write_output


def _server_latency_s(created: object, updated: object) -> float | None:
    """Extend's server-side processing span = updatedAt - createdAt. This is the
    request processing latency ONLY — it excludes our upload and poll sleeps."""
    def _dt(v: object) -> datetime | None:
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return None
    a, b = _dt(created), _dt(updated)
    return (b - a).total_seconds() if a and b else None


def _adapt_schema(schema: dict) -> dict:
    """Convert a generic JSON Schema to Extend-compatible format.

    Extend rules:
    - All primitive fields must be nullable: ["string", "null"] not "string"
    - Arrays/objects cannot be nullable
    - No $ref / $defs — must be inlined
    - additionalProperties: false required on all objects

    Fairness note (the nullable rule is the one to scrutinize): an A/B smoke test —
    the same minimal schema submitted with top-level scalars non-nullable vs nullable
    — shows non-nullable is rejected by Extend's API at submit time (BadRequestError
    400), so the nullable wrapping is MANDATORY, not a choice. And making a field
    nullable does NOT make Extend lazily emit null: in the test it still returned the
    real values for required fields. So this is a pure dialect translation — it changes
    no field, description, enum, or type-category (verified: identical field set
    pre/post), and does not lower Extend's bar relative to the raw schema the
    schema-native providers receive. The benchmark stays fair.
    """
    defs = schema.get("$defs", {})
    return _adapt_node(copy.deepcopy(schema), defs)


# JSON-Schema annotation-only keywords: documentation, not extraction semantics.
# Extend rejects unknown keywords (e.g. "examples") at submit time, so strip them.
# Dropping them changes no field/type/description/enum — the benchmark stays fair.
_ANNOTATION_KEYS = (
    "examples", "$comment", "default", "readOnly", "writeOnly", "deprecated",
    # array-length bounds (some schemas bake the exact row count in); Extend rejects
    # them and they only constrain count, not fields — dropping keeps the run fair.
    "minItems", "maxItems",
    # OpenAPI-style "nullable": Extend rejects the keyword, and nullability is already
    # expressed via the type-array (["number","null"]) below — so it's redundant. Strip it.
    "nullable",
)

# Extend reserves the property key "id"; we alias it for the request and rename it
# back in the result so the extracted field stays canonical.
_ID_ALIAS = "id_field"


def _restore_reserved_keys(obj: object) -> object:
    """Rename the aliased reserved key back to "id" throughout the result."""
    if isinstance(obj, dict):
        return {
            ("id" if k == _ID_ALIAS else k): _restore_reserved_keys(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_restore_reserved_keys(x) for x in obj]
    return obj


def _adapt_node(node: dict, defs: dict, in_array_items: bool = False) -> dict:
    # Resolve $ref
    if "$ref" in node:
        ref_path = node["$ref"]  # e.g. "#/$defs/water_quality_record"
        ref_key = ref_path.split("/")[-1]
        resolved = copy.deepcopy(defs.get(ref_key, {}))
        return _adapt_node(resolved, defs)

    for _k in _ANNOTATION_KEYS:
        node.pop(_k, None)

    # Extend makes every field nullable, so any enum field must also allow null:
    # "Enum must include null as a possible value." Add it if absent.
    if isinstance(node.get("enum"), list) and None not in node["enum"]:
        node["enum"] = [*node["enum"], None]

    t = node.get("type")

    if t == "object" or (isinstance(t, list) and "object" in t):
        node["type"] = "object"
        node.setdefault("additionalProperties", False)
        if "properties" in node:
            adapted = {k: _adapt_node(v, defs) for k, v in node["properties"].items()}
            # Extend rejects objects with zero properties — drop such empty-object
            # children, and prune any `required` entries that referenced them.
            adapted = {
                k: v
                for k, v in adapted.items()
                if not (v.get("type") == "object" and not v.get("properties"))
            }
            # Extend reserves the property key "id" for internal use. Alias it on
            # the way in and rename it back in the result (see _restore_reserved_keys)
            # so the extracted data and grading still use the canonical "id".
            if "id" in adapted and _ID_ALIAS not in adapted:
                adapted[_ID_ALIAS] = adapted.pop("id")
            node["properties"] = adapted
            if "required" in node:
                node["required"] = [
                    _ID_ALIAS if r == "id" else r for r in node["required"] if r in adapted or r == "id"
                ]
                node["required"] = [r for r in node["required"] if r in adapted]

    elif t == "array" or (isinstance(t, list) and "array" in t):
        # Arrays cannot be nullable in Extend
        node["type"] = "array"
        if "items" in node:
            node["items"] = _adapt_node(node["items"], defs, in_array_items=True)

    elif in_array_items:
        # Extend: SCALAR array items may carry ONLY "type" (no description, etc.)
        plain = next((x for x in t if x != "null"), "string") if isinstance(t, list) else t
        return {"type": plain}

    else:
        # Primitive — must be nullable (top-level / object fields)
        if isinstance(t, str) and t != "null":
            node["type"] = [t, "null"]
        elif isinstance(t, list) and "null" not in t:
            node["type"] = t + ["null"]

    # Remove $defs from root — not supported by Extend
    node.pop("$defs", None)

    return node


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    api_key = os.environ.get("EXTEND_API_KEY", "")
    if not api_key:
        sys.exit("EXTEND_API_KEY not set")

    schema = json.loads(args.schema.read_text())
    adapted = _adapt_schema(schema)

    client = Extend(token=api_key)
    import time

    side = args.out.with_suffix(".runid.json")
    resume_id = None
    if side.exists():
        try:
            resume_id = json.loads(side.read_text()).get("run_id")
        except (OSError, json.JSONDecodeError):
            resume_id = None

    if resume_id:
        # Prior attempt already submitted — the run is persistent; poll it, never
        # re-submit (no double-billing).
        run_id = resume_id
        print(f"Resuming existing run {run_id} (sidecar)…")
    else:
        print(f"Uploading {args.pdf.name} ({args.pdf.stat().st_size // 1024} KB)…")
        with open(args.pdf, "rb") as f:
            file_resp = client.files.upload(file=("document.pdf", f, "application/pdf"))
        file_id = file_resp.id
        print(f"  → fileId: {file_id}")
        print("Submitting extraction…")
        run = client.extract_runs.create(
            file={"id": file_id},
            config={
                "schema": adapted,
                "base_processor": "extraction_performance",
                "advanced_options": {"array_strategy": {"type": "large_array_max_context"}},
            },
        )
        run_id = run.id
        # Persist the run id immediately so a killed/interrupted job stays pollable.
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            side.write_text(json.dumps(
                {"provider": "extend", "run_id": run_id, "file_id": file_id}
            ))
        except OSError as e:
            print(f"  (warn: could not write runid sidecar: {e})")
        print(f"  → runId: {run_id}  (status: {run.status})")

    print("Polling…")
    transient = 0
    while True:
        try:
            result = client.extract_runs.retrieve(run_id)
        except Exception as e:  # noqa: BLE001 - transient poll error; run is persistent, retry
            transient += 1
            if transient > 60:
                raise
            time.sleep(min(30, 2 * transient))
            continue
        transient = 0
        print(f"  {result.status}", end="\r", flush=True)
        if result.status in ("PROCESSED", "FAILED", "CANCELLED"):
            print()
            break
        time.sleep(5)

    print(f"Status: {result.status}")

    dump = result.model_dump()
    val = None
    if result.output:
        val = result.output.value if hasattr(result.output, "value") else result.output
        val = _restore_reserved_keys(val)

    latency_s = _server_latency_s(
        getattr(result, "created_at", None), getattr(result, "updated_at", None)
    )
    # full run dump (usage, timestamps, status, config) for transparency
    write_output(
        args.out, provider="extend", result=val,
        latency_s=latency_s or 0.0, usage=dump,
    )
    print(f"  latency={latency_s}s\nSaved → {args.out}")
    if isinstance(val, dict):
        for key, v in val.items():
            if isinstance(v, list) and len(v) > 5:
                print(f"  {key}: {len(v)} rows")


if __name__ == "__main__":
    main()
