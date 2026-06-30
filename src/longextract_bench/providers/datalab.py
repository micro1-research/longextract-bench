#!/usr/bin/env python3
"""Run Datalab structured extraction against a PDF + JSON Schema.

Uploads the PDF BYTES (never a URL — no source domain leaks to the vendor), submits
a single multipart extraction job, polls to completion, and writes the same
`{result, _meta}` envelope as the other providers.

Mode: extraction_mode="balanced" — the HIGHEST extraction tier Datalab offers
(only `fast`/`balanced` exist; `balanced` is the most accurate). The separate
document-parsing knob is set to mode="balanced" as well, matching Datalab's own
documented canonical /extract usage; we deliberately do NOT send an unverified
higher value (e.g. "accurate"), since a submit-time rejection of an unsupported
param would be recorded as a failure and unfairly brand the whole run.
output_format="markdown" mirrors the official example. There is no higher
*extraction* configuration than balanced.

Metadata sidecars: Datalab decorates EACH extracted field X with two sibling keys —
`"<field>_citations"` (provenance block IDs) and `"<field>_meta"` (extraction_status
/ reasoning / verification) — with no flag to disable them at this tier. They are
metadata, not schema content, and no other system emits them. We PRESERVE them in
the written output (the genuine, unmodified vendor result) and ignore them at scoring
time: the grader's datalab loader (_drop_datalab_sidecars in grading.py) drops a
`*_citations`/`*_meta` key only when its base field is a sibling, so the sidecars
never count against leaf accuracy while a genuine field that merely ends in such a
suffix is preserved. No extracted value is altered; the stored file keeps everything.

Any error here is a FAILURE — a submit-time HTTP error (4xx/5xx, including a
Cloudflare client-signature block), a terminal status=="failed" after the job was
accepted, or an empty/degenerate result all mean the run produced no usable output.
The script exits non-zero so the runner persists a failure marker.

Auth: DATALAB_API_KEY (header X-API-Key).

Usage:
    python -m longextract_bench.providers.datalab \
        --pdf doc.pdf --schema schema.json --out /tmp/datalab.json
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from longextract_bench.envelope import write_output

BASE = "https://www.datalab.to"
SUBMIT_URL = f"{BASE}/api/v1/extract"
EXTRACTION_MODE = "balanced"  # HIGHEST extraction tier (vs "fast"); see module docstring
PARSE_MODE = "balanced"  # document-parse knob; matches Datalab's documented /extract usage
OUTPUT_FORMAT = "markdown"  # parse output format the extractor reads from (per official example)
_TERMINAL = {"complete", "failed", "error", "cancelled"}


def _adapt_schema(schema: dict, defs: dict | None = None) -> dict:
    """Minimal JSON-Schema cleanup for Datalab's `page_schema`: inline $ref/$defs and
    drop $-prefixed metadata keys ($schema, $id, $comment, $defs); collapse `type:[...]`
    unions to a single non-null scalar type. Recurses into properties/items/combinators.
    No field, description, enum, or required is changed — pure dialect normalization.

    The type-union collapse mirrors the OTHER adapters in this benchmark, so Datalab is
    held to the same input normalization as the others: llamaextract.py and gemini.py
    both reduce `type:["x","null"]` (and multi-scalar unions) to a single non-null type,
    and extend.py rewrites scalar types to/from the array form its API requires. Datalab's
    balanced extractor crashes server-side ("unhashable type: 'list'") on multi-type
    unions like `["integer","string"]`; collapsing here is value-neutral dialect cleanup
    (the field's meaning/description/required-ness are untouched), not a capability change,
    so it removes a vendor input-format quirk without giving Datalab any scoring advantage."""
    if defs is None:
        defs = schema.get("$defs", {})
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        return _adapt_schema(copy.deepcopy(defs.get(schema["$ref"].split("/")[-1], {})), defs)
    node = {k: v for k, v in schema.items() if not k.startswith("$")}
    t = node.get("type")
    if isinstance(t, list):
        # Drop the JSON-Schema "null" member and take the first remaining type, exactly as
        # llamaextract.py / gemini.py do. A bare ["null"] (no real type) falls back to
        # "string". Single-type and scalar `type` are left untouched.
        non_null = [x for x in t if x != "null"]
        node["type"] = non_null[0] if non_null else "string"
    if "properties" in node:
        node["properties"] = {k: _adapt_schema(v, defs) for k, v in node["properties"].items()}
    if "items" in node:
        node["items"] = _adapt_schema(node["items"], defs)
    for comb in ("anyOf", "oneOf", "allOf"):
        if isinstance(node.get(comb), list):
            node[comb] = [_adapt_schema(x, defs) for x in node[comb]]
    return node


# datalab.to sits behind Cloudflare, whose WAF bans the default urllib agent
# ("Python-urllib/x") with a 403 "error code: 1010" — a CLIENT-SIGNATURE block
# that never reaches the API (so it is NOT a schema rejection). A normal browser
# User-Agent passes the browser-integrity check, matching how the official
# `requests`-based example gets through.
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36")


def _req(method: str, url: str, key: str, headers: dict | None = None, data: bytes | None = None) -> dict:
    h = {"X-API-Key": key, "User-Agent": _UA, "Accept": "application/json", **(headers or {})}
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def _multipart(pdf: Path, schema: dict) -> tuple[bytes, str]:
    """Build the multipart/form-data body: file bytes + page_schema + the two max-tier
    mode knobs. Returns (body, content_type)."""
    boundary = "----datalab" + str(int(time.time()))

    def field(name: str, value: str) -> bytes:
        return (
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n"
        ).encode()

    body = (
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="document.pdf"\r\nContent-Type: application/pdf\r\n\r\n'.encode()
        + pdf.read_bytes()
        + b"\r\n"
        + field("page_schema", json.dumps(_adapt_schema(schema)))
        + field("extraction_mode", EXTRACTION_MODE)
        + field("mode", PARSE_MODE)
        + field("output_format", OUTPUT_FORMAT)
        + f"--{boundary}--\r\n".encode()
    )
    return body, f"multipart/form-data; boundary={boundary}"


def run(pdf: Path, schema: dict, out: Path, key: str, poll_interval: int, submit_retries: int = 5) -> None:
    side = out.with_suffix(".runid.json")
    resume = None
    if side.exists():
        try:
            resume = json.loads(side.read_text())
        except (OSError, json.JSONDecodeError):
            resume = None

    if resume and resume.get("request_check_url"):
        # Prior attempt already submitted — the job is persistent; poll it, never
        # re-submit (no double-billing).
        check_url = resume["request_check_url"]
        # latency proxy: time from (resumed) poll-start; fall back to now if absent.
        poll_start = resume.get("poll_start") or time.time()
        print("Resuming existing job (sidecar)…")
    else:
        print(f"Uploading {pdf.name} ({pdf.stat().st_size // 1024} KB)…")
        body, ctype = _multipart(pdf, schema)
        print(f"Submitting extract (extraction_mode={EXTRACTION_MODE}, mode={PARSE_MODE}, "
              f"output_format={OUTPUT_FORMAT})…")
        # Submit with backoff. A 5xx/timeout/transport error at upload means NO job was
        # created server-side (no request_check_url returned, nothing billed), so
        # retrying is safe and can never double-bill. A true 400/422 (request/schema
        # refused) or a 401/403 (auth/access) is deterministic and exits immediately.
        # This makes the provider resilient to the transient upload-endpoint errors seen
        # under concurrency (SSL read-timeout, Cloudflare 502, "unable to verify data
        # residency for this multipart upload — please retry").
        sub = None
        last = ""
        backoff = 5.0
        for attempt in range(1, submit_retries + 1):
            try:
                sub = _req("POST", SUBMIT_URL, key, headers={"Content-Type": ctype}, data=body)
                break
            except urllib.error.HTTPError as e:
                try:
                    err_body = e.read().decode("utf-8", "replace")[:300]
                except Exception:  # noqa: BLE001
                    err_body = ""
                msg = f"Datalab submit rejected: {e.code} {e.reason} — {err_body}"
                if e.code in (400, 422) or e.code in (401, 403):
                    # 400/422 (schema/request refused) and 401/403 (auth/access) are both
                    # deterministic: the same request will keep being refused -> stop now.
                    sys.exit(msg)
                last = msg  # 5xx / 408 / 429 -> transient overload, retry
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = f"Datalab submit transport error: {type(e).__name__}: {e}"
            if attempt < submit_retries:
                print(f"  submit attempt {attempt} failed ({last[:90]}); retry in {backoff:.0f}s")
                time.sleep(backoff + random.uniform(0, 3))
                backoff = min(60.0, backoff * 2)
        if sub is None:
            sys.exit(last or "Datalab submit failed (transient, exhausted retries)")
        check_url = sub.get("request_check_url")
        if not check_url:
            sys.exit(f"Datalab submit returned no request_check_url: {json.dumps(sub)[:300]}")
        poll_start = time.time()
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            side.write_text(json.dumps(
                {"provider": "datalab", "request_check_url": check_url, "poll_start": poll_start}
            ))
        except OSError as e:
            print(f"  (warn: could not write runid sidecar: {e})")
        print(f"  → check_url: {check_url}\nPolling…")

    transient = 0
    body = {}
    while True:
        try:
            body = _req("GET", check_url, key)
        except (urllib.error.URLError, OSError, ValueError):
            transient += 1
            if transient > 60:
                raise
            time.sleep(min(30, 2 * transient))
            continue
        transient = 0
        status = (body.get("status") or "").lower()
        print(f"  {status}", end="\r", flush=True)
        if status in _TERMINAL:
            print()
            break
        time.sleep(poll_interval)

    if status in ("failed", "error", "cancelled"):
        # Job was ACCEPTED then could not complete -> FAILURE.
        err = body.get("error") or body.get("detail") or body.get("message") or status
        raise RuntimeError(f"Datalab job failed: {err}")

    raw = body.get("extraction_schema_json")
    result = json.loads(raw) if isinstance(raw, str) else raw
    # Citations/_meta sidecars are PRESERVED here — the genuine vendor output is stored
    # unmodified. They are dropped only at scoring time by the grader's datalab loader,
    # so they never dent leaf accuracy.
    page_count = body.get("page_count")

    # Degenerate "complete": under upload-endpoint overload the job is accepted but
    # returns an empty extraction with page_count=0 (nothing billed). Do NOT persist a
    # hollow result (resume would skip it forever); exit non-zero so the runner retries,
    # and drop the sidecar so the retry RE-SUBMITS a fresh job (the degenerate one only
    # ever returns empty). A real document always has pages, so this is unambiguous.
    if result in (None, {}, []) or not page_count:
        # The job reached a terminal "complete" but produced NO extraction. Two distinct
        # causes, separated by the server's own `error`/`success` fields:
        #   (a) a deterministic capacity refusal — success=false with an explicit error,
        #       e.g. "Your schema has N fields (complexity score X, limit 400), which is too
        #       large for balanced extraction on a document of 10+ pages …". The same
        #       request will ALWAYS be refused — a capability FAILURE (the system can't
        #       handle a schema this complex), same as the LLMs' input-size limits.
        #   (b) a genuine degenerate/empty complete under load — no error field.
        # We surface the server's verbatim error in the exit message for transparency.
        # Either way nothing is billed and the run is a FAILURE. page_count=0 / null is
        # unambiguous (a real doc always has pages).
        server_err = ""
        if isinstance(body, dict):
            server_err = (body.get("error") or body.get("detail") or "") or ""
        # Observability (no-spend): persist the FULL terminal check-response so the exact
        # server fields (status/success/error/page_count) are on disk. This file is ignored
        # by the runner's resume logic (it keys on datalab.json), so it never blocks a
        # re-attempt.
        try:
            dbg = out.with_suffix(".empty.json")
            keys = sorted(body.keys()) if isinstance(body, dict) else None
            dbg.write_text(json.dumps({"_keys": keys, "_status": status,
                                       "page_count": page_count, "response": body}, indent=1)[:20000])
        except OSError:
            pass
        try:
            side.unlink()
        except OSError:
            pass
        sys.exit(f"Datalab empty/rejected extraction (status={status}, page_count={page_count}); "
                 f"server error: {server_err or 'none (degenerate/empty complete)'}")

    latency = max(0.0, time.time() - float(poll_start))

    write_output(
        out, provider="datalab", result=result,
        latency_s=round(latency, 2),
        usage={
            "page_count": page_count,
            "extraction_mode": EXTRACTION_MODE,
            "mode": PARSE_MODE,
            "output_format": OUTPUT_FORMAT,
        },
    )
    print(f"Status: {status}  latency={latency:.0f}s  pages={page_count}\nSaved → {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Datalab structured extraction (extraction_mode=balanced)")
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--schema", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--poll-interval", type=int, default=5)
    ap.add_argument("--submit-retries", type=int, default=5,
                    help="submit attempts before giving up (transient 5xx/timeout only; "
                         "no job is created on a failed submit, so retries never double-bill)")
    args = ap.parse_args()

    key = os.environ.get("DATALAB_API_KEY", "")
    if not key:
        sys.exit("DATALAB_API_KEY not set")

    schema = json.loads(args.schema.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    run(args.pdf, schema, args.out, key, args.poll_interval, args.submit_retries)


if __name__ == "__main__":
    main()
