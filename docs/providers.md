# Providers

Each of the seven systems is driven by a standalone runner in
`src/longextract_bench/providers/`. Every runner takes the same three arguments and
writes the same `{result, _meta}` envelope:

```bash
python -m longextract_bench.providers.<system> \
    --pdf document.pdf --schema schema.json --out out.json
```

Every system runs in its **highest-accuracy configuration**. Where a vendor's API
forbids a setting we would otherwise standardize (e.g. pinning temperature to 0), the
constraint is API-inherent and is documented per system below — it is not a methodology
choice. Each system's schema-dialect adapter is noted per system below; every adapter
is a pure translation that changes no field, description, enum, or type-category.

| System | Module | API surface | Default model / tier | Auth |
| --- | --- | --- | --- | --- |
| Reducto | `reducto` | `/extract_async` (upload → submit → poll) | deep extract, model `v2` | `REDUCTO_API_KEY` |
| Extend | `extend` | `extend-ai` SDK `extract_runs` | `extraction_performance` processor | `EXTEND_API_KEY` |
| LlamaExtract | `llamaextract` | LlamaCloud v2 `/extract` | `agentic` tier | `LLAMA_CLOUD_API_KEY` |
| OpenAI | `openai` | `responses` API | `gpt-5.5-pro` | `OPENAI_API_KEY` |
| Anthropic Claude | `claude` | Messages API (streaming) | `claude-opus-4-8` | `ANTHROPIC_API_KEY` |
| Google Gemini | `gemini` | `generateContent` | `gemini-3.1-pro-preview` | `GEMINI_API_KEY` |
| Datalab | `datalab` | `/extract` | `balanced` tier | `DATALAB_API_KEY` |

Model IDs and tiers are the runner defaults at the time of writing; they are overridable
and should be recorded at run time, since scores are point-in-time
([`methodology.md`](methodology.md) §6).

---

## Reducto

* **Mode:** uploads the PDF, submits an `/extract_async` job, and polls to completion.
* **Max-accuracy config:** `deep_extract` on, `deep_extract_model = v2`, agentic table
  enrichment on, citations off, empty system prompt (the schema field descriptions drive
  extraction — parity with the prompt-driven systems).
* **Schema:** passed directly; no adapter needed.
* **Latency:** the server-reported job `duration` (API processing only — excludes upload,
  queue wait, and poll sleeps).
* **Resume:** the job id is persisted to a sidecar, so an interrupted run re-polls rather
  than re-submitting (no double-billing).

## Extend

* **Mode:** `extend-ai` SDK — upload, `extract_runs.create`, poll.
* **Max-accuracy config:** `base_processor = "extraction_performance"`,
  `array_strategy = large_array_max_context` (the large-array mode for long documents).
* **Schema:** adapted to Extend's dialect (`$ref` inlined, `additionalProperties:false`,
  nullable primitives, reserved-key `id` aliasing) — a pure translation.
* **Latency:** `updatedAt − createdAt` (server-side processing span).
* **Resume:** the run id is persisted to a sidecar.

## LlamaExtract

* **Mode:** uploads the PDF bytes (never a URL — no source domain leaks to the vendor),
  submits a stateless v2 extraction job, polls to completion.
* **Max-accuracy config:** `tier = "agentic"`, the highest tier available on a standard
  key. The premium `agentic_plus` tier is rejected with HTTP 422 on a standard key, so
  `agentic` is the maxed-out mode here.
* **Schema:** adapted (`$ref` inlined, type-unions collapsed so array branches keep their
  `items`).
* **Latency:** `updated_at − created_at` (server-side processing span).
* **Resume:** the job id + project id are persisted to a sidecar.

## OpenAI

* **Mode:** `responses` API with the PDF sent as a base64 `input_file`.
* **Max-accuracy config:** `reasoning.effort = "high"`; native `json_schema` response
  format (`strict:false`); no output-token cap.
* **API-inherent constraint:** `temperature` is rejected by this model entirely, so —
  unlike Gemini — it cannot be pinned to 0.
* **No silent truncation:** an `incomplete` run (e.g. token-limited) is a hard failure,
  not a truncated result.
* **Latency:** the server-reported `openai-processing-ms` header, falling back to a tight
  client timer around the single request.

## Anthropic Claude

* **Mode:** Messages API (streaming, required at this output size); PDF uploaded via the
  Files API beta.
* **Max-accuracy config:** extended thinking via `thinking.type = "adaptive"` (the only
  thinking mode this model accepts, and the max-reasoning equivalent — the model scales
  its own thinking for hard tasks); schema bound as the `emit_extraction` tool's
  `input_schema` so output stays schema-shaped; output ceiling raised to 128K via the
  128K-output beta.
* **API-inherent constraints (documented for fairness, not methodology choices):**
  extended thinking forbids a forced `tool_choice`, so `tool_choice = "auto"` is used (the
  model still reliably calls the tool; a turn with no tool call is a hard failure); and
  `temperature` may only be 1 when thinking is on, so it is left unset.
* **No silent truncation:** a `max_tokens` stop is a hard failure.

## Google Gemini

* **Mode:** uploads the PDF via the Files API, then a single non-streaming
  `generateContent` call.
* **Max-accuracy config:** native `response_schema` (constrained decoding — parity with
  OpenAI's `json_schema`); `temperature = 0`; no output-token cap.
* **Schema:** adapted to Gemini's OpenAPI subset (`nullable:true` dialect).
* **No silent truncation:** a `MAX_TOKENS` finish reason is a hard failure.
* **Latency:** a client timer around the single `generateContent` request (Gemini exposes
  no server-side timing field).

## Datalab

* **Mode:** submits an `/extract` job and polls to completion.
* **Max-accuracy config:** `extraction_mode = "balanced"` (the highest extraction tier,
  vs `fast`), `parse_mode = "balanced"`, markdown parse output.
* **Schema:** adapted to Datalab's dialect.
* **Error handling:** any submit-time HTTP rejection, post-accept "job failed", or
  empty/degenerate result exits non-zero, so the run is recorded as a failure (see
  [`methodology.md`](methodology.md) §5). The `*_citations`/`*_meta` sidecars Datalab
  emits are preserved in the stored output and dropped only at scoring time.
* **Resume:** the job id is persisted to a sidecar.

---

## Shared behavior

* **One envelope.** Every runner writes `{result, _meta:{provider, latency_s, usage}}`
  via `envelope.write_output`, so the grader reads all seven uniformly.
* **Resumable & idempotent.** Providers with persistent server-side jobs (Reducto,
  Extend, LlamaExtract, Datalab) persist their job id to a `*.runid.json` sidecar and
  re-poll on retry instead of re-submitting — so a retried timeout never re-bills a
  generation.
* **Loud, not silent.** Token-limited / truncated / incomplete responses are treated as
  hard failures and discarded, never scored as if complete.
