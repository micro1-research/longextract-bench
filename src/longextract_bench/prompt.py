"""The single extraction prompt shared by every prompt-driven provider.

The prompt MUST be identical across providers so no model gets a wording
advantage. The schema's own field descriptions carry all task-specific
instruction, matching the prompt-free, schema-native providers (Reducto,
Extend).
"""

from __future__ import annotations

import json


def build_extract_prompt(schema: dict) -> str:
    """Build the extraction prompt for a given JSON Schema.

    Schema-only guidance: the field descriptions carry all the instruction, so a
    prompt-driven model (OpenAI, Gemini, Claude) receives the same task framing as
    a schema-native one (Reducto, Extend).
    """
    return (
        "Extract structured data from the attached PDF document according to the "
        "following JSON Schema:\n\nSchema:\n" + json.dumps(schema, indent=2)
    )
