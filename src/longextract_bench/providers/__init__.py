"""The production extraction systems evaluated by LongExtractBench.

Each provider is a standalone module runnable as
`python -m longextract_bench.providers.<name> --pdf … --schema … --out …`.
All systems are peers; the order here is not a ranking.
"""

PROVIDERS = (
    "reducto",
    "extend",
    "llamaextract",
    "openai",
    "claude",
    "gemini",
    "datalab",
)
