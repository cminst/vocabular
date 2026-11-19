"""Read the shared OLMo 2 mix in reproducible shuffled shard order."""

from __future__ import annotations

from contextlib import contextmanager
import gzip
import io
import json
from pathlib import Path
import random
from unicodedata import normalize

import zstandard


# Approximate OLMo 2 November 2024 mix, in billions of source tokens.
SOURCE_WEIGHTS = {
    "dclm": 3_700,
    "starcoder": 83,
    "pes2o": 58.6,
    "arxiv": 20.8,
    "open-web-math": 12.2,
    "algebraic-stack": 11.8,
    "wiki": 3.66,
}


@contextmanager
def open_shard(path: Path):
    if path.name.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            yield stream
    else:
        with path.open("rb") as raw:
            with zstandard.ZstdDecompressor().stream_reader(raw) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as stream:
                    yield stream


def documents(root: Path, source: str, seed: int = 42):
    """Yield NFC text without caching raw dataset shards."""
    directory = root / "data" / source
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    files = sorted(path for path in directory.rglob("*")
                   if path.name.endswith((".json.gz", ".jsonl.zstd")))
    if not files:
        raise ValueError(f"no JSON shards found for {source}: {directory}")
    random.Random(f"{seed}:{source}").shuffle(files)
    for path in files:
        with open_shard(path) as stream:
            for line in stream:
                record = json.loads(line)
                text = record.get("text")
                if isinstance(text, str) and text:
                    yield normalize("NFC", text)
