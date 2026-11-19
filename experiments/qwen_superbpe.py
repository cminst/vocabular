"""Continue Qwen's fixed byte-level BPE with gated SuperBPE merges."""

from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, perf_counter
from typing import Iterable
from unicodedata import normalize

from tokenizers import Tokenizer


BASE_MODEL = "Qwen/Qwen3.5-2B-Base"
BASE_MODEL_VOCAB_SIZE = 248_320
EXTRA_TOKENS = 32_000
BASE_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"


def _source_details(tokenizer_json: Path) -> tuple[dict, list[dict], int]:
    source = json.loads(tokenizer_json.read_text(encoding="utf-8"))
    regular = len(source["model"]["vocab"])
    specials = sorted(source.get("added_tokens", []), key=lambda item: item["id"])
    return source, specials, regular


def convert_word_model(tokenizer_json: Path, output_path: Path) -> dict:
    """Write Qwen's existing BPE as a fastboundlessbpe first-stage model.

    The regular token order and merge order are preserved exactly. The model is
    only a bridge for second-stage training; Qwen's original tokenizer JSON is
    retained as the source of truth for special tokens and inference.
    """
    raw = tokenizer_json.read_bytes()
    source = json.loads(raw)
    model = source["model"]
    if model["type"] != "BPE" or model.get("dropout") is not None:
        raise ValueError("expected a deterministic byte-level BPE tokenizer")
    if source.get("normalizer") != {"type": "NFC"}:
        raise ValueError("expected Qwen's NFC normalizer")

    pretok = source["pre_tokenizer"]
    if pretok.get("type") != "Sequence" or len(pretok["pretokenizers"]) != 2:
        raise ValueError("unexpected Qwen pre-tokenizer structure")
    split, byte_level = pretok["pretokenizers"]
    if (
        split.get("type") != "Split"
        or split.get("behavior") != "Isolated"
        or byte_level.get("type") != "ByteLevel"
        or byte_level.get("add_prefix_space")
        or byte_level.get("use_regex")
    ):
        raise ValueError("unexpected Qwen pre-tokenizer settings")
    pattern = split["pattern"].get("Regex")
    if not isinstance(pattern, str):
        raise ValueError("expected a regex pre-tokenizer")

    vocab = model["vocab"]
    tokens = [None] * len(vocab)
    for token, token_id in vocab.items():
        if not isinstance(token_id, int) or not 0 <= token_id < len(tokens):
            raise ValueError(f"invalid regular token ID: {token_id}")
        if tokens[token_id] is not None:
            raise ValueError(f"duplicate regular token ID: {token_id}")
        if not token or any(char.isspace() for char in token):
            raise ValueError(f"invalid byte-encoded token: {token!r}")
        tokens[token_id] = token
    if any(token is None for token in tokens):
        raise ValueError("regular token IDs must be contiguous")

    merges = model["merges"]
    for rank, pair in enumerate(merges):
        if not isinstance(pair, str) or pair.count(" ") != 1:
            raise ValueError(f"invalid merge at rank {rank}")
        left, right = pair.split(" ")
        if left not in vocab or right not in vocab or left + right not in vocab:
            raise ValueError(f"merge at rank {rank} is absent from the vocabulary")

    added = source.get("added_tokens", [])
    special_ids = [item["id"] for item in added]
    if len(set(special_ids)) != len(special_ids) or any(i < len(tokens) for i in special_ids):
        raise ValueError("added-token IDs overlap the regular vocabulary")
    if any("\n" in item["content"] for item in added):
        raise ValueError("special token contains a newline")

    config = {
        "tau": 1.1,
        "is_super": False,
        "superbpe_mode": False,
        "blowup": False,
        "main_regex": pattern,
        "script_specific_regex": None,
        "script_specific_scripts": None,
        "merge_pattern": r"\p{L}",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as out:
        out.write("BoundlessBPE v2 word\n")
        out.write(f"vocabulary\n{len(tokens)}\n")
        for token_id, token in enumerate(tokens):
            out.write(f"{token_id} {token} 0 0\n")
        out.write(f"special_tokens\n{len(added)}\n")
        for item in sorted(added, key=lambda item: item["id"]):
            out.write(f"{item['id']} {item['content']}\n")
        out.write("words\n")
        out.write(json.dumps(config, ensure_ascii=True, separators=(",", ":")) + "\n")
        out.write(f"merges\n{len(merges)}\n")
        for rank, pair in enumerate(merges):
            left, right = pair.split(" ")
            out.write(f"{rank} {left} {right} 0 1\n")
        out.write("deletions\n0\n")

    return {
        "source_tokenizer_sha256": sha256(raw).hexdigest(),
        "regular_vocab_size": len(tokens),
        "added_token_count": len(added),
        "merge_count": len(merges),
        "pretokenizer_regex": pattern,
        "word_model_path": str(output_path),
    }


def audit_base_tokenizer(
    tokenizer_json: Path, word_model: Path, texts: Iterable[str]
) -> dict:
    """Require native first-stage IDs to equal Hugging Face IDs on real text."""
    from boundlessbpe import FastTokenizer

    expected = Tokenizer.from_file(str(tokenizer_json))
    actual = FastTokenizer()
    actual.load(str(word_model))
    documents = 0
    utf8_bytes = 0
    token_count = 0
    for documents, text in enumerate(texts, start=1):
        reference = expected.encode(text, add_special_tokens=False).ids
        native = actual.encode(normalize("NFC", text), allowed_special="all")
        if native != reference:
            position = next(
                (i for i, (a, b) in enumerate(zip(reference, native)) if a != b),
                min(len(reference), len(native)),
            )
            raise ValueError(
                f"Qwen first-stage mismatch in document {documents}, token {position}: "
                f"HF={reference[position:position+5]}, native={native[position:position+5]}"
            )
        utf8_bytes += len(text.encode("utf-8"))
        token_count += len(native)
    if documents == 0:
        raise ValueError("base tokenizer audit received no documents")
    return {
        "documents": documents,
        "utf8_bytes": utf8_bytes,
        "token_count": token_count,
        "exact_id_match": True,
    }


class QwenSuperTokenizer:
    """Use native SuperBPE while retaining all original Qwen token IDs.

    Native SuperBPE stores regular tokens before special tokens, so its new
    regular IDs would collide with Qwen's special IDs. This adapter maps new
    regular IDs above Qwen's configured output-vocabulary size and maps native
    special IDs back to their original Qwen IDs.
    """

    def __init__(
        self,
        tokenizer_json: Path,
        super_model: Path,
        output_vocab_size: int = BASE_MODEL_VOCAB_SIZE,
    ) -> None:
        from boundlessbpe import FastTokenizer

        _, specials, regular = _source_details(tokenizer_json)
        if any(item["id"] >= output_vocab_size for item in specials):
            raise ValueError("original special token exceeds the fixed output vocabulary")
        self.native = FastTokenizer()
        self.native.load(str(super_model))
        self.regular_base = regular
        self.regular_total = self.native.get_vocab_size(False)
        if self.regular_total < regular:
            raise ValueError("SuperBPE model is smaller than Qwen's regular vocabulary")
        self.output_vocab_size = output_vocab_size
        self.new_token_count = self.regular_total - regular
        self.input_vocab_size = output_vocab_size + self.new_token_count
        self.special_to_native = {
            item["id"]: self.regular_total + i for i, item in enumerate(specials)
        }
        self.native_to_special = {v: k for k, v in self.special_to_native.items()}
        if self.native.get_vocab_size(True) != self.regular_total + len(specials):
            raise ValueError("native SuperBPE special-token count differs from Qwen")

    def encode(self, text: str, *, allowed_special: str = "all") -> list[int]:
        native_ids = self.native.encode(normalize("NFC", text), allowed_special)
        return self.map_native_ids(native_ids)

    def map_native_ids(self, native_ids: Iterable[int]) -> list[int]:
        """Map an existing native token sequence without retokenizing its text."""
        return [self._from_native(i) for i in native_ids]

    def decode(self, ids: Iterable[int]) -> str:
        return self.native.decode([self._to_native(i) for i in ids])

    def decode_bytes(self, ids: Iterable[int]) -> bytes:
        """Return exact token bytes, including pieces that are not valid UTF-8."""
        return self.native.decode_bytes([self._to_native(i) for i in ids])

    def _from_native(self, token_id: int) -> int:
        if token_id < self.regular_base:
            return token_id
        if token_id < self.regular_total:
            return self.output_vocab_size + token_id - self.regular_base
        if token_id in self.native_to_special:
            return self.native_to_special[token_id]
        raise ValueError(f"unknown native token ID: {token_id}")

    def _to_native(self, token_id: int) -> int:
        if token_id < self.regular_base:
            return token_id
        if token_id in self.special_to_native:
            return self.special_to_native[token_id]
        if self.output_vocab_size <= token_id < self.output_vocab_size + self.new_token_count:
            return self.regular_base + token_id - self.output_vocab_size
        raise ValueError(f"unused or unknown Qwen/SuperBPE token ID: {token_id}")


def _corpus(training_dir: Path) -> tuple[dict, Path]:
    metadata_path = training_dir / "train_meta.json"
    text_path = training_dir / "train.jsonl"
    if not metadata_path.is_file() or not text_path.is_file():
        raise ValueError(f"missing cached training corpus: {training_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = (metadata.get("dataset"), metadata.get("config"))
    if (
        metadata.get("format") != "jsonl-text-v1"
        or source not in {("HuggingFaceFW/fineweb", "sample-10BT"),
                          ("allenai/olmo-mix-1124", "default")}
        or (source[0] == "allenai/olmo-mix-1124"
            and metadata.get("normalization") != "NFC")
        or metadata.get("split") != "train"
        or metadata.get("text_column") != "text"
        or not isinstance(metadata.get("cached_utf8_bytes"), int)
        or metadata["cached_utf8_bytes"] < 1
        or not isinstance(metadata.get("cached_documents"), int)
        or metadata["cached_documents"] < 1
    ):
        raise ValueError("expected a cached FineWeb or NFC OLMo training corpus")
    return metadata, text_path


def corpus_sample(training_dir: Path, limit: int = 1_000) -> list[str]:
    """Read the same document prefix for a cheap exact-ID audit."""
    from itertools import islice

    _, text_path = _corpus(training_dir)
    with text_path.open(encoding="utf-8") as handle:
        return [json.loads(line)["text"] for line in islice(handle, limit)]


def prepare_training_cache(source_dir: Path, training_dir: Path) -> dict:
    """Apply Qwen's NFC normalization to the already cached FineWeb text."""
    from experiments.superbpe_resume import file_sha256

    source_meta, source_path = _corpus(source_dir)
    text_path = training_dir / "train.jsonl"
    metadata_path = training_dir / "train_meta.json"
    if text_path.is_file() and metadata_path.is_file():
        source_hash = file_sha256(source_path)
        if source_meta.get("text_sha256") not in (None, source_hash):
            raise ValueError("raw corpus checksum differs from its metadata")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("normalization") != "NFC"
            or metadata.get("source_training_cache") != source_meta
            or metadata.get("source_text_sha256") != source_hash
        ):
            raise ValueError("normalized training cache belongs to a different source")
        print(f"Using cached NFC training text: {text_path}", flush=True)
        return metadata

    training_dir.mkdir(parents=True, exist_ok=True)
    temporary_text = training_dir / "train.jsonl.part"
    source_digest = sha256()
    utf8_bytes = 0
    raw_utf8_bytes = 0
    documents = 0
    with source_path.open("rb") as source, temporary_text.open("w", encoding="utf-8") as out:
        for documents, line in enumerate(source, start=1):
            source_digest.update(line)
            try:
                raw_text = json.loads(line)["text"]
            except (json.JSONDecodeError, KeyError) as exc:
                raise ValueError(f"invalid cached JSONL document {documents:,}") from exc
            raw_utf8_bytes += len(raw_text.encode("utf-8"))
            text = normalize("NFC", raw_text)
            utf8_bytes += len(text.encode("utf-8"))
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            if documents % 250_000 == 0:
                print(
                    f"NFC normalized {documents:,} / "
                    f"{source_meta['cached_documents']:,} documents",
                    flush=True,
                )
    if (
        documents != source_meta["cached_documents"]
        or raw_utf8_bytes != source_meta["cached_utf8_bytes"]
    ):
        raise ValueError("training cache contents differ from their metadata")
    source_hash = source_digest.hexdigest()
    if source_meta.get("text_sha256") not in (None, source_hash):
        raise ValueError("raw corpus checksum differs from its metadata")
    os.replace(temporary_text, text_path)
    metadata = {
        **{key: source_meta[key] for key in ("format", "dataset", "config", "split", "text_column")},
        "byte_limit": utf8_bytes,
        "cached_utf8_bytes": utf8_bytes,
        "cached_documents": documents,
        "text_path": str(text_path),
        "normalization": "NFC",
        "source_training_cache": source_meta,
        "source_text_sha256": source_hash,
    }
    temporary_meta = training_dir / "train_meta.json.part"
    temporary_meta.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_meta, metadata_path)
    print(f"Saved NFC training cache: {utf8_bytes:,} UTF-8 bytes", flush=True)
    return metadata


def prepare_source(source_json: Path, source_dir: Path, training_dir: Path) -> dict:
    """Persist a pinned source tokenizer and verify its native BPE bridge."""
    from experiments.superbpe_resume import file_sha256

    _corpus(training_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    saved_json = source_dir / "tokenizer.json"
    word_model = source_dir / "word.model"
    source_hash = file_sha256(source_json)
    if saved_json.is_file():
        if file_sha256(saved_json) != source_hash:
            raise ValueError("cached Qwen tokenizer differs from the pinned source")
    else:
        temporary = saved_json.with_suffix(".json.part")
        shutil.copy2(source_json, temporary)
        os.replace(temporary, saved_json)
    if not word_model.is_file():
        temporary = word_model.with_suffix(".model.part")
        details = convert_word_model(saved_json, temporary)
        os.replace(temporary, word_model)
    else:
        details = convert_word_model(saved_json, word_model.with_suffix(".verify.part"))
        generated = word_model.with_suffix(".verify.part")
        try:
            if file_sha256(generated) != file_sha256(word_model):
                raise ValueError("cached Qwen word model differs from its source")
        finally:
            generated.unlink()
    if details["regular_vocab_size"] != 248_044:
        raise ValueError("unexpected Qwen regular vocabulary size")
    texts = corpus_sample(training_dir)
    texts.extend([
        "<|im_start|>assistant\nHello.<|im_end|>",
        " emojis 🙂 文本\n123",
        "cafe\u0301 and A\u030a",
    ])
    audit = audit_base_tokenizer(saved_json, word_model, texts)
    metadata = {
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "output_vocab_size": BASE_MODEL_VOCAB_SIZE,
        **details,
        "word_model_sha256": file_sha256(word_model),
        "base_parity_audit": audit,
    }
    metadata["word_model_path"] = str(word_model)
    metadata_path = source_dir / "source.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def completed_summary(
    model_dir: Path, source_dir: Path, training_dir: Path,
    extra_tokens: int = EXTRA_TOKENS,
) -> dict | None:
    """Only reuse a complete model for this exact source and corpus."""
    if not model_dir.exists():
        return None
    run_path = model_dir / "run.json"
    model_path = model_dir / "tokenizer.model"
    if not run_path.is_file() or not model_path.is_file():
        raise ValueError(f"incomplete Qwen SuperBPE model: {model_dir}")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    source = json.loads((source_dir / "source.json").read_text(encoding="utf-8"))
    corpus, _ = _corpus(training_dir)
    if (
        run.get("model_type") != "qwen_gated_superbpe"
        or run.get("supermerge_count_requested") != extra_tokens
        or run.get("source_tokenizer_sha256") != source["source_tokenizer_sha256"]
        or run.get("word_model_sha256") != source["word_model_sha256"]
        or run.get("training_cache") != corpus
        or run.get("vocab_size_actual") != source["regular_vocab_size"] + extra_tokens
    ):
        raise ValueError(f"completed Qwen SuperBPE model has a different configuration: {model_dir}")
    return run


def seed_extension(
    seed_model_dir: Path, target_model_dir: Path,
    source_dir: Path, training_dir: Path, extra_tokens: int,
) -> bool:
    """Copy a verified completed prefix into a separate resumable run."""
    from experiments.gated_superbpe import (
        FASTBOUNDLESSBPE_RESUME_PATCH, FASTBOUNDLESSBPE_REVISION,
        trainer_fingerprint,
    )
    from experiments.superbpe_resume import file_sha256, prepare_resume_dir

    seed_run_path = seed_model_dir / "run.json"
    if not seed_run_path.is_file():
        raise ValueError(f"missing completed seed model: {seed_model_dir}")
    seed_count = json.loads(seed_run_path.read_text(encoding="utf-8")).get(
        "supermerge_count_actual"
    )
    if not isinstance(seed_count, int) or not 0 < seed_count < extra_tokens:
        raise ValueError("seed must contain fewer completed SuperBPE merges than the target")
    seed_run = completed_summary(seed_model_dir, source_dir, training_dir, seed_count)
    if seed_run is None or seed_run.get("supermerge_count_actual") != seed_count:
        raise ValueError("seed model is incomplete")

    source = json.loads((source_dir / "source.json").read_text(encoding="utf-8"))
    corpus, text_path = _corpus(training_dir)
    regular = source["regular_vocab_size"]
    target_resume_dir = target_model_dir.with_name(f"{target_model_dir.name}.resume")
    prepare_resume_dir(
        target_resume_dir,
        {**corpus, "source_tokenizer_sha256": source["source_tokenizer_sha256"],
         "word_model_sha256": source["word_model_sha256"]},
        text_path, regular + extra_tokens, regular,
        FASTBOUNDLESSBPE_REVISION, FASTBOUNDLESSBPE_RESUME_PATCH,
        trainer_fingerprint(),
    )
    source_resume_dir = seed_model_dir.with_name(f"{seed_model_dir.name}.resume")
    source_manifest_path = source_resume_dir / "manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError(f"missing seed resume manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    target_manifest = json.loads(
        (target_resume_dir / "manifest.json").read_text(encoding="utf-8")
    )
    if (
        source_manifest.get("total_vocab_size") != regular + seed_count
        or any(
            source_manifest.get(key) != value
            for key, value in target_manifest.items()
            if key != "total_vocab_size"
        )
    ):
        raise ValueError("seed checkpoint belongs to a different corpus or trainer")

    filename = f"tokenizer_{regular + seed_count}.model"
    source_checkpoint = source_resume_dir / filename
    target_checkpoint = target_resume_dir / filename
    if not source_checkpoint.is_file():
        raise ValueError(f"missing final seed checkpoint: {source_checkpoint}")
    if target_checkpoint.is_file():
        if file_sha256(target_checkpoint) != file_sha256(source_checkpoint):
            raise ValueError("target seed checkpoint differs from the completed model")
        return False
    temporary = target_resume_dir / f"{filename}.part"
    shutil.copy2(source_checkpoint, temporary)
    os.replace(temporary, target_checkpoint)
    print(f"Seeded {seed_count:,} completed SuperBPE merges from {seed_model_dir}", flush=True)
    return True


def train_qwen_superbpe(
    training_dir: Path, source_dir: Path, output_dir: Path, resume_dir: Path,
    extra_tokens: int = EXTRA_TOKENS,
) -> dict:
    """Train only second-stage gated merges, using Qwen's fixed BPE as stage one."""
    from boundlessbpe import FastSuperBpeTrainer, FastTokenizer
    from experiments.gated_superbpe import (
        FASTBOUNDLESSBPE_RESUME_PATCH, FASTBOUNDLESSBPE_REVISION,
        trainer_fingerprint,
    )
    from experiments.superbpe_resume import (
        CHECKPOINT_INTERVAL, latest_checkpoint, prepare_resume_dir,
    )

    if extra_tokens < 1:
        raise ValueError("extra token count must be positive")
    corpus, text_path = _corpus(training_dir)
    source = json.loads((source_dir / "source.json").read_text(encoding="utf-8"))
    word_path = source_dir / "word.model"
    tokenizer_json = source_dir / "tokenizer.json"
    regular = source["regular_vocab_size"]
    total = regular + extra_tokens
    if source["word_model_path"] != str(word_path):
        raise ValueError("source metadata points to a different word model")
    from experiments.superbpe_resume import file_sha256
    if (
        file_sha256(word_path) != source["word_model_sha256"]
        or file_sha256(tokenizer_json) != source["source_tokenizer_sha256"]
    ):
        raise ValueError("Qwen source files changed after preparation")
    prepare_resume_dir(
        resume_dir,
        {**corpus, "source_tokenizer_sha256": source["source_tokenizer_sha256"],
         "word_model_sha256": source["word_model_sha256"]},
        text_path, total, regular,
        FASTBOUNDLESSBPE_REVISION, FASTBOUNDLESSBPE_RESUME_PATCH,
        trainer_fingerprint(),
    )
    resume_model, resumed_from = latest_checkpoint(resume_dir, regular, total)
    options = {"resume_model_file": str(resume_model)} if resume_model else {}
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / "tokenizer"
    started = perf_counter()
    print(
        f"Training {extra_tokens:,} gated SuperBPE merges on Qwen's "
        f"{regular:,} regular tokens from {corpus['cached_utf8_bytes']:,} bytes; "
        f"resuming after {resumed_from:,} merges",
        flush=True,
    )
    trainer = FastSuperBpeTrainer(main_regex=source["pretokenizer_regex"])
    trainer.train(
        filepath=str(text_path), outprefix=str(prefix),
        num_lines=corpus["cached_documents"], vocab_size=extra_tokens,
        recalc=0, word_model_file=str(word_path),
        max_bytes=corpus["cached_utf8_bytes"],
        checkpoint_iterations=CHECKPOINT_INTERVAL,
        verbose=True, progress_interval=512,
        save_pretokens=None, greedy_split=False,
        min_count=15, max_ngram_len=30, **options,
    )
    trainer.save(str(prefix))
    model_path = output_dir / "tokenizer.model"
    native = FastTokenizer()
    native.load(str(model_path))
    if native.get_vocab_size(False) != total:
        raise ValueError("SuperBPE stopped before all requested merges were learned")
    adapter = QwenSuperTokenizer(tokenizer_json, model_path)
    if adapter.new_token_count != extra_tokens:
        raise ValueError("Qwen adapter exposes the wrong number of new tokens")
    _, specials, _ = _source_details(tokenizer_json)
    for item in specials:
        if adapter.encode(item["content"]) != [item["id"]]:
            raise ValueError(f"original Qwen added token changed ID: {item['id']}")
    if not adapter.decode_bytes([BASE_MODEL_VOCAB_SIZE]):
        raise ValueError("first new SuperBPE token has no bytes")
    samples = corpus_sample(training_dir, limit=256)
    samples.extend([
        "<|im_start|>assistant\nTesting new SuperBPE.🙂<|im_end|>",
        "cafe\u0301 and A\u030a",
    ])
    base = Tokenizer.from_file(str(tokenizer_json))
    validation = {
        "documents": len(samples), "utf8_bytes": 0,
        "base_token_count": 0, "super_token_count": 0,
        "new_token_occurrences": 0, "exact_nfc_roundtrip": True,
    }
    for sample in samples:
        ids = adapter.encode(sample)
        reference = base.encode(sample, add_special_tokens=False).ids
        if adapter.decode(ids) != normalize("NFC", sample):
            raise ValueError("Qwen SuperBPE round-trip failed")
        if len(ids) > len(reference):
            raise ValueError("new SuperBPE model increased token count on a validation document")
        validation["utf8_bytes"] += len(sample.encode("utf-8"))
        validation["base_token_count"] += len(reference)
        validation["super_token_count"] += len(ids)
        validation["new_token_occurrences"] += sum(i >= BASE_MODEL_VOCAB_SIZE for i in ids)
    summary = {
        "model_type": "qwen_gated_superbpe",
        "base_model": BASE_MODEL,
        "base_revision": BASE_REVISION,
        "source_tokenizer_sha256": source["source_tokenizer_sha256"],
        "word_model_sha256": source["word_model_sha256"],
        "training_cache": corpus,
        "supermerge_count_requested": extra_tokens,
        "supermerge_count_actual": adapter.new_token_count,
        "base_regular_vocab_size": regular,
        "base_output_vocab_size": BASE_MODEL_VOCAB_SIZE,
        "vocab_size_actual": total,
        "input_vocab_size": BASE_MODEL_VOCAB_SIZE + extra_tokens,
        "output_vocab_size": BASE_MODEL_VOCAB_SIZE,
        "new_token_id_range": [BASE_MODEL_VOCAB_SIZE, BASE_MODEL_VOCAB_SIZE + extra_tokens - 1],
        "source_tokenizer_path": str(tokenizer_json),
        "tokenizer_path": str(model_path),
        "resumed_from_supermerges": resumed_from,
        "checkpoint_interval": CHECKPOINT_INTERVAL,
        "resume_dir": str(resume_dir),
        "validation": validation,
        "implementation": {
            "repository": "https://github.com/kensho-technologies/fastboundlessbpe",
            "revision": FASTBOUNDLESSBPE_REVISION,
            "fingerprint": trainer_fingerprint(),
        },
        "training_seconds": perf_counter() - started,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    (output_dir / "run.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def run_training(
    training_dir: Path, source_dir: Path, model_dir: Path,
    status_path: Path, commit, extra_tokens: int = EXTRA_TOKENS,
    seed_model_dir: Path | None = None,
) -> dict:
    """Keep the Rust trainer outside Modal's heartbeat process and save snapshots."""
    from experiments.gated_superbpe import publish_completed_model
    from experiments.superbpe_resume import publish_checkpoints
    from experiments.superbpe_runner import process_rss_kib, write_status

    cached = completed_summary(model_dir, source_dir, training_dir, extra_tokens)
    if cached is not None:
        print(f"Using completed Qwen SuperBPE model: {model_dir}", flush=True)
        return cached
    if seed_model_dir is not None and seed_extension(
        seed_model_dir, model_dir, source_dir, training_dir, extra_tokens
    ):
        commit()
    resume_dir = model_dir.with_name(f"{model_dir.name}.resume")
    with TemporaryDirectory(prefix="qwen-superbpe-") as scratch:
        staged = Path(scratch) / "model"
        command = [
            sys.executable, "-m", "experiments.qwen_superbpe", "train",
            str(training_dir), str(source_dir), str(staged), str(resume_dir),
            str(extra_tokens),
        ]
        process = subprocess.Popen(command)
        started = monotonic()
        last_commit = started
        peak_rss_kib = 0
        published = {p.name for p in resume_dir.glob("tokenizer_*.model")}
        try:
            while True:
                try:
                    returncode = process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    returncode = None
                rss = process_rss_kib(process.pid)
                peak_rss_kib = max(peak_rss_kib, rss or 0)
                new = (
                    publish_checkpoints(staged, resume_dir, published)
                    if staged.is_dir() and resume_dir.is_dir()
                    else []
                )
                published.update(new)
                status = {
                    "state": "running" if returncode is None else (
                        "completed" if returncode == 0 else "failed"
                    ),
                    "elapsed_seconds": round(monotonic() - started, 1),
                    "current_rss_kib": rss,
                    "sampled_peak_rss_kib": peak_rss_kib,
                    "checkpoint_files": sorted(published),
                    "trainer_returncode": returncode,
                }
                write_status(status_path, status)
                if new or monotonic() - last_commit >= 60 or returncode is not None:
                    commit()
                    last_commit = monotonic()
                    print(
                        f"Qwen SuperBPE: {status['elapsed_seconds']:.0f}s, "
                        f"RSS {rss or 0:,} KiB, checkpoints {len(published)}",
                        flush=True,
                    )
                if returncode is not None:
                    if returncode:
                        raise subprocess.CalledProcessError(returncode, command)
                    break
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
        summary = publish_completed_model(staged, model_dir)
        status.update({"model_dir": str(model_dir), "training_seconds": summary["training_seconds"]})
        write_status(status_path, status)
        commit()
        return summary


if __name__ == "__main__":
    if len(sys.argv) != 7 or sys.argv[1] != "train":
        raise SystemExit(
            "usage: python -m experiments.qwen_superbpe "
            "train TRAIN_DIR SOURCE_DIR OUTPUT_DIR RESUME_DIR EXTRA_TOKENS"
        )
    train_qwen_superbpe(
        *(Path(arg) for arg in sys.argv[2:6]), extra_tokens=int(sys.argv[6])
    )
