"""Concatenate built token corpora into one training stream (application tracks: domain + general text)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def concat_corpora(parts: Sequence[Path], out_dir: Path, *, extra_manifest: dict[str, Any] | None = None,
                   reuse: bool = False) -> dict[str, Any]:
    """Join built corpora (same tokenizer, alias table, dtype and EOS) into one corpus directory.

    Tokens are appended in order and span positions shifted, so a domain part and a general-text
    part become one training stream whose mix by tokens is exact; uniform window sampling then
    draws from the parts in proportion to their sizes. The manifest lists every part's tokens."""
    manifests = [json.loads((Path(p) / "manifest.json").read_text()) for p in parts]
    if reuse and (out_dir / "manifest.json").exists():
        existing = json.loads((out_dir / "manifest.json").read_text())
        same_parts = [(q["path"], q["tokens"]) for q in existing.get("parts", [])] == \
            [(str(p), m["tokens"]) for p, m in zip(parts, manifests)]
        if same_parts and existing.get("alias_table_sha256") == manifests[0]["alias_table_sha256"]:
            return existing          # built earlier from the same parts: resume past it
    for key in ("tokenizer", "alias_table_sha256", "dtype", "eos_id", "boundary"):
        if len({str(m.get(key)) for m in manifests}) != 1:
            raise ValueError(f"cannot concatenate corpora that differ in {key}")
    out_dir.mkdir(parents=True, exist_ok=True)
    offset = 0
    arrays: dict[str, list[np.ndarray]] = {}
    with (out_dir / "tokens.bin").open("wb") as handle:
        for part in parts:
            with (Path(part) / "tokens.bin").open("rb") as source:
                for chunk in iter(lambda: source.read(1 << 24), b""):
                    handle.write(chunk)
            with np.load(Path(part) / "spans.npz") as data:
                for key in data.files:
                    values = data[key]
                    if key in ("start", "end", "inject"):
                        values = (values.astype(np.int64) + offset).astype(values.dtype)
                    arrays.setdefault(key, []).append(values)
            offset += json.loads((Path(part) / "manifest.json").read_text())["tokens"]
    if offset >= 2**31:
        raise ValueError("corpus too large for int32 span positions")
    np.savez(out_dir / "spans.npz", **{k: np.concatenate(v) for k, v in arrays.items()})
    first = manifests[0]
    manifest = {"tokens": int(offset), "documents": sum(m["documents"] for m in manifests), "dtype": first["dtype"],
                "tokenizer": first["tokenizer"], "tokenizer_revision": first.get("tokenizer_revision"),
                "alias_table_sha256": first["alias_table_sha256"], "boundary": first["boundary"],
                "spans": int(sum(m["spans"] for m in manifests)), "eos_id": first["eos_id"],
                "skipped_documents": sum(m.get("skipped_documents", 0) for m in manifests),
                "parts": [{"path": str(p), "tokens": m["tokens"], "documents": m["documents"]} for p, m in zip(parts, manifests)],
                **(extra_manifest or {})}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
