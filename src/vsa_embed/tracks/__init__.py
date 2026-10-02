"""Application tracks T2–T6 (E8): per-track ontology, domain documents, synthetic concepts and items.

A track object is what `experiments/track_corpus.py` drives; every method is deterministic given the
config. `prepare` materialises the domain documents as `docs/{eval,train}.jsonl.gz` under the track's
data root (downloads are separate and recorded in `~/data/vsa-llm/DATA_SOURCES.md`).
"""

from __future__ import annotations

import gzip
import importlib
import json
from pathlib import Path
from typing import Any, Iterator, Sequence

from ..ontologies.wordnet import FrameOntology
from .common import SyntheticConcept

TRACKS = {
    "t2": "vsa_embed.tracks.devtools:DevToolsTrack",
    "t3": "vsa_embed.tracks.product:ProductTrack",
    "t4": "vsa_embed.tracks.chemistry:ChemistryTrack",
    "t5": "vsa_embed.tracks.glossary:GlossaryTrack",
    "t6": "vsa_embed.tracks.legal:LegalTrack",
}


class Track:
    """Interface of one application track (subclasses override what they need)."""

    name = "track"

    def __init__(self, config: dict[str, Any], data_root: Path) -> None:
        self.config, self.data_root = config, data_root
        self.docs_dir = data_root / "docs"

    # documents ---------------------------------------------------------------------------------
    def prepare(self) -> dict[str, Any]:
        """Write `docs/eval.jsonl.gz` and `docs/train.jsonl.gz`; return a summary."""
        raise NotImplementedError

    def documents(self, split: str) -> Iterator[str]:
        yield from read_jsonl_texts(self.docs_dir / f"{split}.jsonl.gz")

    # ontology ----------------------------------------------------------------------------------
    def ontology(self) -> FrameOntology:
        raise NotImplementedError

    def fixed_holdout(self, ontology: FrameOntology) -> list[int] | None:
        """Concept indices of a holdout fixed by construction (None: choose by presample counts)."""
        return None

    def synthetic(self, ontology: FrameOntology, forbidden: set[str]) -> list[SyntheticConcept]:
        """Invented concepts for the E5.4 zero-shot scenario (frames only, no text)."""
        return []

    # items -------------------------------------------------------------------------------------
    def items(self, context: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        """Task items by file stem. `context` has the ontology, holdout, synthetic concepts and
        per-concept training frequencies (see `track_corpus.run`)."""
        return {}


def load_track(name: str, config: dict[str, Any], data_root: Path) -> Track:
    module, _, cls = TRACKS[name].partition(":")
    return getattr(importlib.import_module(module), cls)(config, data_root)


def read_jsonl_texts(path: Path, *, field: str = "text") -> Iterator[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)[field]


def write_jsonl_texts(path: Path, texts: Sequence[str] | Iterator[str]) -> dict[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    documents = chars = 0
    with gzip.open(str(path) + ".part", "wt", encoding="utf-8") as handle:
        for text in texts:
            handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            documents += 1; chars += len(text)
    Path(str(path) + ".part").rename(path)
    return {"documents": documents, "chars": chars}
