"""SemCor (Raganato et al. 2017 unified format) sense-tagged sentences for E1.

Each instance gives the sentence text, the character span of the tagged word, its lemma/POS and
its WordNet synset (from the sense key).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SenseInstance:
    sentence_id: str
    text: str
    start: int
    end: int
    lemma: str
    pos: str
    synset: str


def load_semcor(root: Path, wordnet: Any) -> list[SenseInstance]:
    base = root / "Training_Corpora" / "SemCor"
    keys: dict[str, str] = {}
    for line in (base / "semcor.gold.key.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            keys[parts[0]] = parts[1]          # first gold key
    instances: list[SenseInstance] = []
    synset_of: dict[str, str | None] = {}
    for _, sentence in ET.iterparse(base / "semcor.data.xml", events=("end",)):
        if sentence.tag != "sentence":
            continue
        words, position, tagged = [], 0, []
        for token in sentence:
            text = token.text or ""
            start = position + (1 if words else 0)
            words.append(text)
            position = start + len(text)
            if token.tag == "instance" and token.get("id") in keys:
                tagged.append((token.get("id"), start, position, token.get("lemma"), token.get("pos")))
        text = " ".join(words)
        for instance_id, start, end, lemma, pos in tagged:
            key = keys[instance_id]
            if key not in synset_of:
                try:
                    synset_of[key] = wordnet.lemma_from_key(key).synset().name()
                except Exception:
                    synset_of[key] = None
            if synset_of[key] is not None:
                instances.append(SenseInstance(sentence.get("id"), text, start, end, lemma, pos, synset_of[key]))
        sentence.clear()
    return instances
