"""Raw files of the toolkit benchmark sets (decision 63, WP TK-B): pinned sources, download, integrity record.

Every set is fetched over HTTPS from a pinned commit or dataset revision into `~/data/vsa-llm/benchmarks/<name>/raw/`
(one new directory per set; nothing here executes downloaded content) and recorded in `<name>/source.json`: URL, size,
sha256 (computed twice, both passes must agree), the licence as stated by the source, and the retrieval date. The
adapters (`vsa_embed.benchmarks.adapters`) read only these files.

| Set | Source | Licence (as stated by the source) | Committed to the repository |
|---|---|---|---|
| `comps` | GitHub `kanishkamisra/comps` (COMPS, Misra, Rayz & Ettinger, EACL 2023) | Apache-2.0 (`LICENSE`) | derived items (attribution) |
| `alcuna` | Google Drive folder of GitHub `Arvid-pku/ALCUNA` (Yin, Huang & Wan, EMNLP 2023) | MIT (repository `LICENSE`) | derived items (attribution) |
| `entity-inferences` | GitHub `yasumasaonoe/entity_knowledge_propagation` (Onoe et al., ACL 2023) | **none stated** (no licence file) | digests only |
| `reversal` | Hugging Face `lberglund/reversal_curse` (Berglund et al., ICLR 2024) | MIT (dataset card) | digests only |
| `lre` | GitHub `evandez/relations` (Hernandez et al., ICLR 2024) | MIT (`LICENSE`) | digests only |
| `bear` | Hugging Face `lm-pub-quiz/BEAR` (Wiland, Ploner & Akbik, Findings NAACL 2024) | CC BY-SA 4.0 (dataset card) | digests only |
| `popqa` | Hugging Face `akariasai/PopQA` (Mallen et al., ACL 2023) | none on the card (code repository MIT) | digests only |

    python -m vsa_embed.benchmarks.sources fetch [--sets comps alcuna ...] [--root ~/data/vsa-llm/benchmarks]
    python -m vsa_embed.benchmarks.sources verify [--sets ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

ROOT = Path("~/data/vsa-llm/benchmarks").expanduser()
USER_AGENT = "vsa-llm-research (TK-B benchmark adapters; contact via repository owner)"
GH = "https://raw.githubusercontent.com"
HF = "https://huggingface.co/datasets"


@dataclass(frozen=True)
class Source:
    name: str
    title: str
    citation: str
    licence: str
    licence_evidence: str
    commit: str                                   # git commit or dataset revision the URLs are pinned to
    files: dict[str, str] = field(default_factory=dict)     # relative path under raw/ → URL
    commit_items: bool = False                    # derived items may be committed (licence allows redistribution)


def _gh(repo: str, commit: str, paths: Sequence[str]) -> dict[str, str]:
    return {p: f"{GH}/{repo}/{commit}/{p}" for p in paths}


def _hf(dataset: str, revision: str, paths: Sequence[str]) -> dict[str, str]:
    return {p: f"{HF}/{dataset}/resolve/{revision}/{p}" for p in paths}


COMPS_COMMIT = "5a9dc6c403b32a276c80a8ccef9956faf50ed074"
ALCUNA_COMMIT = "963d6f9d8a7e877cdc714d55b0d2522332a2e6c3"
EKP_COMMIT = "938295c1ead0622350f2e6c9d6122964af123e05"
LRE_COMMIT = "1b9ec3cf2b8368e42bde7e80fcef384312a6ec07"
REVERSAL_REVISION = "2b0a35c0f1730a60957319f9b25f4858e7012e44"
BEAR_REVISION = "bf98c22bd0d4211e0fbd3fb23f12e9ec9cb193f0"
POPQA_REVISION = "098765c79ea10a2cb19c828324e33281b8336ec0"
# ALCUNA's data live in the Google Drive folder its README links (1P2Yt4XM-uSzfJoec4psIhpk-mfm-K3R1); file ids below.
ALCUNA_DRIVE = {"meta_data.jsonl": "1kolOjXhS5AWI20RnwpA--xZf2ghojCxB", "id2question.json": "19xjgOuFZe7WdAglX71OgUJXJoqDnPUzp"}

LRE_RELATIONS = {
    "factual": ["city_in_country", "company_ceo", "company_hq", "country_capital_city", "country_currency", "country_language",
                "country_largest_city", "food_from_country", "landmark_in_country", "landmark_on_continent",
                "person_band_lead_singer", "person_father", "person_mother", "person_native_language", "person_occupation",
                "person_plays_instrument", "person_plays_position_in_sport", "person_plays_pro_sport", "person_university",
                "pokemon_evolutions", "presidents_birth_year", "presidents_election_year", "product_by_company",
                "star_constellation", "superhero_archnemesis", "superhero_person"],
    "commonsense": ["fruit_inside_color", "fruit_outside_color", "object_superclass", "substance_phase", "task_done_by_person",
                    "task_done_by_tool", "word_sentiment", "work_location"],
    "linguistic": ["adj_antonym", "adj_comparative", "adj_superlative", "verb_past_tense", "word_first_letter", "word_last_letter"],
    "bias": ["characteristic_gender", "degree_gender", "name_birthplace", "name_gender", "name_religion", "occupation_age",
             "occupation_gender"],
}
REVERSAL_FILES = ["README.md"] + [f"name_description_dataset/{f}.jsonl" for f in (
    "all", "all_prompts_train", "both_prompts_test", "both_prompts_train", "d2p_prompts_test", "d2p_prompts_train",
    "d2p_reverse_prompts_test", "d2p_reverse_prompts_test_randomized", "p2d_prompts_test", "p2d_prompts_train",
    "p2d_reverse_prompts_test", "p2d_reverse_prompts_test_randomized", "unrealized_examples", "validation_prompts")]
EI_FILES = [f"data/entity_inferences/{f}.json" for f in (
    "disaster_explicit_attribute_independent", "disaster_implicit_attribute_independent",
    "fake_person_explicit_attribute_dependent", "fake_person_implicit_attribute_dependent_adjective",
    "tv_show_explicit_attribute_dependent", "tv_show_implicit_attribute_dependent_adjective")]

SOURCES: dict[str, Source] = {
    "comps": Source(
        "comps", "COMPS (Conceptual Minimal Pair Sentences): COMPS-WUGS",
        "Misra, Rayz & Ettinger (2023). COMPS: Conceptual Minimal Pair Sentences for testing Robust Property Knowledge and "
        "its Inheritance in Pre-trained Language Models. EACL 2023.",
        "Apache-2.0", "repository LICENSE (Apache License 2.0, Copyright 2022 Kanishka Misra); HF card kanishka/comps: apache-2.0",
        COMPS_COMMIT,
        _gh("kanishkamisra/comps", COMPS_COMMIT, ["LICENSE", "README.md", "data/README.md", "data/comps/comps_wugs.jsonl",
                                                  "data/comps/comps_base.jsonl", "data/concept_senses.csv",
                                                  "data/pseudowords.txt"]),
        commit_items=True),
    "alcuna": Source(
        "alcuna", "ALCUNA: artificial entities from the EOL taxonomy (KnowGen)",
        "Yin, Huang & Wan (2023). ALCUNA: Large Language Models Meet New Knowledge. EMNLP 2023.",
        "MIT", "repository LICENSE (MIT, Copyright 2023 ArvidYin); the data are linked from the repository README", ALCUNA_COMMIT,
        {**_gh("Arvid-pku/ALCUNA", ALCUNA_COMMIT, ["LICENSE", "README.md"]),
         **{f"dataset/{name}": f"https://drive.usercontent.google.com/download?id={fid}&export=download&confirm=t"
            for name, fid in ALCUNA_DRIVE.items()}},
        commit_items=True),
    "entity-inferences": Source(
        "entity-inferences", "Entity Inferences (and ECBD) — learning new entities from descriptions",
        "Onoe, Zhang, Padmanabhan, Durrett & Choi (2023). Can LMs Learn New Entities from Descriptions? Challenges in "
        "Propagating Injected Knowledge. ACL 2023.",
        "none stated", "no LICENSE file in the repository and no licence in its README (checked 2026-10-08): research use, "
        "not redistributed", EKP_COMMIT,
        _gh("yasumasaonoe/entity_knowledge_propagation", EKP_COMMIT, ["README.md", *EI_FILES])),
    "reversal": Source(
        "reversal", "Reversal curse: fictitious name–description pairs",
        "Berglund, Tong, Kaufmann, Balesni, Stickland, Korbak & Evans (2024). The Reversal Curse: LLMs trained on "
        "\"A is B\" fail to learn \"B is A\". ICLR 2024.",
        "MIT", "Hugging Face dataset card lberglund/reversal_curse: license mit", REVERSAL_REVISION,
        _hf("lberglund/reversal_curse", REVERSAL_REVISION, REVERSAL_FILES)),
    "lre": Source(
        "lre", "LRE relations (linear relational embeddings)",
        "Hernandez, Sen Sharma, Haklay, Meng, Wattenberg, Andreas, Belinkov & Bau (2024). Linearity of Relation Decoding "
        "in Transformer Language Models. ICLR 2024.",
        "MIT", "repository LICENSE (MIT)", LRE_COMMIT,
        _gh("evandez/relations", LRE_COMMIT, ["LICENSE", "README.md",
                                              *[f"data/{kind}/{name}.json" for kind, names in LRE_RELATIONS.items() for name in names]])),
    "bear": Source(
        "bear", "BEAR (lm-pub-quiz): relational knowledge by ranking statements",
        "Wiland, Ploner & Akbik (2024). BEAR: A Unified Framework for Evaluating Relational Knowledge in Causal and Masked "
        "Language Models. Findings of NAACL 2024.",
        "CC-BY-SA-4.0", "Hugging Face dataset card lm-pub-quiz/BEAR: license cc-by-sa-4.0", BEAR_REVISION,
        _hf("lm-pub-quiz/BEAR", BEAR_REVISION, ["README.md", "BEAR/test-00000-of-00001.parquet"])),
    "popqa": Source(
        "popqa", "PopQA (entity-popularity QA)",
        "Mallen, Asai, Zhong, Das, Khashabi & Hajishirzi (2023). When Not to Trust Language Models: Investigating "
        "Effectiveness of Parametric and Non-Parametric Memories. ACL 2023.",
        "none stated on the dataset card", "Hugging Face dataset card akariasai/PopQA has no license field; the code repository "
        "AlexTMallen/adaptive-retrieval is MIT: research use, not redistributed", POPQA_REVISION,
        _hf("akariasai/PopQA", POPQA_REVISION, ["README.md", "test.tsv"])),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(url: str, path: Path) -> int:
    import requests
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=120, stream=True) as response:
        response.raise_for_status()
        if "text/html" in (response.headers.get("content-type") or "") and not url.endswith((".md", ".html")):
            raise RuntimeError(f"{url} returned HTML (a login or virus-scan page?), not the file")
        size = 0
        with open(tmp, "wb") as handle:
            for block in response.iter_content(1 << 20):
                handle.write(block)
                size += len(block)
        expected = response.headers.get("content-length")
        if expected is not None and int(expected) != size and response.headers.get("content-encoding") in (None, "identity"):
            raise RuntimeError(f"{url}: {size} bytes, server said {expected}")
    tmp.replace(path)
    return size


def fetch(name: str, root: Path = ROOT, *, force: bool = False) -> dict[str, Any]:
    """Download one set's raw files (skipping files already present unless `force`) and write `source.json`."""
    source = SOURCES[name]
    folder = Path(root).expanduser() / name
    raw = folder / "raw"
    records = []
    for rel, url in source.files.items():
        path = raw / rel
        if force or not path.exists():
            _download(url, path)
        first, second = sha256_file(path), sha256_file(path)
        if first != second:
            raise RuntimeError(f"{path}: two sha256 passes disagree ({first} vs {second})")
        records.append({"path": f"raw/{rel}", "url": url, "bytes": path.stat().st_size, "sha256": first})
    info = {"name": name, "title": source.title, "citation": source.citation, "licence": source.licence,
            "licence_evidence": source.licence_evidence, "pinned": source.commit, "commit_items": source.commit_items,
            "retrieved": time.strftime("%Y-%m-%d"), "files": records}
    (folder / "source.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def verify(name: str, root: Path = ROOT) -> list[str]:
    """Paths whose sha256 no longer matches `source.json` (empty: all intact)."""
    folder = Path(root).expanduser() / name
    info = json.loads((folder / "source.json").read_text())
    return [r["path"] for r in info["files"] if not (folder / r["path"]).exists() or sha256_file(folder / r["path"]) != r["sha256"]]


def raw_dir(name: str, root: Path = ROOT) -> Path:
    folder = Path(root).expanduser() / name / "raw"
    if not folder.exists():
        raise FileNotFoundError(f"{folder} missing: run `python -m vsa_embed.benchmarks.sources fetch --sets {name}`")
    return folder


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "verify", "list"))
    parser.add_argument("--sets", nargs="*", default=list(SOURCES))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    for name in args.sets:
        if args.command == "list":
            s = SOURCES[name]
            print(f"{name}: {s.title} | {s.licence} | {len(s.files)} files | pinned {s.commit}")
        elif args.command == "fetch":
            info = fetch(name, args.root, force=args.force)
            print(f"{name}: {len(info['files'])} files, {sum(f['bytes'] for f in info['files']):,} bytes")
        else:
            bad = verify(name, args.root)
            print(f"{name}: {'OK' if not bad else 'MISMATCH ' + ', '.join(bad)}")


if __name__ == "__main__":
    main()
