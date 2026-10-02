"""WP-T2: identifier alias normalization, scaled private libraries, real API docs, the T2 track builder."""

import bz2
import gzip
import hashlib
import io
import json
import random
import tarfile
import zlib
from pathlib import Path

import pytest

from vsa_embed.benchmarks.devtools_libraries import (AliasMatcher, generate_private_libraries, hidden_names, leakage_audit,
                                                     library_documents, qualname, referenced_names)
from vsa_embed.data.api_docs import (PythonIndex, build_snapshot, clean_path, cpython_doc_units, docstring_raises,
                                     docstring_returns, node_symbols_and_units, parse_inventory, parse_module, read_jsonl_gz)
from vsa_embed.ontologies.devtools import build_devtools_ontology, param_atom
from vsa_embed.span_channel import LINKER_VERSION, AliasTable, CausalLinker
from vsa_embed.tracks.devtools import choose_real_holdout, interleave


def _link(table: AliasTable, text: str) -> list[tuple[str, int]]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    offsets = tok(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    return [(text[offsets[s.start_token][0]:offsets[s.end_token][1]].strip(), s.entry)
            for s in CausalLinker(table, min_subtokens=1).link(text, offsets)]


# -- identifier alias normalization (linker) ---------------------------------------------------------

def test_identifier_normalization_links_snake_case_and_dotted_code_symbols() -> None:
    pairs = [("fetch_record_batch", 0), ("lib.module.fetch_record_batch", 0), ("RecordBatch", 1)]
    text = "rows = fetch_record_batch(n)\nfrom lib.module.fetch_record_batch import x\nRecordBatch()"
    identifier = AliasTable.from_pairs(pairs, normalization="identifier")
    entry = identifier.alias_to_entry["fetch_record_batch"]
    linked = _link(identifier, text)
    assert ("fetch_record_batch", entry) in linked and ("lib.module.fetch_record_batch", entry) in linked
    default = AliasTable.from_pairs(pairs)
    assert "fetch record batch" in default.alias_to_entry                  # `_` → space: never matches code
    assert [surface for surface, _ in _link(default, text)] == ["RecordBatch"]
    with pytest.raises(ValueError):
        AliasTable.from_pairs(pairs, normalization="nope")


def test_default_alias_tables_keep_their_digest_and_identifier_mode_round_trips(tmp_path: Path) -> None:
    from vsa_embed.evaluation.channel_probes import load_alias_table, save_alias_table
    pairs = [("new_york", 0), ("bank", 1), ("bank", 2)]
    table = AliasTable.from_pairs(pairs, holdout=[2], include_holdout=True)
    legacy = json.dumps({"aliases": sorted(table.alias_to_entry.items()), "entries": table.entry_concepts,
                         "holdout": sorted(table.holdout), "version": LINKER_VERSION}).encode()
    assert table.digest() == hashlib.sha256(legacy).hexdigest() and table.normalization == "default"
    save_alias_table(table, tmp_path / "a.json")
    assert "normalization" not in json.loads((tmp_path / "a.json").read_text())
    code = AliasTable.from_pairs(pairs, holdout=[2], include_holdout=True, normalization="identifier")
    assert code.digest() != table.digest() and code.without_holdout().normalization == "identifier"
    save_alias_table(code, tmp_path / "b.json")
    loaded = load_alias_table(tmp_path / "b.json")
    assert loaded == code and loaded.digest() == code.digest()


# -- private libraries --------------------------------------------------------------------------------

def test_c6_v1_still_regenerates_bit_identically(tmp_path: Path) -> None:
    wn = pytest.importorskip("nltk.corpus").wordnet
    try:
        forbidden = set(wn.all_lemma_names())
    except LookupError:
        pytest.skip("WordNet data not installed")
    from vsa_embed.benchmarks.devtools import build
    reference = Path(__file__).resolve().parents[1] / "experiments" / "c6-devtools-benchmark" / "v1"
    build(tmp_path, seed=20261001, forbidden=forbidden)
    for path in sorted(reference.iterdir()):
        assert (tmp_path / path.name).read_bytes() == path.read_bytes(), path.name


SMALL = dict(libraries=3, typescript_fraction=0.34, modules=(2, 3), classes=(2, 3), functions=(3, 5), methods=(1, 3),
             exceptions=(2, 4), zero_shot=12, zipf=1.6, zipf_offset=5)


def test_private_libraries_are_deterministic_prefix_free_and_follow_language_conventions() -> None:
    forbidden = {"bank", "flora", "tree"}
    a = generate_private_libraries(seed=3, forbidden=forbidden, **SMALL)
    assert a == generate_private_libraries(seed=3, forbidden=forbidden, **SMALL)
    symbols = a["symbols"]
    unique = {s["name"].lower() for s in symbols}
    shared = [s for s in symbols if s["overrides"]]                 # only overriding methods share a name
    assert len(unique) == len(symbols) - len(shared)
    assert not any(x != y and y.startswith(x) for x in unique for y in unique)
    assert not any(w.startswith(x) for x in unique for w in forbidden)
    stems = {stem for s in symbols if not s["overrides"] for stem in s["stems"]}
    assert not stems & forbidden
    for s in symbols:
        if s["kind"] in ("function", "method") and not s["overrides"]:
            assert ("_" in s["name"]) == (s["language"] == "python") and s["name"].islower() == (s["language"] == "python")
        if s["kind"] == "constant":
            assert s["name"].isupper() and "_" in s["name"]
    splits = {x: sum(s["split"] == x for s in symbols) for x in ("train", "heldout", "zeroshot")}
    assert splits["zeroshot"] == 12 and splits["heldout"] > 0
    held_classes = {(s["library"], s["name"]) for s in symbols if s["split"] == "heldout" and s["kind"] == "class"}
    assert all(s["split"] == "heldout" for s in symbols if s["kind"] == "method" and (s["library"], s["owner"]) in held_classes)
    zero = {(s["library"], s["name"]) for s in symbols if s["split"] == "zeroshot"}
    refs = referenced_names(s for s in symbols if s["split"] != "zeroshot")
    assert not zero & set(refs)
    assert all(s["kind"] in ("function", "method", "class") for s in symbols if s["split"] == "heldout")


def test_niche_libraries_only_rescale_weights() -> None:
    plain = generate_private_libraries(seed=6, **SMALL)
    niche = generate_private_libraries(seed=6, niche_libraries=1, niche_weight=0.01, **SMALL)
    tail = niche["libraries"][-1]["name"]
    assert [l["niche"] for l in niche["libraries"]] == [False] * 2 + [True]
    for a, b in zip(plain["symbols"], niche["symbols"]):
        assert {k: v for k, v in a.items() if k != "weight"} == {k: v for k, v in b.items() if k != "weight"}
        assert b["weight"] == pytest.approx(a["weight"] * (0.01 if a["library"] == tail else 1.0))


def test_private_documents_hide_heldout_and_zeroshot_symbols() -> None:
    data = generate_private_libraries(seed=4, **SMALL)
    train = list(library_documents(data, split="train", seed=4, max_chars=400_000))
    evaluation = list(library_documents(data, split="eval", seed=4, max_chars=200_000, uniform_focus=0.5))
    assert leakage_audit(data, train)["leaked_into_train"] == []
    held = [s["name"] for s in data["symbols"] if s["split"] == "heldout"]
    joined_eval = "\n".join(evaluation)
    assert any(name in joined_eval for name in held)
    zero = [s["name"] for s in data["symbols"] if s["split"] == "zeroshot"]
    assert not any(name in "\n".join(train) + joined_eval for name in zero)
    assert leakage_audit(data, train + [f"x = {held[0]}(y)"])["leaked_into_train"]
    assert hidden_names(data, "eval") == {(s["library"], s["name"]) for s in data["symbols"] if s["split"] == "zeroshot"}


def test_alias_matcher_follows_the_linker_word_start_rule() -> None:
    matcher = AliasMatcher(["fetch_rows", "lib.mod", "mod.fetch_rows"])
    counts = matcher.count("lib.mod.fetch_rows(x); fetch_rows_all(); prefetch_rows(); obj.fetch_rows")
    assert counts["fetch_rows"] == 3            # after '.', at a start, and as a prefix of fetch_rows_all
    assert counts["lib.mod"] == 1 and counts["mod.fetch_rows"] == 1
    assert matcher.lines_without("keep me\nfetch_rows()\nalso kept") == "keep me\nalso kept"
    with pytest.raises(ValueError):
        AliasMatcher(["two words"])


# -- real API documentation ---------------------------------------------------------------------------

def _toy_sources(root: Path) -> dict[str, Path]:
    site = root / "site"
    (site / "toypkg").mkdir(parents=True)
    (site / "toypkg" / "__init__.py").write_text(
        '"""Toy package."""\nfrom ._impl import fetch_record_batch, RecordReader, ReaderError\n'
        '__all__ = ["fetch_record_batch", "RecordReader", "ReaderError"]\n')
    (site / "toypkg" / "_impl.py").write_text('''
import os
from .errors import BaseFailure


class ReaderError(BaseFailure):
    """Raised when a record cannot be read."""


class RecordReader:
    """Reads records from a file.

    .. versionadded:: 1.4
    """

    def read_batch(self, size: int = 10) -> list:
        """Read a batch of records.

        Raises
        ------
        ReaderError
            If the file is truncated.
        """
        # read the header before the body of the file
        if size < 0:
            raise ReaderError("negative size")
        return []


def fetch_record_batch(path: str, size: int = 10) -> RecordReader:
    """Fetch one batch of records.

    Returns:
        RecordReader: a reader positioned at the batch.
    """
    # open the file lazily so that errors surface late
    return RecordReader()


def old_fetch(path):
    """Fetch records.

    .. deprecated:: 2.0
       Use :func:`fetch_record_batch` instead.
    """
    return fetch_record_batch(path)
''')
    (site / "toypkg" / "errors.py").write_text('class BaseFailure(Exception):\n    """Base error."""\n')
    std = root / "std"
    std.mkdir()
    (std / "toystd.py").write_text('def dumps(obj, indent=None):\n    """Serialize obj to a string."""\n    return ""\n')
    raw = root / "raw"
    raw.mkdir()
    body = zlib.compress(b"toystd py:module 0 library/toystd.html#module-$ -\n"
                         b"toystd.dumps py:function 1 library/toystd.html#$ -\n"
                         b"toystd.Encoder py:class 1 library/toystd.html#$ -\n")
    (raw / "objects.inv").write_bytes(b"# Sphinx inventory version 2\n# Project: Python\n# Version: 3.12\n"
                                      b"# The remainder of this file is compressed using zlib.\n" + body)
    page = ('"toystd" --- toy serializer\n***************************\n\nThe module has one function.\n\n'
            'Basic usage\n===========\n\ntoystd.dumps(obj, indent=None)\n\n   Serialize *obj*; see also toystd.Encoder.\n')
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as archive:
        data = page.encode()
        info = tarfile.TarInfo("python-3.12-docs-text/library/toystd.txt")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    (raw / "docs.tar.bz2").write_bytes(buffer.getvalue())
    node = {"modules": [{"type": "module", "name": "fs", "textRaw": "File system", "desc": "<p>The <code>fs</code> module.</p>",
                         "methods": [{"type": "method", "name": "readFile", "textRaw": "`fs.readFile(path[, options], callback)`",
                                      "meta": {"added": ["v0.1.29"]},
                                      "signatures": [{"params": [{"textRaw": "`path` {string}", "name": "path", "type": "string"}],
                                                      "return": {"textRaw": "Returns: {Promise}", "type": "Promise"}}],
                                      "desc": "<p>Reads a file &amp; returns it.</p>"},
                                     {"type": "method", "name": "require", "textRaw": "`require(id)`", "signatures": [{"params": []}]}],
                         "classes": [{"type": "class", "name": "fs.Dirent", "textRaw": "Class: `fs.Dirent`", "desc": "<p>An entry.</p>",
                                      "methods": [{"type": "method", "name": "isFile", "textRaw": "`dirent.isFile()`",
                                                   "stability": 0, "signatures": [{"params": [], "return": {"type": "boolean"}}]}]}]}]}
    (raw / "node.json").write_text(json.dumps(node))
    return {"site": site, "std": std, "raw": raw}


def test_python_module_parsing_resolves_schemas_and_reexports(tmp_path: Path) -> None:
    src = _toy_sources(tmp_path)
    parsed = [parse_module("toypkg", src["site"] / "toypkg" / "__init__.py", is_package=True),
              parse_module("toypkg._impl", src["site"] / "toypkg" / "_impl.py", is_package=False),
              parse_module("toypkg.errors", src["site"] / "toypkg" / "errors.py", is_package=False)]
    index = PythonIndex(parsed)
    defs = index.defs
    fetch = defs["toypkg._impl.fetch_record_batch"]
    assert fetch["returns"] == "toypkg._impl.RecordReader" and [p["name"] for p in fetch["params"]] == ["path", "size"]
    assert fetch["comments"] == ["# open the file lazily so that errors surface late"]
    method = defs["toypkg._impl.RecordReader.read_batch"]
    assert method["raises"] == ["toypkg._impl.ReaderError"] and method["kind"] == "method"
    assert defs["toypkg._impl.RecordReader"]["since"] == "1.4"
    assert defs["toypkg._impl.ReaderError"]["kind"] == "exception" and defs["toypkg._impl.ReaderError"]["bases"] == ["toypkg.errors.BaseFailure"]
    assert defs["toypkg._impl.old_fetch"]["deprecated"] and defs["toypkg._impl.old_fetch"]["deprecated_by"] == "fetch_record_batch"
    paths = index.public_paths("toypkg")
    assert paths["toypkg._impl.fetch_record_batch"] == {"toypkg.fetch_record_batch"}
    assert paths["toypkg._impl.RecordReader.read_batch"] == {"toypkg.RecordReader.read_batch"}
    assert clean_path("toypkg.fetch_record_batch") and not clean_path("toypkg._impl.x") and not clean_path("a..b")
    assert docstring_raises("Raises:\n    ValueError: if bad.\n") == ["ValueError"]
    assert docstring_returns("Returns\n-------\nout : ndarray\n    The result.\n") == "ndarray"


def test_inventory_cpython_pages_and_node_api(tmp_path: Path) -> None:
    src = _toy_sources(tmp_path)
    assert ("toystd.dumps", "py:function") in parse_inventory(src["raw"] / "objects.inv")
    units = list(cpython_doc_units(src["raw"] / "docs.tar.bz2"))
    assert units and "toystd.dumps(obj, indent=None)" in "\n".join(u["text"] for u in units)
    symbols, node_units = node_symbols_and_units(src["raw"] / "node.json")
    by_path = {s["path"]: s for s in symbols}
    assert by_path["node:fs.readFile"]["since"] == "v0" and by_path["node:fs.readFile"]["returns"] == "Promise"
    assert by_path["node:dirent.isFile"]["status"] == "deprecated" and by_path["node:dirent.isFile"]["owner"] == "node:fs.Dirent"
    assert any("Reads a file & returns it." in u["text"] for u in node_units)


def test_snapshot_keeps_dotted_public_aliases_only(tmp_path: Path) -> None:
    src = _toy_sources(tmp_path)
    stats = build_snapshot(tmp_path / "snap", packages=["toypkg"], site_packages=src["site"], stdlib_dir=src["std"],
                           cpython_docs=src["raw"] / "docs.tar.bz2", inventory=src["raw"] / "objects.inv",
                           node_api=src["raw"] / "node.json", english_words={"reader"})
    records = {r["path"]: r for r in read_jsonl_gz(tmp_path / "snap" / "symbols.jsonl.gz")}
    assert records["toypkg._impl.fetch_record_batch"]["aliases"] == ["toypkg.fetch_record_batch"]
    assert "RecordReader" in records["toypkg._impl.RecordReader"]["aliases"]          # bare CamelCase, not a word
    assert records["toystd.dumps"]["aliases"] == ["toystd.dumps"]
    assert "node:require" not in records                                              # undotted: never an alias
    assert all("." in a or a[:1].isupper() for r in records.values() for a in r["aliases"])
    assert stats["files"]["symbols.jsonl.gz"] == hashlib.sha256((tmp_path / "snap" / "symbols.jsonl.gz").read_bytes()).hexdigest()


def test_real_holdout_avoids_contained_prefix_and_referenced_aliases() -> None:
    records = [
        {"path": "a.f", "kind": "function", "aliases": ["a.f"]},                         # prefix of a.fx
        {"path": "a.fx", "kind": "function", "aliases": ["a.fx"]},
        {"path": "b.g", "kind": "function", "aliases": ["b.g"]},                         # inside c.b.g
        {"path": "c.b.g", "kind": "function", "aliases": ["c.b.g"]},
        {"path": "d.E", "kind": "class", "aliases": ["d.E"]},                           # a base of d.F
        {"path": "d.F", "kind": "class", "aliases": ["d.F"], "bases": ["d.E"]},
        {"path": "e.h", "kind": "function", "aliases": ["e.h"]},
    ]
    mentions = {r["path"]: 8 for r in records}
    held = choose_real_holdout(records, mentions, fraction=1.0, min_mentions=5, seed=1)
    assert set(held) == {"c.b.g", "d.F", "e.h"}
    assert choose_real_holdout(records, {**mentions, "e.h": 2}, fraction=1.0, min_mentions=5, seed=1) == ["c.b.g", "d.F"]


def test_distractors_match_the_shape_of_the_answer() -> None:
    from vsa_embed.tracks.devtools import SIGNATURE, filler_shape, shaped_choice_items, shaped_entailment_items
    pools = {"returns": ["int", "str", "float", "bool", "BrainshToulk", "ZelkFra", "GroushZham", "KlaxPou"],
             "member_of": ["brainsh", "zelkra", "pouxa", "BrainshToulk", "ZelkFra", "GroushZham"]}
    concepts = [{"concept": f"c{i}", "surface": f"fn_{i}", "split": "train",
                 "facts": {"returns": [rng_value], "member_of": [owner]}}
                for i, (rng_value, owner) in enumerate([("list", "dulka"), ("TrimVou", "TrimVou")])]
    rows = shaped_choice_items(concepts, SIGNATURE, pools, track="t2", task="signature_probe", seed=1, max_paraphrases=1)
    for row in rows:
        shapes = {filler_shape(choice.strip()) for choice in row["choices"]}
        assert len(shapes) == 1, row["choices"]
    pairs = shaped_entailment_items(concepts, SIGNATURE, pools, track="t2", task="zeroshot_entailment", seed=1)
    assert len(pairs) == 8 and [r["label"] for r in pairs[:2]] == [1, 0]


def test_interleave_spreads_the_secondary_stream_by_characters() -> None:
    primary = iter(["p" * 100] * 10_000)
    secondary = ["s" * 100] * 50
    out = list(interleave(primary, secondary, 0.1, 50_000))
    first_half = out[:len(out) // 2]
    assert sum(src for src, _ in out) == 50 and 20 <= sum(src for src, _ in first_half) <= 30
    assert sum(len(t) for _, t in out) >= 50_000


def test_devtools_ontology_frames_resolve_with_fallbacks() -> None:
    data = generate_private_libraries(seed=5, **SMALL)
    real = [{"path": "toypkg._impl.fetch", "kind": "function", "library": "toypkg", "language": "python", "module": "toypkg._impl",
             "owner": None, "aliases": ["toypkg.fetch"], "params": [{"name": "maxRetries", "annotation": "int", "optional": True}],
             "returns": "toypkg._impl.Reader", "raises": ["ValueError"], "bases": [], "status": "stable", "since": "1.2",
             "deprecated_by": None},
            {"path": "toypkg._impl.Reader", "kind": "class", "library": "toypkg", "language": "python", "module": "toypkg._impl",
             "owner": None, "aliases": ["toypkg.Reader", "ToyReader"], "params": [], "returns": None, "raises": [],
             "bases": ["Exception"], "status": "stable", "since": None, "deprecated_by": None}]
    onto = build_devtools_ontology(data, real, real_heldout=["toypkg._impl.fetch"], max_atomics=190, max_degree=8)
    assert len(onto.atomic_names) == 190 and onto.metadata["fillers_replaced"] and all(len(f) <= 8 for f in onto.frames)
    assert all(frame and all(0 <= a < len(onto.atomic_names) for _, a in frame) for frame in onto.frames)
    named = {onto.concept_names[i]: [(onto.relation_names[r], onto.atomic_names[a]) for r, a in f] for i, f in enumerate(onto.frames)}
    fetch = named["real:toypkg._impl.fetch"]
    assert ("returns", "symbol:toypkg.Reader") in fetch or ("returns", "kind:class") in fetch
    assert onto.metadata["splits"][onto.concept_names.index("real:toypkg._impl.fetch")] == "heldout"
    atoms = set(onto.atomic_names)
    assert all(a in atoms for frame in onto.metadata["zeroshot_frames"].values() for _, a in frame)
    assert len(onto.metadata["zeroshot_frames"]) == 12 and param_atom("maxRetries") == "param:max_retries"


# -- the track builder ----------------------------------------------------------------------------------

def _track_config(tmp_path: Path, src: dict[str, Path], general: Path, tokenizer: str, name: str) -> dict:
    return {
        "experiment": f"t2-tiny-{name}", "track": "t2", "seed": 3, "tokenizer": tokenizer, "workers": 1,
        "items_dir": str(tmp_path / f"items-{name}"),
        "paths": {"data_root": str(tmp_path / f"data-{name}"), "general_shards": [str(general)]},
        "devtools": {"docs_dir": str(tmp_path / "docs"), **{k: list(v) if isinstance(v, tuple) else v for k, v in SMALL.items()},
                     "heldout_fraction": 0.15, "eval_uniform_focus": 0.5, "train_chars": 250_000, "eval_chars": 90_000,
                     "forbidden_general_docs": 40},
        "real_libraries": {"snapshot_dir": str(tmp_path / "snap"), "raw_dir": str(src["raw"]), "python_packages": ["toypkg"],
                           "site_packages": str(src["site"]), "stdlib_dir": str(src["std"]), "cpython_docs": "docs.tar.bz2",
                           "inventory": "objects.inv", "node_api": "node.json", "eval_buckets": 3000, "heldout_fraction": 0.5,
                           "holdout_min_mentions": 1, "max_page_chars": 4000, "max_train_share": 0.3, "max_eval_share": 0.3},
        "linker": {"alias_normalization": "identifier"},
        "ontology": {"max_atomics": 8192, "max_degree": 16},
        "data": {"eval_tokens": 25_000, "contamination_docs": 40, "presample_tokens": 0, "holdout_fraction": 0.1,
                 "holdout_min_count": 5, "general_skip_docs": 0, "general_eval_docs": 20, "train_domain_tokens": 30_000,
                 "train_total_tokens": 45_000, "min_subtokens": 2, "cardinality_docs": 20},
        "items": {"train_per_bin": 5, "relations_per_symbol": 2},
        "feasibility_strict": {"heldout_min_entries_5plus": 300, "heldout_min_occurrences": 2000, "rare_min_entries": 300,
                               "rare_min_occurrences": 2000, "trainer_windows": 16, "window_length": 256},
        "cardinality_tokenizers": [tokenizer],
    }


def test_t2_builder_end_to_end_with_identifier_linking_and_shared_items(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from vsa_embed.data.corpus import TokenCorpus
    from vsa_embed.experiments.track_corpus import run
    src = _toy_sources(tmp_path)
    rng = random.Random(0)
    words = "the river city school garden market history music science water light energy family story".split()
    pq.write_table(pa.table({"text": [" ".join(rng.choice(words) for _ in range(120)) + "." for _ in range(200)]}),
                   tmp_path / "general.parquet")
    summaries = {}
    for name, tokenizer in (("gpt2", "gpt2"), ("smollm2", "HuggingFaceTB/SmolLM2-135M")):
        config = _track_config(tmp_path, src, tmp_path / "general.parquet", tokenizer, name)
        summaries[name] = run(config, tmp_path / f"run-{name}")
    docs = json.loads((tmp_path / "docs" / "documents_summary.json").read_text())
    assert docs["audit"]["leaked_into_train"] == 0 and docs["train"]["real_documents"] > 0
    onto = torch.load(tmp_path / "data-gpt2" / "ontology.pt", weights_only=False)
    assert onto["alias_normalization"] == "identifier" and (tmp_path / "data-gpt2" / "alias_table.json").exists()
    held = set(onto["heldout_entries"])
    train = TokenCorpus.open(tmp_path / "data-gpt2" / "train")
    evaluation = TokenCorpus.open(tmp_path / "data-gpt2" / "eval")
    assert not set(train.spans["entry"].tolist()) & held
    assert set(evaluation.spans["entry"].tolist()) & set(onto["heldout_real_entries"])
    linking = summaries["gpt2"]["linking_by_normalization"]
    assert linking["identifier"]["linked_concepts"] > linking["default"]["linked_concepts"]
    for path in sorted((tmp_path / "items-gpt2").iterdir()):                         # items do not depend on the tokenizer
        assert path.read_bytes() == (tmp_path / "items-smollm2" / path.name).read_bytes(), path.name
    record = json.loads((tmp_path / "items-gpt2" / "holdout.json").read_text())
    assert {"signature_probe.jsonl", "doc_qa_cloze.jsonl", "property_probe.jsonl", "zeroshot_property.jsonl",
            "zeroshot_entailment.jsonl"} <= set(record["files"])
    rows = [json.loads(line) for line in (tmp_path / "items-gpt2" / "signature_probe.jsonl").read_text().splitlines()]
    assert {r["source"] for r in rows} <= {"private", "real"} and all(0 <= r["label"] < len(r["choices"]) for r in rows)
    assert len({r["id"] for r in rows}) == len(rows)
    assert (tmp_path / "run-gpt2" / "feasibility_strict.json").exists()
    config = _track_config(tmp_path, src, tmp_path / "general.parquet", "gpt2", "pinned")
    config["data"]["expected_holdout_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="expected_holdout_sha256"):
        run(config, tmp_path / "run-pinned")
