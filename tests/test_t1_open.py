"""T1-open (MeSH + PubMed): entry-term parsing, PubMed extraction, holdout closure, linker holdout,
document mixing, feasibility counts, cardinality vs brute force, item builders."""

import gzip
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from vsa_embed.data.pubmed import (citation_text, extract_file, iter_citations, parse_md5_sidecar, pmid_bucket,
                                   section_label)
from vsa_embed.experiments.t1_items import (neighbourhood, pubmedqa_items, pubmedqa_with_context, rare_neighbour_items,
                                            sentence_around)
from vsa_embed.experiments.t1_open_corpus import (assert_alias_disjoint, choose_track_holdout, containment_index,
                                                  holdout_closure, load_config, mix_documents, stratum_counts, verdict,
                                                  whole_word_substrings, window_mask)
from vsa_embed.ontologies.mesh import (MeshAliasPolicy, build_mesh_ontology, build_mesh_track_ontology,
                                       mesh_track_aliases, natural_order_candidates)
from vsa_embed.span_channel import AliasTable, CausalLinker, cardinality_report, normalize_alias


def _term(text: str, *, permuted: bool = False, tag: str = "NON", preferred: bool = False) -> str:
    return (f'<Term ConceptPreferredTermYN="{"Y" if preferred else "N"}" IsPermutedTermYN="{"Y" if permuted else "N"}" '
            f'LexicalTag="{tag}" RecordPreferredTermYN="{"Y" if preferred else "N"}"><String>{text}</String></Term>')


def _descriptor(ui: str, name: str, trees: list[str], terms: list[str], cls: str = "1") -> str:
    tree_xml = "".join(f"<TreeNumber>{t}</TreeNumber>" for t in trees)
    return (f'<DescriptorRecord DescriptorClass="{cls}"><DescriptorUI>{ui}</DescriptorUI><DescriptorName><String>{name}</String>'
            f'</DescriptorName><TreeNumberList>{tree_xml}</TreeNumberList><ConceptList><Concept PreferredConceptYN="Y">'
            f'<ScopeNote>Note for {name}.</ScopeNote><TermList>{"".join(terms)}</TermList></Concept></ConceptList></DescriptorRecord>')


@pytest.fixture
def mesh_file(tmp_path: Path) -> Path:
    records = [
        _descriptor("D000001", "Arthritis", ["C05.550.114"], [_term("Arthritis", preferred=True), _term("Arthritides")]),
        _descriptor("D000002", "Arthritis, Rheumatoid", ["C05.550.114.154"], [
            _term("Arthritis, Rheumatoid", preferred=True), _term("Rheumatoid Arthritis", permuted=True),
            _term("RA", tag="ABB")]),
        _descriptor("D000003", "Hand, Foot and Mouth Disease", ["C01.925.782"], [_term("Hand, Foot and Mouth Disease", preferred=True)]),
        _descriptor("D000004", "DNA", ["D13.444.308"], [_term("DNA", tag="ACR", preferred=True), _term("Deoxyribonucleic Acid"),
                                                         _term("Acid, Deoxyribonucleic", permuted=True)]),
        _descriptor("D000005", "Review", ["V02.912"], [_term("Review", preferred=True), _term("Review Literature")], cls="2"),
        _descriptor("D000006", "Volition", ["F01.658.957"], [_term("Volition", preferred=True), _term("Will")]),
        _descriptor("D000007", "Leukemia, Lymphoblastic, Acute", ["C04.557.337.428"], [
            _term("Leukemia, Lymphoblastic, Acute", preferred=True), _term("Acute Lymphoblastic Leukemia", permuted=True)]),
        _descriptor("D000008", "Arthritis, Juvenile", ["C05.550.114.099"], [
            _term("Arthritis, Juvenile", preferred=True), _term("Juvenile Rheumatoid Arthritis")]),
    ]
    path = tmp_path / "desc.xml.gz"
    with gzip.open(path, "wt") as handle:
        handle.write('<?xml version="1.0"?><DescriptorRecordSet>' + "".join(records) + "</DescriptorRecordSet>")
    return path


# -- MeSH entry terms -----------------------------------------------------------------------------

def test_natural_order_candidates() -> None:
    assert "rheumatoid arthritis" in natural_order_candidates("Arthritis, Rheumatoid")
    assert "acute lymphoblastic leukemia" in natural_order_candidates("Leukemia, Lymphoblastic, Acute")
    assert natural_order_candidates("Rheumatoid Arthritis") == set()
    assert natural_order_candidates("1,2-Dipalmitoylphosphatidylcholine") == set()   # no ", ": not inverted


def test_mesh_track_aliases_policy(mesh_file: Path) -> None:
    onto = build_mesh_track_ontology(mesh_file)
    aliases = {normalize_alias(a): onto.concept_names[c] for a, c in onto.alias_pairs}
    # inverted heading dropped because its natural order is a listed (permuted) term; that term links
    assert "arthritis, rheumatoid" not in aliases and aliases["rheumatoid arthritis"] == "D000002"
    assert "leukemia, lymphoblastic, acute" not in aliases and aliases["acute lymphoblastic leukemia"] == "D000007"
    # an unresolved comma form is kept as listed; no natural form is invented
    assert aliases["hand, foot and mouth disease"] == "D000003" and "foot and mouth disease hand" not in aliases
    assert "arthritis, juvenile" in aliases and "juvenile arthritis" not in aliases
    # bare abbreviation dropped; an abbreviation that is the heading is kept
    assert "ra" not in aliases and aliases["dna"] == "D000004"
    assert "acid, deoxyribonucleic" not in aliases and aliases["deoxyribonucleic acid"] == "D000004"
    # publication types excluded; function words dropped
    assert "review" not in aliases and "review literature" not in aliases
    assert "will" not in aliases and aliases["volition"] == "D000006"
    stats = onto.metadata["alias_stats"]
    assert stats["dropped_resolved_inverted"] == 3 and stats["dropped_abbreviation"] == 1
    assert stats["descriptors_excluded_class"] == 1 and stats["dropped_function_word"] == 1
    assert onto.metadata["parents"][onto.concept_names.index("D000002")] == ["D000001"]


def test_mesh_e2_aliases_unchanged(mesh_file: Path) -> None:
    """The E2 builder keeps every listed term verbatim (new behaviour lives under the new name)."""
    onto = build_mesh_ontology(mesh_file)
    aliases = {a for a, _ in onto.alias_pairs}
    assert {"Arthritis, Rheumatoid", "RA", "Will", "Review"} <= aliases
    track = build_mesh_track_ontology(mesh_file)
    assert track.frames == onto.frames and track.concept_names == onto.concept_names


def test_policy_can_keep_inverted(mesh_file: Path) -> None:
    from vsa_embed.ontologies.mesh import _parse
    records = sorted(_parse(mesh_file), key=lambda r: r["ui"])
    pairs, _ = mesh_track_aliases(records, MeshAliasPolicy.from_config({"drop_resolved_inverted": False, "drop_lexical_tags": []}))
    aliases = {normalize_alias(a) for a, _ in pairs}
    assert {"arthritis, rheumatoid", "ra"} <= aliases


# -- PubMed ------------------------------------------------------------------------------------------

PUBMED = """<?xml version="1.0"?><PubmedArticleSet>
<PubmedArticle><MedlineCitation><PMID Version="1">111</PMID><Article><Journal><JournalIssue><PubDate><Year>2026</Year></PubDate></JournalIssue></Journal>
<ArticleTitle>Ozone and <i>toothpaste</i>.</ArticleTitle><Abstract>
<AbstractText Label="BACKGROUND">Dentinal hypersensitivity is common in adults and causes pain in many patients worldwide today.</AbstractText>
<AbstractText Label="METHODS">We measured 0.7 mg/m<sup>3</sup> of ozone in eighty patients over six months in two groups.<b>IMPORTANCE</b>It works well.</AbstractText>
</Abstract><Language>eng</Language></Article><MeshHeadingList><MeshHeading><DescriptorName UI="D010146">Pain</DescriptorName></MeshHeading></MeshHeadingList></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID Version="1">112</PMID><Article><Journal><JournalIssue><PubDate><MedlineDate>1998 Spring</MedlineDate></PubDate></JournalIssue></Journal>
<ArticleTitle>No abstract here.</ArticleTitle><Language>eng</Language></Article></MedlineCitation></PubmedArticle>
<PubmedArticle><MedlineCitation><PMID Version="1">113</PMID><Article><Journal><JournalIssue><PubDate><MedlineDate>1998 Spring</MedlineDate></PubDate></JournalIssue></Journal>
<ArticleTitle>[Ein Titel.]</ArticleTitle><Abstract><AbstractText>One two three four five six seven eight nine ten eleven twelve thirteen.</AbstractText></Abstract><Language>ger</Language></Article></MedlineCitation></PubmedArticle>
</PubmedArticleSet>"""


def test_pubmed_parsing_and_extraction(tmp_path: Path) -> None:
    path = tmp_path / "pubmed26n0001.xml.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(PUBMED)
    records = list(iter_citations(path))
    assert [r["pmid"] for r in records] == [111, 112, 113]
    assert records[1]["year"] == 1998 and records[0]["mesh"] == ["D010146"]
    text = citation_text(records[0])
    assert text.startswith("Ozone and toothpaste.\n\nBackground: Dentinal")
    assert "mg/m3 of ozone" in text and "\nImportance: It works well." in text and "METHODS" not in text
    counts = extract_file(path, tmp_path / "out.parquet", min_abstract_words=10)
    assert counts["kept"] == 1 and counts["no_abstract"] == 1 and counts["not_english"] == 1
    assert section_label("MATERIALS AND METHODS") == "Materials and methods" and section_label("Aims") == "Aims"
    assert parse_md5_sidecar("MD5(pubmed26n0001.xml.gz)= 2b09615f8dc1dc37a3db6bf924e439ca\n") == "2b09615f8dc1dc37a3db6bf924e439ca"
    assert pmid_bucket(41575207) == pmid_bucket(41575207) and 0 <= pmid_bucket(1) < 10_000


# -- holdout -------------------------------------------------------------------------------------------

def _table() -> AliasTable:
    return AliasTable.from_pairs([("rheumatoid arthritis", 0), ("juvenile rheumatoid arthritis", 1), ("arthritis", 2),
                                  ("still disease", 1), ("adult onset still disease", 3), ("lung", 4), ("lung cancer", 5),
                                  ("small cell lung cancer", 6), ("lungs", 4), ("arthritides", 2)])


def test_whole_word_substrings() -> None:
    parts = whole_word_substrings("juvenile rheumatoid arthritis")
    assert {"juvenile", "rheumatoid", "arthritis", "rheumatoid arthritis", "juvenile rheumatoid"} <= parts
    assert "arthritis" in whole_word_substrings("anti-arthritis agents") and "rthritis" not in whole_word_substrings("arthritis x")


def test_holdout_closure_is_alias_disjoint() -> None:
    table = _table()
    entry = table.alias_to_entry
    index = containment_index(table)
    held = holdout_closure({entry["rheumatoid arthritis"]}, table, index)
    # juvenile RA contains the held-out alias; adult-onset Still disease contains an alias of juvenile RA
    assert held == {entry["rheumatoid arthritis"], entry["juvenile rheumatoid arthritis"], entry["adult onset still disease"]}
    concepts = sorted({c for e in held for c in table.entry_concepts[e]})
    full = AliasTable.from_pairs([(a, table.entry_concepts[e][0]) for a, e in table.alias_to_entry.items()],
                                 holdout=concepts, include_holdout=True)
    assert_alias_disjoint(full, full.without_holdout())
    leaky = AliasTable.from_pairs([(a, table.entry_concepts[e][0]) for a, e in table.alias_to_entry.items()],
                                  holdout=[0], include_holdout=True)
    with pytest.raises(AssertionError):
        assert_alias_disjoint(leaky, leaky.without_holdout())   # "juvenile rheumatoid arthritis" still trains


def test_choose_track_holdout_excludes_hubs() -> None:
    table = _table()
    entry = table.alias_to_entry
    names = [f"C{i}" for i in range(7)]
    counts = Counter({e: 50 for e in range(len(table.entry_concepts))})
    lengths = {e: 2 for e in range(len(table.entry_concepts))}
    result = choose_track_holdout(counts, lengths, table, names, fraction=1.0, min_count=5, seed=0, max_containing=1)
    # "arthritis" (contained in 3 other entries) and "lung" (2) are hubs, never chosen
    assert entry["arthritis"] not in result["chosen_entries"] and entry["lung"] not in result["chosen_entries"]
    assert result["excluded_hub_entries"] == 2
    assert set(result["heldout_entries"]) >= set(result["chosen_entries"])
    again = choose_track_holdout(counts, lengths, table, names, fraction=1.0, min_count=5, seed=0, max_containing=1)
    assert again["sha256"] == result["sha256"]


def _tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    except OSError:
        pytest.skip("gpt2 tokenizer not cached")


def test_heldout_aliases_never_in_train_spans(tmp_path: Path) -> None:
    from vsa_embed.data.corpus import TokenCorpus, build_corpus
    tokenizer = _tokenizer()
    table = _table()
    held = holdout_closure({table.alias_to_entry["rheumatoid arthritis"]}, table, containment_index(table))
    concepts = sorted({c for e in held for c in table.entry_concepts[e]})
    pairs = [(a, table.entry_concepts[e][0]) for a, e in table.alias_to_entry.items()]
    full = AliasTable.from_pairs(pairs, holdout=concepts, include_holdout=True)
    train = full.without_holdout()
    texts = ["Juvenile rheumatoid arthritis and rheumatoid arthritis differ from arthritis.",
             "Adult onset Still disease, small cell lung cancer and the lungs."] * 6
    build_corpus(texts, tmp_path / "train", tokenizer_name="gpt2", table=train, eos_id=50256, max_tokens=10_000, workers=1, batch_texts=4)
    build_corpus(texts, tmp_path / "eval", tokenizer_name="gpt2", table=full, eos_id=50256, max_tokens=10_000, workers=1, batch_texts=4)
    held_entries = full.heldout_entries()
    held_aliases = {a for a, e in full.alias_to_entry.items() if e in held_entries}
    corpus = TokenCorpus.open(tmp_path / "train")
    assert corpus.spans["entry"].size > 0 and not set(corpus.spans["entry"].tolist()) & held_entries
    for start, end in zip(corpus.spans["start"], corpus.spans["end"]):
        surface = tokenizer.decode(corpus.tokens[start:end + 1]).strip().lower().rstrip(".,")
        assert surface not in held_aliases and not whole_word_substrings(surface) & held_aliases
    evaluation = TokenCorpus.open(tmp_path / "eval")
    assert set(evaluation.spans["entry"].tolist()) & held_entries     # evaluation links them


# -- mixing, feasibility --------------------------------------------------------------------------------

def test_mix_documents_tracks_shares() -> None:
    domain = iter(["a" * 40] * 1000)
    general = iter(["b" * 100] * 1000)
    log: list[int] = []
    docs = list(mix_documents([domain, general], [0.5, 0.5], [4.0, 5.0], log=log))
    tokens = [sum(len(d) / 4.0 for d, s in zip(docs, log) if s == 0), sum(len(d) / 5.0 for d, s in zip(docs, log) if s == 1)]
    assert abs(tokens[0] / sum(tokens) - 0.5) < 0.01 and len(log) == len(docs)
    assert Counter(log)[0] == 1000                       # stops when the chosen stream runs out
    again: list[int] = []
    list(mix_documents([iter(["a" * 40] * 1000), iter(["b" * 100] * 1000)], [0.5, 0.5], [4.0, 5.0], log=again))
    assert again == log


def test_source_token_shares_exact_and_reuse_guard(tmp_path: Path) -> None:
    from vsa_embed.data.corpus import build_corpus
    from vsa_embed.experiments.t1_open_corpus import guard_reuse, source_token_shares
    tokenizer = _tokenizer()
    domain = [f"Rheumatoid arthritis case {i} with juvenile onset." for i in range(30)]
    general = [f"A long general web document number {i} about the lungs and many other things." for i in range(30)]
    log: list[int] = []
    texts = mix_documents([iter(domain), iter(general)], [0.5, 0.5], [4.5, 4.5], log=log)
    table = _table()
    build_corpus(texts, tmp_path / "mix", tokenizer_name="gpt2", table=table, eos_id=50256, max_tokens=400, workers=1,
                 batch_texts=4, extra_manifest={"max_tokens_requested": 400, "stream_signature": "abc"})
    shares = source_token_shares(tmp_path / "mix", log, 50256)
    documents = json.loads((tmp_path / "mix" / "manifest.json").read_text())["documents"]
    kept = [(s, d) for s, d in zip(log, mix_documents([iter(domain), iter(general)], [0.5, 0.5], [4.5, 4.5]))][:documents]
    expected = Counter()
    for source, text in kept:
        expected[source] += len(tokenizer.encode(text)) + 1          # + EOS
    assert shares["exact"] and shares["tokens"] == {"pubmed": expected[0], "general": expected[1]}
    guard_reuse(tmp_path / "mix", table, 400, "abc")
    with pytest.raises(ValueError):
        guard_reuse(tmp_path / "mix", table, 4000, "abc")      # another budget
    with pytest.raises(ValueError):
        guard_reuse(tmp_path / "mix", table, 400, "other")     # another document stream
    guard_reuse(tmp_path / "mix", AliasTable.from_pairs([("x", 0)]), 4000, "other")   # another table: rebuilt, fine


def test_stream_signatures_separate_streams() -> None:
    from vsa_embed.experiments.t1_open_corpus import TrackDocuments
    docs = TrackDocuments(pubmed_paths=[Path("a.parquet")], eval_buckets=800, general_shards=["s.parquet"], general_skip=5000,
                          eval_general_docs=5000)
    from dataclasses import replace
    smaller_eval = replace(docs, eval_pubmed_docs=100)
    assert docs.signature("train") == smaller_eval.signature("train")          # eval size never changes training
    assert docs.signature("eval-pubmed") != smaller_eval.signature("eval-pubmed")
    assert docs.signature("train") != replace(docs, eval_buckets=300).signature("train")


def test_load_config_extends(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text("a: 1\ndata: {x: 1, y: 2}\nhosts: [{name: h, t: 5}]\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\ndata: {y: 3}\nhosts: [{name: h, t: 1}]\n")
    assert load_config(tmp_path / "child.yaml") == {"a": 1, "data": {"x": 1, "y": 3}, "hosts": [{"name": "h", "t": 1}]}


def test_stratum_counts_and_windows() -> None:
    spans = {"entry": np.array([0, 0, 0, 0, 0, 1, 2, 2, 3]), "length": np.array([2, 2, 2, 2, 2, 1, 3, 3, 2]),
             "start": np.array([0, 5, 10, 20, 30, 40, 50, 60, 1990]), "inject": np.array([1, 6, 11, 21, 31, 40, 52, 62, 2001])}
    heldout = np.array([True, False, False, False])
    frequency = np.array([0, 3, 5, 200])
    counts = stratum_counts(spans, threshold=2, heldout=heldout, frequency=frequency)
    assert counts["heldout_occurrences"] == 5 and counts["heldout_entries_5plus"] == 1
    assert counts["rare_occurrences"] == 2 and counts["rare_entries_linked"] == 1      # entry 1 is ℓ = 1
    mask = window_mask(spans, [0, 1000], 1000)
    assert mask.tolist() == [True] * 8 + [False]                                         # last span crosses 2000
    criteria = {"heldout_min_entries_5plus": 1, "heldout_min_occurrences": 5, "rare_min_entries": 1, "rare_min_occurrences": 3}
    assert verdict(counts, criteria, rare_measured=True)["verdict"].startswith("feasible for the held-out stratum only")


# -- cardinality vs brute force ---------------------------------------------------------------------------

def _brute_force_spans(text: str, offsets, aliases: dict[str, int]) -> list[tuple[int, int, int]]:
    """Naive prefix-causal linking: at every token end, the longest alias the text ends with that
    starts at a word boundary (single-space texts only)."""
    lowered = text.lower()
    spans = []
    for token, (token_start, end) in enumerate(offsets):
        if end <= token_start:
            continue
        best = None
        for alias, entry in aliases.items():
            start = end - len(alias)
            if start >= 0 and lowered[start:end] == alias and (start == 0 or not (lowered[start - 1].isalnum() or lowered[start - 1] == "_")):
                if best is None or len(alias) > len(best[0]):
                    best = (alias, entry, start)
        if best:
            first = max(t for t, (s, _) in enumerate(offsets) if s <= best[2])
            spans.append((best[1], token - first + 1, first))
    return spans


def _brute_force_rows(texts: list[str], tokenizer, aliases: dict[str, int], thresholds=(1, 2, 3, 4)):
    rows = []
    linked = []
    total = 0
    for text in texts:
        offsets = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
        total += len(offsets)
        linked.append(_brute_force_spans(text, offsets, aliases))
    for threshold in thresholds:
        counts = Counter(e for spans in linked for e, length, _ in spans if length >= threshold)
        rows.append({"linked_entries": len(counts), "span_occurrences": sum(counts.values()), "tokens": total})
    return rows


def test_cardinality_matches_brute_force_synthetic() -> None:
    tokenizer = _tokenizer()
    table = _table()
    texts = ["Juvenile rheumatoid arthritis and rheumatoid arthritis differ from arthritis in the lungs.",
             "Small cell lung cancer is a lung cancer; arthritides are many.", "Nothing to link here at all."]
    rows = cardinality_report(table, tokenizer, texts)
    brute = _brute_force_rows(texts, tokenizer, table.alias_to_entry)
    for row, expected in zip(rows, brute):
        assert (row["linked_entries"], row["span_occurrences"], row["tokens"]) == \
               (expected["linked_entries"], expected["span_occurrences"], expected["tokens"])


MESH = Path("~/data/vsa-llm/mesh/desc2026.gz").expanduser()
PUBMED_TEXT = Path("~/data/vsa-llm/pubmed/text-2026/pubmed26n1334.parquet").expanduser()


@pytest.mark.skipif(not (MESH.exists() and PUBMED_TEXT.exists()), reason="MeSH 2026 / extracted PubMed not on this machine")
def test_cardinality_matches_brute_force_on_pubmed_sample() -> None:
    from itertools import islice
    from vsa_embed.data.pubmed import iter_pubmed
    tokenizer = _tokenizer()
    table = AliasTable.from_pairs(build_mesh_track_ontology(MESH).alias_pairs)
    texts = [" ".join(r["text"].split()) for r in islice(iter_pubmed([PUBMED_TEXT]), 12)]   # single spaces (brute force)
    lowered = " ".join(texts).lower()
    present = {a: e for a, e in table.alias_to_entry.items() if a in lowered}           # aliases that can match at all
    rows = cardinality_report(table, tokenizer, texts)
    brute = _brute_force_rows(texts, tokenizer, present)
    assert rows[1]["span_occurrences"] > 20
    for row, expected in zip(rows, brute):
        assert (row["linked_entries"], row["span_occurrences"]) == (expected["linked_entries"], expected["span_occurrences"])


# -- items -----------------------------------------------------------------------------------------------

def test_sentence_around() -> None:
    text = "First sentence here. The rheumatoid arthritis cohort was large. Last one."
    start = text.index("rheumatoid")
    sentence, s, e = sentence_around(text, start, start + len("rheumatoid arthritis"))
    assert sentence == "The rheumatoid arthritis cohort was large." and sentence[s:e] == "rheumatoid arthritis"
    long = "word " * 200 + "rheumatoid arthritis " + "word " * 200
    start = long.index("rheumatoid")
    sentence, s, e = sentence_around(long, start, start + 20, limit=100)
    assert len(sentence) <= 100 and sentence[s:e] == "rheumatoid arthritis"


def test_pubmedqa_items(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    rows = [{"pubid": 10 + i, "question": f"Does {i} work?", "context": {"contexts": ["A.", "B."], "labels": ["BACKGROUND", "RESULTS"]},
             "long_answer": "Yes.", "final_decision": ["yes", "no", "maybe"][i % 3]} for i in range(6)]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "qa.parquet")
    (tmp_path / "gt.json").write_text(json.dumps({"10": "yes", "11": "no"}))
    items = pubmedqa_items(tmp_path / "qa.parquet", tmp_path / "gt.json", seed=1)
    assert [i["official_test"] for i in items] == [True, True, False, False, False, False]
    assert "context" not in items[0] and items[2]["fold"] is not None and items[0]["fold"] is None
    assert items == pubmedqa_items(tmp_path / "qa.parquet", tmp_path / "gt.json", seed=1)
    joined = pubmedqa_with_context(items, tmp_path / "qa.parquet")
    assert joined[0]["context"] == "Background: A.\nResults: B." and joined[0]["long_answer"] == "Yes."
    rows[0]["context"] = {"contexts": ["changed."], "labels": ["BACKGROUND"]}
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "qa.parquet")
    with pytest.raises(ValueError):
        pubmedqa_with_context(items, tmp_path / "qa.parquet")


def test_context_pointers_resolve(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vsa_embed.experiments.t1_items import _context, resolve_contexts
    text = "Title here.\n\nBackground: The rheumatoid arthritis cohort was large. Next sentence."
    start = text.index("rheumatoid")
    pointer = _context({"pmid": 5, "start": start, "end": start + 20, "text": text})
    assert "text" not in pointer and pointer["alias"] == "rheumatoid arthritis"
    path = tmp_path / "p.parquet"
    pq.write_table(pa.table({"pmid": [5], "year": [2026], "text": [text], "mesh": [[]]}), path)
    resolved = resolve_contexts([{"id": "x", "context": pointer}], [path])[0]["context"]
    assert resolved["text"] == "Background: The rheumatoid arthritis cohort was large."
    assert resolved["text"][resolved["span"][0]:resolved["span"][1]] == "rheumatoid arthritis"


def test_rare_neighbour_selection_is_preregistered() -> None:
    names = [f"D{i:03d}" for i in range(60)]
    parents = [[] if i == 0 else ["D000"] for i in range(60)]
    metadata = {"trees": [["C01"] if i else ["C"] for i in range(60)], "parents": parents, "headings": [f"H{i}" for i in range(60)],
                "related": [[] for _ in range(60)], "actions": [[] for _ in range(60)], "scope_notes": ["n"] * 60}
    table = AliasTable.from_pairs([(f"concept number {i}", i) for i in range(60)])
    concept_entry = {table.entry_concepts[e][0]: e for e in range(60)}
    heldout = {concept_entry[i] for i in range(1, 26)}
    frequency = np.zeros(60, dtype=int)
    for i in range(26, 50):
        frequency[concept_entry[i]] = 3
    occurrences = {e: [{"pmid": 1, "start": 0, "end": 4, "text": "text of it."}] for e in range(60)}
    kwargs = dict(occurrences=occurrences, table=table, metadata=metadata, concept_names=names, heldout=heldout,
                  frequency=frequency, entry_length={e: 3 for e in range(60)}, seed=7)
    items, info = rare_neighbour_items(**kwargs)
    assert info["pool_sizes"] == {"heldout": 25, "rare": 24} and len(items) == 40
    assert Counter(i["stratum"] for i in items) == {"heldout": 20, "rare": 20}
    assert [i["ui"] for i in items] == [i["ui"] for i in rare_neighbour_items(**kwargs)[0]]
    gold = neighbourhood(5, metadata, names)
    assert gold["parents"] == [{"ui": "D000", "heading": "H0"}] and gold["siblings_total"] == 58
