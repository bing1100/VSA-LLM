"""C7 application tracks: adapters, synthetic glossary, shared items, feasibility and the builder."""

import gzip
import hashlib
import json
import random
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vsa_embed.benchmarks.glossary import documents, generate_glossary, leakage_audit, term_frame, write_documents
from vsa_embed.data.corpus import TokenCorpus
from vsa_embed.experiments.track_corpus import FEASIBILITY, add_synthetic, feasibility, recommend_min_subtokens
from vsa_embed.ontologies.chebi import build_chebi_ontology, formula_elements, keep_alias, parse_obo
from vsa_embed.ontologies.eurovoc import build_eurovoc_ontology, domain_name, mt_name
from vsa_embed.ontologies.glossary import build_glossary_ontology
from vsa_embed.ontologies.google_product import build_google_product_ontology, name_parts, singular
from vsa_embed.span_channel import AliasTable
from vsa_embed.tracks.common import (RelationTemplates, SyntheticConcept, choice_items, entailment_items, mention_item,
                                     occurrences, write_jsonl)
from vsa_embed.tracks.legal import extract_definitions
from vsa_embed.tracks.product import product_text


# -- T5 glossary generator ----------------------------------------------------------------------

def test_glossary_is_deterministic_with_unique_novel_names_and_fixed_splits() -> None:
    forbidden = {"bank", "flora", "brim"}
    a = generate_glossary(seed=7, terms=400, zero_shot=30, forbidden=forbidden)
    assert a == generate_glossary(seed=7, terms=400, zero_shot=30, forbidden=forbidden)
    names = [t["name"] for t in a["terms"]]
    assert len(names) == len(set(names))
    stems = [s.lower() for t in a["terms"] for s in t["stems"]]
    assert len(stems) == len(set(stems)) and not set(stems) & forbidden
    splits = {s: sum(t["split"] == s for t in a["terms"]) for s in ("train", "heldout", "zeroshot")}
    assert splits["zeroshot"] == 30 and 30 <= splits["heldout"] <= 45
    assert all(t["type"] not in ("team", "department", "committee") for t in a["terms"] if t["split"] == "heldout")
    acronyms = [al.lower() for t in a["terms"] for al in t["aliases"][1:]]
    assert acronyms and len(acronyms) == len(set(acronyms))
    words = forbidden | set(stems)
    # an acronym is never a prefix of a word, a stem or another acronym (prefix-causal linker)
    assert not any(w.startswith(x) for x in acronyms for w in words)
    assert not any(x != y and y.startswith(x) for x in acronyms for y in acronyms)
    zero = {t["name"] for t in a["terms"] if t["split"] == "zeroshot"}
    assert not any(f in zero for t in a["terms"] for fillers in t["relations"].values() for f in fillers)


def test_glossary_documents_hide_heldout_and_zeroshot_terms(tmp_path: Path) -> None:
    glossary = generate_glossary(seed=3, terms=300, zero_shot=20)
    train = list(documents(glossary, split="train", seed=3, max_chars=300_000))
    evaluation = list(documents(glossary, split="eval", seed=3, max_chars=150_000, uniform_focus=0.5))
    assert leakage_audit(glossary, train)["leaked_into_train"] == []
    held = [t["name"] for t in glossary["terms"] if t["split"] == "heldout"]
    assert any(name in "\n".join(evaluation) for name in held)
    zero = [t["name"] for t in glossary["terms"] if t["split"] == "zeroshot"]
    assert not any(name in "\n".join(train + evaluation) for name in zero)
    leaky = train + [f"Notes: {held[0]} is new."]
    assert leakage_audit(glossary, leaky)["leaked_into_train"] == [held[0]]
    summary = write_documents(glossary, tmp_path, seed=3, train_chars=50_000, eval_chars=20_000)
    assert summary["leaked_into_train"] == 0 and (tmp_path / "train.jsonl.gz").exists()


def test_glossary_ontology_resolves_every_frame_including_zero_shot_terms() -> None:
    glossary = generate_glossary(seed=5, terms=300, zero_shot=20)
    onto = build_glossary_ontology(glossary)
    assert len(onto.concept_names) == sum(t["split"] != "zeroshot" for t in glossary["terms"])
    assert all(frame and all(0 <= a < len(onto.atomic_names) for _, a in frame) for frame in onto.frames)
    synthetic = [SyntheticConcept(t["name"], t["aliases"], term_frame(t)) for t in glossary["terms"] if t["split"] == "zeroshot"]
    before = len(onto.concept_names)
    indices = add_synthetic(onto, synthetic)
    assert indices == list(range(before, before + 20)) and onto.concept_names[-1].startswith("synthetic:")
    with pytest.raises(KeyError):
        add_synthetic(onto, [SyntheticConcept("x", ["x"], [("owned_by", "term:does not exist")])])


# -- shared items and checks ---------------------------------------------------------------------

def test_choice_and_entailment_items_from_frame_facts() -> None:
    templates = {"owner": RelationTemplates(["{x} is owned by", "The owner of {x} is"], "{x} is owned by {y}.")}
    concepts = [{"concept": "c1", "surface": "Quax Hub", "split": "heldout", "facts": {"owner": ["team a"], "other": ["z"]}}]
    pools = {"owner": ["team a", "team b", "team c", "team d", "team e"]}
    items = choice_items(concepts, templates, pools, track="t", task="cloze", seed=1)
    assert len(items) == 2 and items[0]["group"] == items[1]["group"]
    assert items[0]["choices"] == items[1]["choices"] and items[0]["choices"][items[0]["label"]] == " team a"
    assert len(set(items[0]["choices"])) == 4 and items[0]["prompt"] == "Quax Hub is owned by"
    assert choice_items(concepts, templates, {"owner": ["team a", "team b"]}, track="t", task="c", seed=1) == []
    pairs = entailment_items(concepts, templates, pools, track="t", task="ent", seed=1)
    assert [p["label"] for p in pairs] == [1, 0] and pairs[0]["statement"] == "Quax Hub is owned by team a."
    assert "team a" not in pairs[1]["statement"]
    many = [{**concepts[0], "facts": {"owner": ["team a"], "o2": ["x"], "o3": ["y"]}}]
    t3 = {k: templates["owner"] for k in ("owner", "o2", "o3")}
    p3 = {k: ["team a", "team b", "team c", "team d", "x", "y"] for k in t3}
    assert len({i["relation"] for i in choice_items(many, t3, p3, track="t", task="c", seed=2, max_relations=2)}) == 2


def test_occurrences_follow_the_linker_word_start_rule_and_mention_items() -> None:
    texts = ["The Flosh Vault is down.", "zurkflosh vault and flosh vaults"]
    counts = occurrences(["Flosh Vault", "Missing Name"], texts)
    assert counts == {"Flosh Vault": 2, "Missing Name": 0}      # "flosh vaults" counts (no right boundary), "zurkflosh" not
    item = mention_item("Questions about Quax Hub go here.", "quax hub", id="i", split="train")
    assert item["span"] == [16, 24] and item["text"][16:24] == "Quax Hub"
    assert mention_item("nothing here", "quax") is None


def test_write_jsonl_hash_is_reproducible(tmp_path: Path) -> None:
    info = write_jsonl(tmp_path / "a.jsonl", [{"b": 1, "a": 2}])
    assert info["rows"] == 1 and info["sha256"] == hashlib.sha256((tmp_path / "a.jsonl").read_bytes()).hexdigest()
    assert (tmp_path / "a.jsonl").read_text() == '{"a": 2, "b": 1}\n'


# -- feasibility ---------------------------------------------------------------------------------

def _corpus(n_tokens: int, spans: list[tuple[int, int, int]]) -> TokenCorpus:
    """Spans as (start, end, entry); length = end − start + 1; inject = end."""
    spans = sorted(spans, key=lambda s: s[1])
    arrays = {"start": np.array([s for s, _, _ in spans], dtype=np.int64), "end": np.array([e for _, e, _ in spans], dtype=np.int64),
              "inject": np.array([e for _, e, _ in spans], dtype=np.int64), "entry": np.array([x for _, _, x in spans], dtype=np.int64),
              "length": np.array([e - s + 1 for s, e, _ in spans], dtype=np.int64),
              "confidence": np.ones(len(spans), dtype=np.float32)}
    return TokenCorpus(np.zeros(n_tokens, dtype=np.uint16), arrays, {})


def test_feasibility_counts_strata_in_the_evaluation_windows() -> None:
    criteria = {**FEASIBILITY, "window": 100, "eval_windows": [4], "min_heldout_entries": 2, "min_heldout_spans": 4,
                "min_rare_entries": 1, "min_rare_spans": 2, "min_train_entries": 2}
    # train: entry 1 seen 3× (rare), entry 2 seen 20× (frequent), entry 3 seen once with length 1 only
    train = _corpus(5000, [(i * 10, i * 10 + 1, 1) for i in range(3)] + [(100 + i * 10, 101 + i * 10, 2) for i in range(20)]
                    + [(400, 400, 3)])
    # eval: 4 windows of 100 tokens; held-out entries 7 and 8; rare entry 1; unseen entry 9
    spans = []
    for w in range(4):
        base = w * 100
        spans += [(base + 1, base + 2, 7), (base + 10, base + 12, 8), (base + 20, base + 21, 1), (base + 30, base + 31, 9),
                  (base + 40, base + 40, 3)]
    evaluation = _corpus(502, spans)
    rows = feasibility(evaluation, train, {7, 8}, entry_count=10, thresholds=(1, 2), criteria=criteria)
    by = {r["min_subtokens"]: r for r in rows}
    sample = by[2]["samples"][0]
    assert sample["windows"] == 4 and sample["heldout_entries"] == 2 and sample["heldout_spans"] == 8
    assert sample["rare_entries"] == 1 and sample["rare_spans"] == 4 and sample["unseen_entries"] == 1
    assert sample["covered_fraction"] == pytest.approx((2 + 3 + 2 + 2) / 100)
    assert by[2]["train_entries"] == 2 and by[1]["train_entries"] == 3
    assert by[1]["samples"][0]["rare_entries"] == 2      # entry 3 counts at ℓ_min = 1 only
    assert by[2]["feasible"] and recommend_min_subtokens(rows)["min_subtokens"] == 2
    strict = feasibility(evaluation, train, {7, 8}, entry_count=10, thresholds=(1, 2),
                         criteria={**criteria, "min_heldout_entries": 3})
    assert recommend_min_subtokens(strict) == {"min_subtokens": None, "verdict": "infeasible", "failed_checks": ["heldout_entries"]}


# -- ontology adapters on small fixtures -----------------------------------------------------------

OBO = """format-version: 1.2

[Term]
id: CHEBI:24431
name: chemical entity
subset: 3:STAR

[Term]
id: CHEBI:10
name: benzoic acids
subset: 3:STAR
is_a: CHEBI:24431 ! chemical entity

[Term]
id: CHEBI:20
name: benzoic acid
def: "A compound comprising a benzene ring with a carboxy group." []
subset: 3:STAR
synonym: "benzenecarboxylic acid" EXACT IUPAC:NAME [IUPAC]
synonym: "C7H6O2" RELATED [ChEBI]
synonym: "BA" RELATED [ChEBI]
synonym: "Benzoesaeure" RELATED [ChemIDplus]
is_a: CHEBI:10 ! benzoic acids
relationship: RO:0000087 CHEBI:30 ! has role food preservative
property_value: chemrof:generalized_empirical_formula "C7H6O2" xsd:string
property_value: chemrof:charge "0" xsd:integer

[Term]
id: CHEBI:30
name: food preservative
subset: 3:STAR

[Term]
id: CHEBI:40
name: 3-chlorobenzoic acid
subset: 3:STAR
is_a: CHEBI:10 ! benzoic acids
relationship: RO:0018038 CHEBI:20 ! has functional parent benzoic acid
property_value: chemrof:generalized_empirical_formula "C7H5ClO2" xsd:string

[Term]
id: CHEBI:50
name: obsolete thing
subset: 3:STAR
is_obsolete: true

[Term]
id: CHEBI:60
name: two-star thing
subset: 2:STAR

[Typedef]
id: RO:0000087
name: has role
"""


def test_chebi_adapter_parses_relations_aliases_and_elements(tmp_path: Path) -> None:
    path = tmp_path / "chebi.obo.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(OBO)
    records = {r["id"]: r for r in parse_obo(path)}
    assert records["CHEBI:20"]["relations"] == [("has_role", "CHEBI:30")] and records["CHEBI:50"].get("obsolete")
    assert formula_elements("C7H5ClO2") == ["C", "Cl", "O"]
    assert not keep_alias("C7H6O2", is_name=False, max_chars=120) and not keep_alias("BA", is_name=False, max_chars=120)
    assert not keep_alias("Acid", is_name=False, max_chars=120) and keep_alias("benzenecarboxylic acid", is_name=False, max_chars=120)
    onto = build_chebi_ontology(path, max_atomics=12)
    assert onto.concept_names == ["CHEBI:10", "CHEBI:20", "CHEBI:30", "CHEBI:40", "CHEBI:24431"]   # obsolete, 2-star dropped
    i = onto.concept_index["CHEBI:40"]
    edges = {(onto.relation_names[r], onto.atomic_names[a]) for r, a in onto.frames[i]}
    assert {("is_a", "chebi:CHEBI:10"), ("has_functional_parent", "chebi:CHEBI:20"), ("contains_element", "element:Cl"),
            ("branch", "branch:chemical entity")} <= edges
    aliases = {a for a, c in onto.alias_pairs if c == onto.concept_index["CHEBI:20"]}
    assert aliases == {"benzoic acid", "benzenecarboxylic acid", "Benzoesaeure"}
    assert "benzenecarboxylic acid" in onto.metadata["all_surface_forms"]


TAXONOMY = """# Google_Product_Taxonomy_Version: 2021-09-21
1 - Home & Garden
2 - Home & Garden > Kitchen & Dining
3 - Home & Garden > Kitchen & Dining > Cookware
4 - Home & Garden > Kitchen & Dining > Cookware > Skillets & Frying Pans
5 - Home & Garden > Kitchen & Dining > Cookware > Accessories
6 - Home & Garden > Kitchen & Dining > Cookware > Woks
"""


def test_google_product_adapter_frames_aliases_and_shopify_attributes(tmp_path: Path) -> None:
    (tmp_path / "tax.txt").write_text(TAXONOMY)
    categories = {"version": "x", "verticals": [{"name": "Home", "prefix": "hg", "categories": [
        {"id": "s1", "full_name": "Home & Garden > Kitchen & Dining > Cookware > Skillets & Frying Pans > Grill Pans",
         "name": "Grill Pans", "attributes": [{"name": "Material", "handle": "material"}]},
        {"id": "s2", "full_name": "Home & Garden > Kitchen & Dining > Cookware > Woks", "name": "Woks",
         "attributes": [{"name": "Material", "handle": "material"}, {"name": "Color", "handle": "color"}]}]}]}
    (tmp_path / "cat.json").write_text(json.dumps(categories))
    (tmp_path / "map.txt").write_text("→ Home & Garden > Kitchen & Dining > Cookware > Skillets & Frying Pans > Grill Pans\n"
                                      "⇒ Home & Garden > Kitchen & Dining > Cookware > Skillets & Frying Pans\n\n"
                                      "→ Home & Garden > Kitchen & Dining > Cookware > Woks\n⇒ Home & Garden > Kitchen & Dining > Cookware > Woks\n")
    assert name_parts("Skillets & Frying Pans") == ["Skillets", "Frying Pans"] and singular("dishes") == "dish"
    onto = build_google_product_ontology(tmp_path / "tax.txt", shopify_categories=tmp_path / "cat.json",
                                         shopify_mapping=tmp_path / "map.txt")
    i = onto.concept_index["gpt:4"]
    edges = {(onto.relation_names[r], onto.atomic_names[a]) for r, a in onto.frames[i]}
    assert {("parent", "category:3"), ("top_category", "category:1"), ("second_category", "category:2"), ("depth", "depth:4"),
            ("name_token", "token:skillet"), ("has_attribute", "attribute:Material"), ("path_token", "token:cookware")} <= edges
    aliases = {a for a, c in onto.alias_pairs if c == i}
    assert {"Skillets & Frying Pans", "Skillets", "Frying Pans", "Frying Pan", "Skillet", "Grill Pans", "Grill Pan"} <= aliases
    assert not any(a.lower() == "accessories" for a, _ in onto.alias_pairs)      # generic single words dropped
    assert onto.metadata["attributes"]["gpt:6"] == ["Color", "Material"]


RDF = """<?xml version="1.0" encoding="UTF-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
<rdf:Description rdf:about="http://eurovoc.europa.eu/100141">
  <rdf:type rdf:resource="http://www.w3.org/2004/02/skos/core#ConceptScheme"/>
  <prefLabel xmlns="http://www.w3.org/2004/02/skos/core#" xml:lang="en">EuroVoc</prefLabel>
</rdf:Description>
<rdf:Description rdf:about="http://eurovoc.europa.eu/100254">
  <rdf:type rdf:resource="http://www.w3.org/2004/02/skos/core#ConceptScheme"/>
  <prefLabel xmlns="http://www.w3.org/2004/02/skos/core#" xml:lang="en">6011 animal product</prefLabel>
  <notation xmlns="http://www.w3.org/2004/02/skos/core#">6011</notation>
</rdf:Description>
<rdf:Description rdf:about="http://eurovoc.europa.eu/100160">
  <rdf:type rdf:resource="http://www.w3.org/2004/02/skos/core#Concept"/>
  <inScheme xmlns="http://www.w3.org/2004/02/skos/core#" rdf:resource="http://eurovoc.europa.eu/domains"/>
  <prefLabel xmlns="http://www.w3.org/2004/02/skos/core#" xml:lang="en">60 AGRI-FOODSTUFFS</prefLabel>
  <notation xmlns="http://www.w3.org/2004/02/skos/core#">60</notation>
</rdf:Description>
<rdf:Description rdf:about="http://eurovoc.europa.eu/1">
  <rdf:type rdf:resource="http://www.w3.org/2004/02/skos/core#Concept"/>
  <inScheme xmlns="http://www.w3.org/2004/02/skos/core#" rdf:resource="http://eurovoc.europa.eu/100141"/>
  <inScheme xmlns="http://www.w3.org/2004/02/skos/core#" rdf:resource="http://eurovoc.europa.eu/100254"/>
  <prefLabel xmlns="http://www.w3.org/2004/02/skos/core#" xml:lang="en">animal product</prefLabel>
  <prefLabel xmlns="http://www.w3.org/2004/02/skos/core#" xml:lang="fr">produit animal</prefLabel>
</rdf:Description>
<rdf:Description rdf:about="http://eurovoc.europa.eu/2">
  <rdf:type rdf:resource="http://www.w3.org/2004/02/skos/core#Concept"/>
  <broader xmlns="http://www.w3.org/2004/02/skos/core#" rdf:resource="http://eurovoc.europa.eu/1"/>
  <related xmlns="http://www.w3.org/2004/02/skos/core#" rdf:resource="http://eurovoc.europa.eu/3"/>
  <inScheme xmlns="http://www.w3.org/2004/02/skos/core#" rdf:resource="http://eurovoc.europa.eu/100141"/>
  <inScheme xmlns="http://www.w3.org/2004/02/skos/core#" rdf:resource="http://eurovoc.europa.eu/100254"/>
  <prefLabel xmlns="http://www.w3.org/2004/02/skos/core#" xml:lang="en">carcase</prefLabel>
  <altLabel xmlns="http://www.w3.org/2004/02/skos/core#" xml:lang="en">animal carcass</altLabel>
</rdf:Description>
<rdf:Description rdf:about="http://eurovoc.europa.eu/3">
  <rdf:type rdf:resource="http://www.w3.org/2004/02/skos/core#Concept"/>
  <broader xmlns="http://www.w3.org/2004/02/skos/core#" rdf:resource="http://eurovoc.europa.eu/2"/>
  <inScheme xmlns="http://www.w3.org/2004/02/skos/core#" rdf:resource="http://eurovoc.europa.eu/100141"/>
  <inScheme xmlns="http://www.w3.org/2004/02/skos/core#" rdf:resource="http://eurovoc.europa.eu/100254"/>
  <prefLabel xmlns="http://www.w3.org/2004/02/skos/core#" xml:lang="en">meat carcase grading</prefLabel>
</rdf:Description>
</rdf:RDF>
"""


def test_eurovoc_adapter_reads_skos_hierarchy_microthesauri_and_domains(tmp_path: Path) -> None:
    with zipfile.ZipFile(tmp_path / "ev.zip", "w") as archive:
        archive.writestr("eurovoc_in_skos_core_concepts.rdf", RDF)
    assert mt_name("6011 animal product") == "animal product" and domain_name("60 AGRI-FOODSTUFFS") == "agri-foodstuffs"
    onto = build_eurovoc_ontology(tmp_path / "ev.zip")
    assert onto.concept_names == ["eurovoc:1", "eurovoc:2", "eurovoc:3"]
    edges = {(onto.relation_names[r], onto.atomic_names[a]) for r, a in onto.frames[onto.concept_index["eurovoc:3"]]}
    assert {("broader", "eurovoc:2"), ("top_term", "eurovoc:1"), ("microthesaurus", "mt:6011"), ("domain", "domain:60")} <= edges
    assert ("animal carcass", 1) in onto.alias_pairs and not any(a == "produit animal" for a, _ in onto.alias_pairs)
    record = onto.metadata["records"]["eurovoc:2"]
    assert record["broader"] == ["animal product"] and record["domains"] == ["agri-foodstuffs"] and record["related"] == ["meat carcase grading"]


def test_pubmed_parser_and_md5(tmp_path: Path) -> None:
    from vsa_embed.data.pubmed import iter_abstracts, verify_md5
    xml = ("<PubmedArticleSet><PubmedArticle><MedlineCitation><PMID>1</PMID><Article><ArticleTitle>Aspirin <i>in vivo</i>"
           "</ArticleTitle><Abstract><AbstractText Label='A'>First part.</AbstractText><AbstractText>Second.</AbstractText>"
           "</Abstract><Language>eng</Language></Article><ChemicalList><Chemical><NameOfSubstance UI='D1'>Aspirin"
           "</NameOfSubstance></Chemical></ChemicalList></MedlineCitation></PubmedArticle><PubmedArticle><MedlineCitation>"
           "<PMID>2</PMID><Article><ArticleTitle>No abstract</ArticleTitle></Article></MedlineCitation></PubmedArticle>"
           "</PubmedArticleSet>")
    path = tmp_path / "p.xml.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(xml)
    rows = list(iter_abstracts(path))
    assert len(rows) == 1 and rows[0]["title"] == "Aspirin in vivo" and rows[0]["abstract"] == "First part. Second."
    assert rows[0]["chemicals"] == ["Aspirin"]
    assert not verify_md5(path)
    Path(str(path) + ".md5").write_text(f"MD5(p.xml.gz)= {hashlib.md5(path.read_bytes()).hexdigest()}\n")
    assert verify_md5(path)


def test_legal_definitions_and_product_text() -> None:
    text = ("For the purposes of this Regulation: (a) ‘consumer’ means any natural person who is acting for purposes outside "
            "his trade; (b) 'trader' means any person acting for purposes relating to his business.")
    assert extract_definitions(text) == [("consumer", "any natural person who is acting for purposes outside his trade"),
                                         ("trader", "any person acting for purposes relating to his business")]
    row = {"product_title": "Lodge <b>Cast Iron</b> Skillet", "product_brand": "Lodge", "product_color": None,
           "product_bullet_point": "Pre-seasoned\n\nOven safe", "product_description": "Great &amp; durable<br>pan"}
    assert product_text(row) == "Lodge Cast Iron Skillet\nBrand: Lodge\n- Pre-seasoned\n- Oven safe\nGreat & durable pan"


# -- the builder end to end (T5, tiny) -------------------------------------------------------------

def _tokenizer_available(name: str) -> bool:
    try:
        from transformers import AutoTokenizer
        AutoTokenizer.from_pretrained(name, local_files_only=True)
        return True
    except (OSError, ImportError):
        return False


@pytest.mark.skipif(not _tokenizer_available("gpt2"), reason="gpt2 tokenizer not cached")
def test_track_builder_end_to_end_on_a_tiny_glossary(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from vsa_embed.experiments.track_corpus import run
    rng = random.Random(0)
    words = "the river city school garden market history music science water light energy family story".split()
    general = [" ".join(rng.choice(words) for _ in range(120)) + "." for _ in range(80)]
    pq.write_table(pa.table({"text": general}), tmp_path / "general.parquet")
    config = {
        "experiment": "t5-tiny", "track": "t5", "seed": 3, "tokenizer": "gpt2", "workers": 1, "items_dir": str(tmp_path / "items"),
        "paths": {"data_root": str(tmp_path / "data"), "general_shards": [str(tmp_path / "general.parquet")]},
        "glossary": {"terms": 200, "zero_shot": 12, "heldout_fraction": 0.1, "zipf": 1.5, "eval_uniform_focus": 0.5,
                     "acronym_fraction": 0.3, "train_chars": 150_000, "eval_chars": 60_000, "forbidden_general_docs": 40},
        "ontology": {"max_atomics": 8192, "max_degree": 16},
        "data": {"eval_tokens": 100_000, "contamination_docs": 40, "presample_tokens": 0, "holdout_fraction": 0.1,
                 "holdout_min_count": 5, "general_skip_docs": 0, "general_eval_docs": 20, "train_domain_tokens": 30_000,
                 "train_total_tokens": 45_000, "min_subtokens": 2, "cardinality_docs": 30},
        "items": {"train_terms_per_frequency_bin": 5, "relations_per_term": 2},
        "cardinality_tokenizers": ["gpt2"],
    }
    summary = run(config, tmp_path / "run")
    for name in ("resolved_config.yaml", "manifest.json", "summary.json", "report.md", "cardinality.json", "feasibility.json",
                 "holdout_concepts.txt"):
        assert (tmp_path / "run" / name).exists(), name
    onto = torch.load(tmp_path / "data" / "ontology.pt", weights_only=False)
    c3_keys = {"entry_count", "atomic_count", "relation_count", "offsets", "relations", "fillers", "heldout_entries",
               "train_frequency", "alias_table_sha256", "holdout_sha256", "relation_names", "atomic_names",
               "entry_concepts", "concept_names"}
    assert c3_keys <= set(onto)
    held = set(onto["heldout_entries"])
    assert set(onto["synthetic_entries"]) <= held and len(onto["synthetic_entries"]) == 12
    assert set(onto["heldout_real_entries"]) | set(onto["synthetic_entries"]) == held
    train = TokenCorpus.open(tmp_path / "data" / "train")
    evaluation = TokenCorpus.open(tmp_path / "data" / "eval")
    assert not set(train.spans["entry"].tolist()) & held                      # linker holdout
    assert set(evaluation.spans["entry"].tolist()) & set(onto["heldout_real_entries"])
    mix = summary["mix"]
    assert mix["domain_tokens"] >= 30_000 and mix["domain_tokens"] + mix["general_tokens"] >= 45_000
    assert train.manifest["tokens"] == mix["domain_tokens"] + mix["general_tokens"]
    record = json.loads((tmp_path / "items" / "holdout.json").read_text())
    assert record["holdout_sha256"] == onto["holdout_sha256"] and record["synthetic_concepts"] == 12
    for name, info in record["files"].items():
        assert hashlib.sha256((tmp_path / "items" / name).read_bytes()).hexdigest() == info["sha256"]
    names = (tmp_path / "items" / "holdout_concepts.txt").read_text().split("\n")[:-1]
    assert hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest() == record["holdout_sha256"]
    assert summary["feasibility"]["verdict"] in {"feasible", "infeasible"}
