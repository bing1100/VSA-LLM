"""E9 novelty screen (decision 55) and the T7 new-vocabulary MeSH adapter: proxies, rank agreement, P0 slice
configs, SCR parsing, the mention counter, the frequency selection and the frame ontology."""

import gzip
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from vsa_embed.experiments import e9_novelty as nov
from vsa_embed.ontologies.mesh import _parse
from vsa_embed.ontologies.mesh_novel import (MentionCounter, NovelVocabularyPolicy, build_novel_ontology, candidate_aliases,
                                             count_mentions, mention_key, parse_supplementary, read_selection,
                                             select_aliases, write_selection)
from vsa_embed.span_channel import AliasTable

# -- rank agreement --------------------------------------------------------------------------------------------------


def test_kendall_spearman_and_exact_p() -> None:
    assert nov.kendall_tau([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert nov.kendall_tau([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    assert nov.spearman([1, 2, 3, 4], [1, 4, 9, 16]) == pytest.approx(1.0)
    assert nov.permutation_p([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1 / 24)
    # a tie in the target (two null tracks): τ-b of a perfect proxy is 5 / √30, and 2 of 24 orderings reach it
    assert nov.kendall_tau([4, 3, 2, 1], [3, 2, 0, 0]) == pytest.approx(5 / math.sqrt(30))
    assert nov.permutation_p([4, 3, 2, 1], [3, 2, 0, 0]) == pytest.approx(2 / 24)
    assert math.isnan(nov.kendall_tau([1, 1, 1], [1, 2, 3]))


def test_agreement_orients_the_proxy_and_ties_null_gains() -> None:
    novelty = [-0.76, -0.48, -0.066, -0.007]                # T5, T4, T1, WordNet
    general_rate = [0.0, 0.036, 0.38, 0.91]                 # shrinks with novelty
    assert nov.agreement(general_rate, novelty, -1)["kendall_tau"] == pytest.approx(1.0)
    gain = [-0.075, -0.012, -0.0001, 0.00015]               # the last two are null
    tied = nov.agreement([15.3, 5.3, 1.26, 1.07], gain, 1, null=0.001)
    assert tied["kendall_tau"] == pytest.approx(5 / math.sqrt(30)) and tied["permutation_p"] == pytest.approx(2 / 24)


# -- proxies -----------------------------------------------------------------------------------------------------------


def _spans(rows: list[tuple[int, int, int, int]]) -> dict[str, np.ndarray]:
    """(start, end, entry, length) → a span table (inject = end)."""
    a = np.asarray(rows, dtype=np.int64).reshape(-1, 4)
    return {"start": a[:, 0], "end": a[:, 1], "inject": a[:, 1], "entry": a[:, 2], "length": a[:, 3]}


def test_span_proxies_counts_windows_lengths_and_general_frequency() -> None:
    # windows [0, 10) and [20, 30) (length 10); the span at 12 lies outside, the 1-subtoken span is below ℓ_min
    spans = _spans([(1, 3, 0, 3), (5, 6, 1, 2), (12, 13, 2, 2), (21, 25, 1, 5), (26, 26, 3, 1)])
    general = _spans([(0, 1, 1, 2), (4, 4, 0, 1), (50, 51, 1, 2)])          # entry 0 only at length 1; one span past the sample
    out = nov.span_proxies(spans, [0, 20], general, general_tokens=40, length=10)
    assert out["linked_spans"] == 3 and out["linked_entries"] == 2
    assert out["subtokens_per_span"] == pytest.approx((3 + 2 + 5) / 3)
    assert out["domain_spans_per_1k"] == pytest.approx(1e3 * 3 / 20)
    assert out["general_spans_per_1k"] == pytest.approx(1e3 * 1 / 40)
    assert out["general_to_domain"] == pytest.approx((1 / 40) / (3 / 20))
    assert out["general_absent_share"] == pytest.approx(1 / 3)                   # entry 0 (one occurrence) is absent
    assert nov.span_proxies(spans, [0, 20], None, 0, length=10)["general_to_domain"] is None


def _metrics(path: Path, losses: dict[str, float], counts: dict[str, int] | None = None, tokens: int = 100) -> None:
    path.mkdir(parents=True, exist_ok=True)
    rows = [{"type": "eval", "tokens": t, "stratum": s, "loss": v * (2 if t == 0 and tokens else 1),
             "stratum_tokens": (counts or {}).get(s, 10)} for t in (0, tokens) for s, v in losses.items()]
    (path / "metrics.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_novelty_gain_and_loss_proxies(tmp_path: Path) -> None:
    strata = {s: 2.0 for s in nov.NOVELTY_STRATA} | {"inside": 1.5, "unlinked": 2.5, "after": 2.0}
    _metrics(tmp_path / "t" / "H-frozen-P0-s1", strata, {"inside": 30}, tokens=0)
    _metrics(tmp_path / "t" / "H-full-C0p-s1", {s: v / 2 for s, v in strata.items()})
    _metrics(tmp_path / "t" / "H-full-C5-s1", {s: v / 2 * (0.9 if s.startswith("after_") else 1) for s, v in strata.items()})
    out = nov.novelty_and_gain(tmp_path, "t", "H")
    assert out["novelty_mean"] == pytest.approx(-0.5) and out["gain_mean"] == pytest.approx(-0.1)
    assert out["gain"]["unlinked"] == pytest.approx(0.0)
    p0 = nov.final_losses(tmp_path / "t" / "H-frozen-P0-s1")        # the last evaluation (tokens 0 here: P0)
    proxies = nov.loss_proxies(p0, linked_spans=10)
    assert proxies["inside_loss"] == pytest.approx(1.5) and proxies["after_unlinked"] == pytest.approx(0.8)
    assert proxies["inside_nats_per_span"] == pytest.approx(1.5 * 30 / 10)


def test_p0_slice_config(tmp_path: Path) -> None:
    config = nov.p0_config(tmp_path / "root", name="cand", windows=256, eval_batch=2)
    assert config["train"]["eval_only"] is True and config["model"]["host_mode"] == "frozen"
    assert config["eval"]["windows"] == 256 and config["eval"]["batch"] == 2
    assert config["data"]["train"] == config["data"]["eval"] == str(tmp_path / "root" / "eval")
    assert config["data"]["ontology"] == str(tmp_path / "root" / "ontology.pt")
    assert config["channel"]["mode"] == "none" and config["experiment"] == "e9-novelty-cand-SmolLM2-360M-P0"


# -- the T7 MeSH adapter -----------------------------------------------------------------------------------------------


def _term(text: str, *, tag: str = "NON", preferred: bool = False) -> str:
    return (f'<Term ConceptPreferredTermYN="N" IsPermutedTermYN="N" LexicalTag="{tag}" '
            f'RecordPreferredTermYN="{"Y" if preferred else "N"}"><String>{text}</String></Term>')


def _descriptor(ui: str, name: str, trees: list[str], *, year: int = 2000, actions: list[str] = ()) -> str:
    pa = "".join(f"<PharmacologicalAction><DescriptorReferredTo><DescriptorUI>{a}</DescriptorUI></DescriptorReferredTo>"
                 "</PharmacologicalAction>" for a in actions)
    return (f'<DescriptorRecord DescriptorClass="1"><DescriptorUI>{ui}</DescriptorUI><DescriptorName><String>{name}</String>'
            f'</DescriptorName><DateIntroduced><Year>{year}</Year></DateIntroduced>'
            f'<PharmacologicalActionList>{pa}</PharmacologicalActionList><TreeNumberList>'
            + "".join(f"<TreeNumber>{t}</TreeNumber>" for t in trees)
            + f'</TreeNumberList><ConceptList><Concept PreferredConceptYN="Y"><TermList>{_term(name, preferred=True)}'
            "</TermList></Concept></ConceptList></DescriptorRecord>")


def _scr(ui: str, name: str, cls: str, year: int, mapped: list[str], actions: list[str], terms: list[str]) -> str:
    mapped_xml = "".join(f"<HeadingMappedTo><DescriptorReferredTo><DescriptorUI>{m}</DescriptorUI></DescriptorReferredTo>"
                         "</HeadingMappedTo>" for m in mapped)
    pa = "".join(f"<PharmacologicalAction><DescriptorReferredTo><DescriptorUI>{a}</DescriptorUI></DescriptorReferredTo>"
                 "</PharmacologicalAction>" for a in actions)
    return (f'<SupplementalRecord SCRClass="{cls}"><SupplementalRecordUI>{ui}</SupplementalRecordUI><SupplementalRecordName>'
            f'<String>{name}</String></SupplementalRecordName><DateIntroduced><Year>{year}</Year></DateIntroduced>'
            f'<HeadingMappedToList>{mapped_xml}</HeadingMappedToList><PharmacologicalActionList>{pa}</PharmacologicalActionList>'
            f'<ConceptList><Concept PreferredConceptYN="Y"><TermList>{"".join(terms)}</TermList></Concept></ConceptList>'
            "</SupplementalRecord>")


@pytest.fixture
def mesh_files(tmp_path: Path) -> tuple[Path, Path]:
    descriptors = [
        _descriptor("D001", "Antibodies", ["D12.776.124"]),
        _descriptor("D002", "Antibodies, Bispecific", ["D12.776.124.050"]),
        _descriptor("D003", "Antineoplastic Agents", ["D27.505.954.248"]),
        _descriptor("D004", "Cuproptosis", ["G04.146"], year=2025),
    ]
    scrs = [
        _scr("C001", "teclistamab", "1", 2022, ["*D002"], ["D003"],
             [_term("teclistamab", preferred=True), _term("Tecvayli", tag="TRD"), _term("JNJ", tag="ABB")]),
        _scr("C002", "olezarsen", "1", 2024, ["D001"], [], [_term("olezarsen", preferred=True), _term("ab", preferred=False)]),
        _scr("C003", "Bisgaard taxon", "4", 2025, ["D001"], [], [_term("Bisgaard taxon", preferred=True)]),
        _scr("C004", "MVPP protocol", "2", 2024, ["D003"], [], [_term("MVPP protocol", preferred=True)]),
    ]
    d, s = tmp_path / "desc.xml.gz", tmp_path / "supp.xml.gz"
    with gzip.open(d, "wt") as handle:
        handle.write('<?xml version="1.0"?><DescriptorRecordSet>' + "".join(descriptors) + "</DescriptorRecordSet>")
    with gzip.open(s, "wt") as handle:
        handle.write('<?xml version="1.0"?><SupplementalRecordSet>' + "".join(scrs) + "</SupplementalRecordSet>")
    return d, s


def test_parse_supplementary_and_descriptor_years(mesh_files) -> None:
    d, s = mesh_files
    scrs = {r["ui"]: r for r in parse_supplementary(s)}
    assert scrs["C001"]["mapped"] == ["D002"] and scrs["C001"]["mapped_major"] == [True]
    assert scrs["C001"]["actions"] == ["D003"] and scrs["C001"]["introduced"] == 2022 and scrs["C001"]["scr_class"] == "1"
    assert [t["lexical_tag"] for t in scrs["C001"]["term_info"]] == ["NON", "TRD", "ABB"]
    assert {r["ui"]: r["introduced"] for r in _parse(d)}["D004"] == 2025


def test_candidate_aliases_policy(mesh_files) -> None:
    d, s = mesh_files
    descriptors, scrs = _parse(d), parse_supplementary(s)
    pairs, stats = candidate_aliases(descriptors, scrs, NovelVocabularyPolicy())
    aliases = {a: ui for a, ui in pairs}
    assert aliases["teclistamab"] == "C001" and aliases["Tecvayli"] == "C001"
    assert "JNJ" not in aliases and "ab" not in aliases                      # abbreviation, too short
    assert "MVPP protocol" not in aliases and aliases["Bisgaard taxon"] == "C003"   # class 2 (protocol) excluded
    assert aliases["Cuproptosis"] == "D004"
    recent = dict(candidate_aliases(descriptors, scrs, NovelVocabularyPolicy(descriptor_min_year=2024, scr_min_year=2024))[0])
    assert "teclistamab" not in recent and "olezarsen" in recent and "Cuproptosis" in recent and "Antibodies" not in recent
    assert stats["scrs_excluded_class"] == 1


def test_mention_counter() -> None:
    assert mention_key("N,N-Dimethyl  Tryptamine") == "n , n - dimethyl tryptamine"
    counter = MentionCounter([mention_key(a) for a in ("teclistamab", "datopotamab deruxtecan", "dato", "n,n-dimethyl")])
    counts = counter.count("Teclistamab and datopotamab deruxtecan (Dato-DXd) vs N, N-dimethyl; teclistamabs are not.")
    assert counts == Counter({"teclistamab": 1, "datopotamab deruxtecan": 1, "dato": 1, "n , n - dimethyl": 1})
    assert counter.mentions("TECLISTAMAB") and not counter.mentions("teclist amab")
    total, documents = count_mentions(["teclistamab", "x", "teclistamab teclistamab"], ["teclistamab"])
    assert total["teclistamab"] == 3 and documents == 3


def test_select_aliases_thresholds_shared_keys_and_frozen_file(tmp_path: Path) -> None:
    pairs = [("teclistamab", "C001"), ("Tecvayli", "C001"), ("Unused Synonym", "C001"), ("olezarsen", "C002"),
             ("Shared Name", "C002"), ("shared  name", "C003"), ("aspirin-like", "C003"), ("aspirinoid", "C003"),
             ("rarename", "C004")]
    domain = Counter({"teclistamab": 9, "tecvayli": 2, "olezarsen": 5, "shared name": 50, "aspirinoid": 40, "rarename": 2})
    general = Counter({"aspirin - like": 3})
    rows, stats = select_aliases(pairs, domain, general, NovelVocabularyPolicy(min_domain_mentions=5, max_general_mentions=0))
    # per record: C001's rare second name counts towards its total (2 + 9); C003 is out because its *other* name is
    # common in general text, though the name that occurs in the domain never occurs there
    assert [(r["ui"], r["alias"]) for r in rows] == [("C001", "Tecvayli"), ("C001", "teclistamab"), ("C002", "olezarsen")]
    assert stats == {"dropped_shared_key": 2, "records_general_frequent": 1, "records_domain_rare": 1,
                     "aliases_unseen_in_domain": 1, "selected_aliases": 3, "selected_records": 2}
    digest = write_selection(tmp_path / "selection.tsv", rows)
    assert read_selection(tmp_path / "selection.tsv", expected_sha256=digest) == rows
    with pytest.raises(ValueError, match="frozen selection changed"):
        read_selection(tmp_path / "selection.tsv", expected_sha256="0" * 64)


def test_novel_ontology_frames_and_fallback(mesh_files) -> None:
    d, s = mesh_files
    descriptors, scrs = _parse(d), parse_supplementary(s)
    selection = [{"alias": "teclistamab", "ui": "C001"}, {"alias": "Tecvayli", "ui": "C001"},
                 {"alias": "olezarsen", "ui": "C002"}, {"alias": "Cuproptosis", "ui": "D004"}]
    onto = build_novel_ontology(descriptors, scrs, selection)
    assert onto.concept_names == ["C001", "C002", "D004"]
    assert onto.metadata["kind"] == ["scr", "scr", "descriptor"] and onto.metadata["headings"][0] == "teclistamab"

    def frame(ui: str) -> set[tuple[str, str]]:
        return {(onto.relation_names[r], onto.atomic_names[a]) for r, a in onto.frames[onto.concept_names.index(ui)]}

    assert frame("C001") == {("mapped_to", "mesh:D002"), ("pharmacological_action", "mesh:D003"), ("record_class", "class:chemical"),
                             ("branch_top", "branch:D"), ("branch_second", "branch:D12")}
    assert ("branch_second", "branch:G04") in frame("D004")
    table = AliasTable.from_pairs(onto.alias_pairs)
    assert table.alias_to_entry["tecvayli"] == table.alias_to_entry["teclistamab"]
    # a dictionary too small for D002 falls back to its tree parent D001 (D12.776.124)
    small = build_novel_ontology(descriptors, scrs, selection, max_atomics=6)
    fillers = {small.atomic_names[a] for r, a in small.frames[0] if small.relation_names[r] == "mapped_to"}
    assert fillers <= {"mesh:D001", "mesh:D002"} and len(small.atomic_names) <= 6
    with pytest.raises(ValueError, match="not in the MeSH files"):
        build_novel_ontology(descriptors, scrs, [{"alias": "x", "ui": "C999"}])


# -- T7 corpus streams: parallel counts, mention filter, minimum PMID -------------------------------------------------


def _parquet(path: Path, rows: list[dict]) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.table({k: [r[k] for r in rows] for k in rows[0]}), path, row_group_size=2)
    return path


def test_parallel_counts_follow_the_stream_order(tmp_path: Path) -> None:
    from vsa_embed.data.pubmed import pmid_bucket
    from vsa_embed.experiments.t1_open_corpus import pubmed_documents
    from vsa_embed.ontologies.mesh_novel import count_general_mentions, count_pubmed_mentions

    train = [p for p in range(1000, 1200) if pmid_bucket(p) >= 800][:5]
    newer = _parquet(tmp_path / "update.parquet", [{"pmid": train[0], "text": "olezarsen olezarsen (revised)"},
                                                    {"pmid": train[1], "text": "olezarsen"}, {"pmid": 5, "text": "olezarsen"}])
    older = _parquet(tmp_path / "base.parquet", [{"pmid": train[0], "text": "olezarsen (first version)"},
                                                  {"pmid": train[2], "text": "teclistamab olezarsen"},
                                                  {"pmid": train[3], "text": "nothing"}])
    paths = [newer, older]
    counts, docs = count_pubmed_mentions(paths, ["olezarsen", "teclistamab"], eval_buckets=800, min_pmid=100, workers=1)
    # the revised record wins (newest file first), PMID 5 is below the minimum: 2 + 1 + 1 olezarsen
    assert counts == Counter({"olezarsen": 4, "teclistamab": 1}) and docs == 4
    counter = MentionCounter(["olezarsen", "teclistamab"])
    serial: Counter = Counter()
    for text in pubmed_documents(paths, eval_buckets=800, split="train", min_pmid=100):
        counter.count(text, serial)
    assert serial == counts                                          # the screen counts what the corpus stream reads
    kept = list(pubmed_documents(paths, eval_buckets=800, split="train", min_pmid=100, keep=counter.mentions))
    assert kept == ["olezarsen olezarsen (revised)", "olezarsen", "teclistamab olezarsen"]

    shards = [_parquet(tmp_path / f"g{i}.parquet", [{"text": f"doc {i}-{j} olezarsen"} for j in range(5)]) for i in range(2)]
    general, n = count_general_mentions([str(s) for s in shards], ["olezarsen"], skip=3, limit=4, workers=1, rows_per_job=1)
    assert n == 4 and general["olezarsen"] == 4                      # documents 3-4 of shard 0 and 0-1 of shard 1


def test_t7_dry_run_plans_without_queueing(tmp_path: Path) -> None:
    import torch
    import yaml
    from vsa_embed.experiments import e9_plan
    from vsa_embed.experiments.e9_tracks import TRACKS
    torch.save({"entry_count": 6000, "atomic_count": 4000, "relation_count": 7}, tmp_path / "counts.pt")
    paths = e9_plan.write_stage("t7", hosts=["SmolLM2-360M"], models=list(e9_plan.MODELS), seeds=[1, 2, 3], track="t7",
                                data_root=tmp_path / "t7", counts_ontology=tmp_path / "counts.pt", root=tmp_path / "e9")
    assert len(paths) == 1 + 3 * 3
    c5 = yaml.safe_load(next(p for p in paths if p.stem == "SmolLM2-360M-full-C5-s2").read_text())
    assert c5["e9_track"] == "t7" and c5["data"]["eval"] == str(tmp_path / "t7" / "eval-pubmed") and c5["eval"]["windows"] == 2048
    planned: list = []
    queue = tmp_path / "jobs"
    queued = e9_plan.queue_jobs(paths, "t7", 62, track="t7", root=tmp_path / "e9", queue_dir=queue, plan=planned)
    assert queued == [] and not queue.exists()                       # nothing queued, no alias table written
    names = {name: (level, command) for name, level, command in planned}
    assert names["t7-SmolLM2-360M-full-C5-s3"][0] == 62 and names["t7-SmolLM2-360M-full-C5-s3-edit-int4"][0] == 63
    assert names["t7-quant-s1-2-3"][0] == 64 and names["t7-report-s1-2-3"][0] == 65
    assert not any(name.endswith("zeroshot") for name in names)       # T7 has no WP-C7 zero-shot items
    edit = names["t7-SmolLM2-360M-full-C5-s1-edit"][1]
    assert edit[edit.index("--alias-table") + 1] == str(TRACKS["t7"].alias_table_path)
    assert edit[edit.index("--new-items") + 1].endswith("new-words-t7-smollm2-v1")
    configs = {Path(p): yaml.safe_load(Path(p).read_text()) for p in paths}
    train = names["t7-SmolLM2-360M-full-C5-s1"][1]
    assert e9_plan.job_hours("t7-SmolLM2-360M-full-C5-s1", train, configs) == pytest.approx(50e6 / 12_400 / 3600)   # ≈ 67 min
    p0 = names["t7-SmolLM2-360M-frozen-P0-s1"][1]
    assert e9_plan.job_hours("t7-SmolLM2-360M-frozen-P0-s1", p0, configs) == e9_plan.EVAL_HOURS["SmolLM2-360M"]["P0"]
    assert e9_plan.job_hours("t7-SmolLM2-360M-full-C5-s1-probes-int4", [], configs) == e9_plan.EVAL_HOURS["SmolLM2-360M"]["probes-int4"]


def test_t7_lexicon_wording() -> None:
    from vsa_embed.experiments.e9_tracks import T7_TEMPLATES, TRACKS, TrackLexicon
    spec = TRACKS["t7"]
    lexicon = TrackLexicon("t7", T7_TEMPLATES, {"mesh:D1": "Antibodies, Bispecific", "class:chemical": "chemical",
                                                "mesh:D3": "Antineoplastic Agents"},
                           category_relations=spec.category_relations, kept_relations=frozenset(spec.kept_relations),
                           edit_relations=spec.edit_relations)
    assert lexicon.prompts("*", "mapped_to") == ["{x} is a kind of", "In MeSH, {x} is indexed under"]
    assert lexicon.statements("*", "mapped_to") == ["{x} is a type of"]
    assert lexicon.statement_answer("mapped_to", "Antibodies, Bispecific") == " Antibodies, Bispecific."
    assert lexicon.statement_answer("record_class", "chemical") == " chemical."
    assert lexicon.prompts("*", "branch_top") is None and "pharmacological_action" in T7_TEMPLATES
    assert spec.eval_split == "eval-pubmed" and spec.zeroshot_items is None


def test_t1_stream_signature_unchanged_without_t7_keys() -> None:
    from vsa_embed.experiments.t1_open_corpus import TrackDocuments
    common = dict(pubmed_paths=[Path("a.parquet")], eval_buckets=800, general_shards=["g.parquet"], general_skip=10,
                  eval_general_docs=5)
    base, filtered = TrackDocuments(**common), TrackDocuments(**common, mention_filter=MentionCounter(["x"]),
                                                               mention_digest="abc", min_pmid=7)
    assert base.signature("train") != filtered.signature("train")
    assert base.signature("eval-general") == filtered.signature("eval-general")    # the general stream is shared
