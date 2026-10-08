"""TK-H3L (decision 63): evaluation data for the *learn* tool and holdout H3 — the common item formats, snapshots and
splits, the ICD-10-CM FY2027 builder, the MeSH 2025 → 2026 time split, and the MedConceptsQA / OET / TaxoExpan / TMN
adapters. Synthetic fixtures only; no network, no licensed data."""

import gzip
import json
import pickle
import zipfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vsa_embed.benchmarks import learn_data as ld
from vsa_embed.benchmarks import learn_icd10cm as icd
from vsa_embed.benchmarks import learn_mesh as lm
from vsa_embed.benchmarks import learn_public as lp

# -- common format ----------------------------------------------------------------------------------------------------------


def test_split_of_is_deterministic_and_near_the_fraction() -> None:
    records = [f"D{i:06d}" for i in range(4000)]
    splits = [ld.split_of(r) for r in records]
    assert splits == [ld.split_of(r) for r in records]
    assert 0.17 < splits.count("dev") / len(splits) < 0.23
    assert ld.split_of("D000001", salt="other") in ("dev", "test")


def test_placement_item_dedupes_and_validates() -> None:
    item = ld.placement_item(set_name="s", record="R1", name="Aerogels", aliases=["Aerogels", "aerogel", "Aerogel", ""],
                             gold_parents=["P", "P", "Q"], gold_relations=[("parent", "P"), ("parent", "P"), ("see_also", "Q")])
    assert item["aliases"] == ["aerogel"] and item["gold_parents"] == ["P", "Q"]
    assert item["gold_relations"] == [["parent", "P"], ["see_also", "Q"]] and item["candidates"] is None
    assert set(item) == set(ld.PLACEMENT_KEYS) and item["id"] == "s:R1"
    with pytest.raises(ValueError):
        ld.validate_placement({**item, "gold_relations": [["parent"]]})
    with pytest.raises(ValueError):
        ld.validate_placement({k: v for k, v in item.items() if k != "evidence"})


def test_rank_item_rejects_duplicates_and_bad_answers() -> None:
    item = ld.rank_item(set_name="s", item_id="i", context="Q?", options=["a", "b"], answer=1,
                        terms=[{"surface": "x", "concept": None}])
    assert item["kind"] == "rank" and item["answer"] == 1
    with pytest.raises(ValueError):
        ld.rank_item(set_name="s", item_id="i", context="Q?", options=["a", "A"], answer=0)
    with pytest.raises(ValueError):
        ld.rank_item(set_name="s", item_id="i", context="Q?", options=["a", "b"], answer=2)
    with pytest.raises(ValueError):
        ld.rank_item(set_name="s", item_id="i", context="Q?", options=["a", "b"], answer=0, terms=[{"surface": "x"}])


def test_write_jsonl_round_trip_and_reproducible_gzip(tmp_path: Path) -> None:
    items = [{"b": 1, "a": "é"}, {"a": 2}]
    first = ld.write_jsonl(tmp_path / "x.jsonl.gz", items)
    data = (tmp_path / "x.jsonl.gz").read_bytes()
    second = ld.write_jsonl(tmp_path / "x.jsonl.gz", items)
    assert first == second and (tmp_path / "x.jsonl.gz").read_bytes() == data
    plain = ld.write_jsonl(tmp_path / "x.jsonl", items)
    assert plain["sha256"] == first["sha256"] and plain["items"] == 2
    assert list(ld.read_jsonl(tmp_path / "x.jsonl.gz")) == items
    assert (tmp_path / "x.jsonl").read_text().splitlines()[0] == '{"a": "é", "b": 1}'


def test_snapshot_round_trip_and_dangling_edges(tmp_path: Path) -> None:
    nodes = [ld.snapshot_node("B", "b", aliases=["b", "bee"]), ld.snapshot_node("A", "a", kind="root")]
    info = ld.write_snapshot(tmp_path / "snap", nodes, [("B", "parent", "A"), ("B", "parent", "A")], description={"x": 1})
    assert info["edges"]["items"] == 1 and info["relations"] == {"parent": 1} and info["x"] == 1
    loaded, edges = ld.load_snapshot(tmp_path / "snap")
    assert loaded["B"]["aliases"] == ["bee"] and edges == [("B", "parent", "A")]
    with pytest.raises(ValueError):
        ld.write_snapshot(tmp_path / "bad", nodes, [("B", "parent", "Z")], description={})


def test_sources_are_pinned_and_fetch_skips_present_files(tmp_path: Path) -> None:
    assert all(len(s.sha256) == 64 and s.url.startswith("https://") for s in ld.SOURCES)
    assert len({s.path for s in ld.SOURCES}) == len(ld.SOURCES)
    path = tmp_path / "f.txt"
    path.write_text("hello")
    source = ld.Source("t", "https://example.invalid/f.txt", "f.txt", ld.file_sha256(path), "test")
    assert ld.fetch_source(source, tmp_path)["status"] == "present"     # no network when the pinned file is there


# -- ICD-10-CM (synthetic releases) ----------------------------------------------------------------------------------------------


def _order_line(n: int, code: str, billable: bool, title: str) -> str:
    return f"{n:05d} {code:<7} {int(billable)} {title[:60]:<60} {title}"


ORDER_2026 = [("I42", False, "Cardiomyopathy"), ("I420", True, "Dilated cardiomyopathy"),
              ("I421", True, "Obstructive hypertrophic cardiomyopathy"), ("I422", True, "Other hypertrophic cardiomyopathy"),
              ("I43", True, "Cardiomyopathy in diseases classified elsewhere"),
              ("K65", False, "Peritonitis"), ("K650", True, "Generalized acute peritonitis"),
              ("K651", True, "Peritoneal abscess"), ("K652", True, "Spontaneous bacterial peritonitis")]
ORDER_2027 = [("I42", False, "Cardiomyopathy"), ("I420", False, "Dilated cardiomyopathy"),
              ("I4200", True, "Dilated cardiomyopathy, unspecified"), ("I4201", True, "Familial-genetic dilated cardiomyopathy"),
              ("I4209", True, "Other dilated cardiomyopathy"),
              ("I421", True, "Obstructive hypertrophic cardiomyopathy"), ("I422", True, "Other hypertrophic cardiomyopathy"),
              ("I43", True, "Cardiomyopathy in diseases classified elsewhere"),
              ("K65", False, "Peritonitis"), ("K650", True, "Generalized acute peritonitis"),
              ("K651", True, "Peritoneal abscess"), ("K652", True, "Spontaneous bacterial peritonitis"),
              ("K6A", False, "Diseases of the pelvis, not elsewhere classified"), ("K6A0", False, "Pelvic abscess"),
              ("K6A01", True, "Prevesical abscess"), ("K6A09", True, "Other pelvic abscess")]


def _tabular(year: int, blocks: list[tuple[str, str, list[str]]]) -> str:
    def diag(name: str, desc: str, inner: str = "", notes: str = "") -> str:
        return f"<diag><name>{name}</name><desc>{desc}</desc>{notes}{inner}</diag>"

    chapters = {"I": ("9", "Diseases of the circulatory system (I00-I99)"), "K": ("11", "Diseases of the digestive system (K00-K95)")}
    out = [f'<?xml version="1.0"?><ICD10CM.tabular><version>{year}</version>']
    for letter, (number, title) in chapters.items():
        out.append(f"<chapter><name>{number}</name><desc>{title}</desc>")
        for block_id, block_title, categories in blocks:
            if block_id.startswith(letter):
                out.append(f'<section id="{block_id}"><desc>{block_title} ({block_id})</desc>')
                for cat in categories:
                    if cat == "I42":
                        inner = diag("I42.0", "Dilated cardiomyopathy",
                                     diag("I42.00", "Dilated cardiomyopathy, unspecified",
                                          notes="<inclusionTerm><note>Congestive cardiomyopathy, NOS</note></inclusionTerm>")
                                     if year == 2027 else "",
                                     notes="" if year == 2027 else "<inclusionTerm><note>Congestive cardiomyopathy</note></inclusionTerm>")
                        out.append(diag("I42", "Cardiomyopathy", inner, "<includes><note>cardiomyopathy NOS</note></includes>"))
                    elif cat == "K6A":
                        out.append(diag("K6A", "Diseases of the pelvis, not elsewhere classified",
                                        diag("K6A.0", "Pelvic abscess", diag("K6A.01", "Prevesical abscess",
                                             notes="<inclusionTerm><note>Retropubic abscess</note></inclusionTerm>"))))
                    else:
                        out.append(diag(cat, cat))
                out.append("</section>")
        out.append("</chapter>")
    out.append("</ICD10CM.tabular>")
    return "".join(out)


@pytest.fixture
def icd_releases() -> tuple[icd.Release, icd.Release]:
    before = icd.Release("fy2026", icd.parse_order([_order_line(i, *row) for i, row in enumerate(ORDER_2026, 1)]),
                         icd.parse_tabular(_tabular(2026, [("I30-I5A", "Other forms of heart disease", ["I42", "I43"]),
                                                           ("K65-K68", "Diseases of peritoneum", ["K65"])])))
    after = icd.Release("fy2027", icd.parse_order([_order_line(i, *row) for i, row in enumerate(ORDER_2027, 1)]),
                        icd.parse_tabular(_tabular(2027, [("I30-I5A", "Other forms of heart disease", ["I42", "I43"]),
                                                          ("K65-K6A", "Diseases of peritoneum and pelvis", ["K65", "K6A"])])))
    return after, before


def test_parse_order_fixed_width() -> None:
    codes = icd.parse_order([_order_line(1, "I4201", True, "Familial-genetic dilated cardiomyopathy"), "garbage", ""])
    assert codes == {"I42.01": {"code": "I4201", "billable": True, "short": "Familial-genetic dilated cardiomyopathy",
                                "long": "Familial-genetic dilated cardiomyopathy", "order": 1}}
    assert icd.dotted("I4201") == "I42.01" and icd.dotted("I42") == "I42" and icd.undotted("K6A.01") == "K6A01"


def test_hierarchy_blocks_and_chains(icd_releases) -> None:
    after, before = icd_releases
    assert icd.code_parent("I42.01", after) == "I42.0" and icd.code_parent("I42", after) == "block:I30-I5A"
    assert icd.chain("K6A.01", after) == ["K6A.0", "K6A", "block:K65-K6A", "chapter:11"]
    assert icd.block_map(after, before) == {"block:I30-I5A": "block:I30-I5A", "block:K65-K6A": "block:K65-K68"}
    definition, inclusion, notes = icd.definition_of("I42", after)
    assert definition == "Cardiomyopathy. Includes: cardiomyopathy NOS." and inclusion == []


def test_placement_items_use_the_nearest_fy2026_ancestor(icd_releases) -> None:
    after, before = icd_releases
    items = {i["record"]: i for i in icd.placement_items(after, before)}
    assert set(items) == {"I42.00", "I42.01", "I42.09", "K6A", "K6A.0", "K6A.01", "K6A.09"}
    assert items["I42.01"]["gold_parents"] == ["I42.0"] and items["I42.01"]["meta"]["parent_billable_in_fy2026"]
    assert items["K6A.01"]["gold_parents"] == ["block:K65-K68"] and items["K6A.01"]["meta"]["new_levels_above"] == 2
    assert items["K6A.01"]["aliases"] == ["Retropubic abscess"]
    assert items["K6A.01"]["definition"] == "Prevesical abscess. Inclusion terms: Retropubic abscess."
    assert ["chapter", "chapter:11"] in items["K6A"]["gold_relations"] and ["block", "block:K65-K68"] in items["K6A"]["gold_relations"]
    assert items["I42.00"]["aliases"] == ["Congestive cardiomyopathy, NOS"]
    assert all(i["candidates"] is None and i["meta"]["split"] in ("dev", "test") for i in items.values())


def test_choice_items_siblings_chapter_and_seed(icd_releases) -> None:
    after, before = icd_releases
    codes = ["I42.01", "K6A.01"]
    first = icd.choice_items(after, codes, seed=7)
    assert first == icd.choice_items(after, codes, seed=7) and len(first) == 4
    for item in first:
        meta = item["meta"]
        assert item["options"][item["answer"]] in (after.codes[meta["code"]]["long"], meta["code"])
        assert len(set(meta["option_codes"])) == 4 and meta["code"] in meta["option_codes"]
        for other in meta["option_codes"]:
            if other != meta["code"]:      # never an ancestor or descendant
                assert not icd.undotted(meta["code"]).startswith(icd.undotted(other))
                assert not icd.undotted(other).startswith(icd.undotted(meta["code"]))
    code2title = first[0]
    assert code2title["context"] == "What is the description of the medical code I42.01 in ICD10CM?"
    assert Counter(code2title["meta"]["distractor_source"].values()) == Counter({"sibling": 2, "chapter": 1})
    assert first[1]["options"] == code2title["meta"]["option_codes"] and first[1]["meta"]["direction"] == "title2code"


def test_snapshot_rows_close_over_nodes(icd_releases, tmp_path: Path) -> None:
    after, before = icd_releases
    nodes, edges = icd.snapshot_rows(before)
    info = ld.write_snapshot(tmp_path / "s", nodes, edges, description={})
    assert info["node_kinds"] == {"block": 2, "category": 3, "chapter": 2, "code": 6}
    assert ("I42.0", "parent", "I42") in edges and ("block:K65-K68", "parent", "chapter:11") in edges


def test_mrconso_parts_join_across_a_split_line(tmp_path: Path) -> None:
    text = "C1|ENG|P|L|PF|S|Y|A|||||ICD10CM|PT|I42.0|Dilated cardiomyopathy|0|N||\nC2|ENG|P|L|PF|S|Y|A|||||MSH|MH|D002311|Cardiomyopathy, Dilated|0|N||\n"
    cut = 30
    with gzip.open(tmp_path / "MRCONSO.RRF.aa.gz", "wt") as handle:
        handle.write(text[:cut])
    with gzip.open(tmp_path / "MRCONSO.RRF.ab.gz", "wt") as handle:
        handle.write(text[cut:])
    assert list(icd.mrconso_lines(tmp_path)) == text.splitlines()


def test_map_umls_parents_and_string_matches(icd_releases) -> None:
    after, before = icd_releases
    items = icd.placement_items(after, before)

    def row(cui: str, sab: str, tty: str, code: str, text: str, lat: str = "ENG", suppress: str = "N") -> str:
        return "|".join([cui, lat, "P", "L", "PF", "S", "Y", "A", "", "", "", sab, tty, code, text, "0", suppress, ""])

    lines = [row("C01", "ICD10CM", "PT", "I42.0", "Dilated cardiomyopathy"),
             row("C01", "MSH", "MH", "D002311", "Cardiomyopathy, Dilated"),
             row("C01", "SNOMEDCT_US", "PT", "399020009", "Dilated cardiomyopathy"),
             row("C01", "MSH", "QAB", "Q000000", "qualifier"),
             row("C02", "SNOMEDCT_US", "PT", "1234", "Prevesical abscess"),
             row("C03", "SNOMEDCT_US", "PT", "999", "Pelvic abscess", suppress="O")]
    result = icd.map_umls(items, after, before, lines, lines_again=lines)
    parents = {p["code"]: p for p in result["parents"]}
    assert parents["I42.0"]["mesh"] == {"D002311": "Cardiomyopathy, Dilated"} and "399020009" in parents["I42.0"]["snomed"]
    matches = {m["code"]: m for m in result["string_matches"]}
    assert matches["K6A.01"]["snomed"] == {"1234": "Prevesical abscess"} and matches["K6A.0"]["cuis"] == []
    assert result["summary"]["parent_codes_with_mesh"] == 1


# -- MeSH 2025 → 2026 ----------------------------------------------------------------------------------------------------------


def _term(text: str, *, tag: str = "NON", preferred: bool = False) -> str:
    return (f'<Term ConceptPreferredTermYN="N" IsPermutedTermYN="N" LexicalTag="{tag}" '
            f'RecordPreferredTermYN="{"Y" if preferred else "N"}"><String>{text}</String></Term>')


def _descriptor(ui: str, name: str, trees: list[str], *, terms: tuple[str, ...] = (), actions: tuple[str, ...] = (),
                note: str = "", year: int = 2000) -> str:
    pa_xml = "".join(f"<PharmacologicalAction><DescriptorReferredTo><DescriptorUI>{a}</DescriptorUI></DescriptorReferredTo>"
                     "</PharmacologicalAction>" for a in actions)
    return (f'<DescriptorRecord DescriptorClass="1"><DescriptorUI>{ui}</DescriptorUI><DescriptorName><String>{name}</String>'
            f'</DescriptorName><DateIntroduced><Year>{year}</Year></DateIntroduced>'
            f'<PharmacologicalActionList>{pa_xml}</PharmacologicalActionList><TreeNumberList>'
            + "".join(f"<TreeNumber>{t}</TreeNumber>" for t in trees)
            + f'</TreeNumberList><ConceptList><Concept PreferredConceptYN="Y"><ScopeNote>{note}</ScopeNote><TermList>'
            + _term(name, preferred=True) + "".join(_term(t) for t in terms) + "</TermList></Concept></ConceptList></DescriptorRecord>")


def _scr(ui: str, name: str, mapped: list[str], *, actions: tuple[str, ...] = (), terms: tuple[str, ...] = (),
         note: str = "", cls: str = "1", year: int = 2026) -> str:
    mapped_xml = "".join(f"<HeadingMappedTo><DescriptorReferredTo><DescriptorUI>{m}</DescriptorUI></DescriptorReferredTo>"
                         "</HeadingMappedTo>" for m in mapped)
    pa_xml = "".join(f"<PharmacologicalAction><DescriptorReferredTo><DescriptorUI>{a}</DescriptorUI></DescriptorReferredTo>"
                     "</PharmacologicalAction>" for a in actions)
    return (f'<SupplementalRecord SCRClass="{cls}"><SupplementalRecordUI>{ui}</SupplementalRecordUI><SupplementalRecordName>'
            f'<String>{name}</String></SupplementalRecordName><DateIntroduced><Year>{year}</Year></DateIntroduced>'
            f'<Note>{note}</Note><HeadingMappedToList>{mapped_xml}</HeadingMappedToList>'
            f'<PharmacologicalActionList>{pa_xml}</PharmacologicalActionList><ConceptList><Concept PreferredConceptYN="Y">'
            f'<TermList>{_term(name, preferred=True)}{"".join(_term(t) for t in terms)}</TermList></Concept></ConceptList>'
            "</SupplementalRecord>")


def _write(path: Path, root: str, records: list[str]) -> Path:
    with gzip.open(path, "wt") as handle:
        handle.write(f'<?xml version="1.0"?><{root}>' + "".join(records) + f"</{root}>")
    return path


BASE_DESC = [_descriptor("D001", "Chemicals", ["D01"]), _descriptor("D002", "Gels", ["D01.100"]),
             _descriptor("D003", "Antineoplastic Agents", ["D27"]), _descriptor("D004", "Diseases", ["C01"]),
             _descriptor("D005", "Infections", ["C01.200"])]


@pytest.fixture
def mesh_releases(tmp_path: Path) -> dict[str, dict[str, Path]]:
    before = {"desc": _write(tmp_path / "desc2025.gz", "DescriptorRecordSet", BASE_DESC),
              "supp": _write(tmp_path / "supp2025.gz", "SupplementalRecordSet", [
                  _scr("C100", "silica aerogel", ["D002"], year=2024), _scr("C101", "oldmab", ["D001"], year=2020),
                  _scr("C102", "gonemab", ["D001"], year=2020)])}
    after = {"desc": _write(tmp_path / "desc2026.gz", "DescriptorRecordSet", BASE_DESC[:3] + [
                 _descriptor("D004", "Diseases", ["C01"]), _descriptor("D005", "Infections", ["C01.200", "C01.300"]),
                 _descriptor("D006", "Silica Aerogel", ["D01.100.010"], terms=("Silica Aerogels",), note="A light solid.",
                             year=2026),
                 _descriptor("D007", "Pelvic Infections", ["C01.200.050"], year=2026),
                 _descriptor("D008", "Pelvic Abscess", ["C01.200.050.010"], year=2026)]),
             "supp": _write(tmp_path / "supp2026.gz", "SupplementalRecordSet", [
                 _scr("C101", "oldmab", ["D001", "D003"], year=2020),
                 _scr("C200", "newzumab", ["D003"], actions=("D003",), terms=("NZM-1",),
                      note="humanized monoclonal antibody; structure in first source"),
                 _scr("C201", "pelvizole", ["D008"], terms=("pelvizole sodium",)),
                 _scr("C202", "rarezide", ["D001"])])}
    return {"before": before, "after": after}


def test_mesh_diff_gold_and_new_edges(mesh_releases) -> None:
    from vsa_embed.ontologies import mesh as mm
    from vsa_embed.ontologies import mesh_novel as mn
    load = lambda p: sorted(mm._parse(p), key=lambda r: r["ui"])                      # noqa: E731
    supp = lambda p: sorted(mn.parse_supplementary(p), key=lambda r: r["ui"])          # noqa: E731
    b, a = mesh_releases["before"], mesh_releases["after"]
    changes = lm.diff(load(b["desc"]), supp(b["supp"]), load(a["desc"]), supp(a["supp"]))
    stats = changes["stats"]
    assert stats["new_descriptors"] == 3 and stats["new_scrs"] == 3 and stats["deleted_scrs"] == 2
    desc = {d["ui"]: d for d in changes["descriptors"]}
    assert desc["D006"]["gold_parents"] == ["D002"] and desc["D006"]["meta"]["promoted_from_scr_before"] == "C100"
    assert desc["D008"]["gold_parents"] == ["D005"] and desc["D008"]["meta"]["parents_new"] == ["D007"]
    scrs = {s["ui"]: s for s in changes["scrs"]}
    assert scrs["C200"]["relations"] == [("mapped_to", "D003"), ("pharmacological_action", "D003")]
    assert scrs["C201"]["gold_parents"] == ["D005"] and scrs["C201"]["meta"]["mapped_new"] == ["D008"]
    assert {(e["source"], e["relation"], e["target"]) for e in changes["new_edges"]} == {("C101", "mapped_to", "D003")}
    assert lm.nearest_existing(["X"], {"X": ["Y"], "Y": ["Z"]}, {"Z"}) == ["Z"]


def test_document_matcher_and_scan(tmp_path: Path) -> None:
    matcher = lm.DocumentMatcher(["pelvic abscess", "newzumab", "nzm - 1"])
    assert matcher.keys_in("Newzumab (NZM-1) treats pelvic abscess; newzumab again.") == {"newzumab", "nzm - 1", "pelvic abscess"}
    path = tmp_path / "a.parquet"
    pq.write_table(pa.table({"pmid": [50, 100, 101, 100], "year": [2026] * 4,
                             "text": ["newzumab", "newzumab here", "nothing", "newzumab revised"],
                             "mesh": [["D006"], ["D006"], ["D006", "D001"], []]}), path)
    mentions, indexed, documents = lm.scan_pubmed([path], ["newzumab"], ["D006"], min_pmid=60, workers=1)
    assert mentions == {"newzumab": {100}} and indexed == {"D006": {100, 101}} and documents == 3
    evidence = lm.evidence_for({"newzumab"}, mentions, indexed["D006"])
    assert evidence["pubmed_2025_26_docs"] == 1 and evidence["pubmed_2025_26_pmids"] == [100]
    assert evidence["pubmed_indexed_docs"] == 2


def test_mesh_build_end_to_end(mesh_releases, tmp_path: Path) -> None:
    texts = ["Newzumab and newzumab.", "NZM-1 was given.", "newzumab trial", "a newzumab study", "newzumab works",
             "Silica aerogels insulate.", "pelvic abscess drained", "Pelvizole for pelvic abscess", "pelvizole sodium"]
    path = tmp_path / "pubmed.parquet"
    pq.write_table(pa.table({"pmid": list(range(41_100_000, 41_100_000 + len(texts))), "year": [2026] * len(texts),
                             "text": texts, "mesh": [["D006"]] * len(texts)}), path)
    manifest = lm.build(tmp_path / "out", tmp_path / "local", workers=1, pubmed=[path], before=mesh_releases["before"],
                        after=mesh_releases["after"], tensor=True)
    items = {i["record"]: i for f in ("placement-dev.jsonl", "placement-test.jsonl", "placement-low-evidence.jsonl")
             for i in ld.read_jsonl(tmp_path / "out" / f)}
    assert set(items) == {"D006", "D007", "D008", "C200", "C201", "C202"}
    assert items["C200"]["evidence"]["pubmed_2025_26_docs"] == 5 and items["C200"]["meta"]["primary"]
    assert items["C200"]["definition"] == "humanized monoclonal antibody"
    assert items["C201"]["evidence"]["pubmed_2025_26_docs"] == 2 and not items["C201"]["meta"]["primary"]
    assert items["D006"]["evidence"]["pubmed_indexed_docs"] == len(texts) and items["D006"]["definition"] == "A light solid."
    assert items["C202"]["evidence"]["pubmed_2025_26_docs"] == 0
    assert manifest["stats"]["new_scrs"] == 3 and manifest["primary"]["items"] == 1
    assert manifest["snapshot"]["nodes"]["items"] == 5 and manifest["snapshot"]["ontology_pt"]["entries"] >= 5
    nodes, edges = ld.load_snapshot(tmp_path / "local" / "snapshot-2025")
    assert ("D005", "parent", "D004") in edges and "D006" not in nodes
    assert [json.loads(l)["source"] for l in (tmp_path / "out" / "new-edges.jsonl").read_text().splitlines()] == ["C101"]
    # the ≥ 1-abstract secondary set is a superset of the primary one; every item carries its T7 group
    min1 = {i["record"] for f in ("placement-min1-dev.jsonl", "placement-min1-test.jsonl") for i in ld.read_jsonl(tmp_path / "out" / f)}
    assert min1 == {"C200", "C201", "D006", "D008"} and manifest["secondary_min1"]["items"] == 4    # D007: 0 abstracts
    assert manifest["role"] == "primary" and "t7_policy" in manifest and manifest["t7_policy"]["field"].startswith("meta.t7_group")
    assert {i["meta"]["t7_group"] for i in items.values()} == {"eval"}
    # the same code builds the secondary 2024 → 2025 split under its own set name and snapshot year
    second = lm.build(tmp_path / "out24", tmp_path / "local24", workers=1, pubmed=[path], before=mesh_releases["before"],
                      after=mesh_releases["after"], tensor=False, split="2024-2025")
    assert second["set"] == "mesh-2024-2025" and second["role"] == "secondary" and "caveat" in second
    assert (tmp_path / "local24" / "snapshot-2024" / "nodes.jsonl.gz").exists()
    assert {i["set"] for i in ld.read_jsonl(tmp_path / "out24" / "placement-test.jsonl")} <= {"mesh-2024-2025"}


def test_t7_group_rule() -> None:
    assert lm.t7_group("scr", True, False) == "seen"
    assert lm.t7_group("scr", True, True) == "eval" and lm.t7_group("scr", False, False) == "eval"
    assert lm.t7_group("descriptor", True, False) == "eval"


# -- MedConceptsQA ----------------------------------------------------------------------------------------------------------------


def _mcqa_row(**override) -> dict:
    row = {"question_id": 7, "answer": "Brugada syndrome", "answer_id": "B", "option1": "Ventricular bigeminy",
           "option2": "Brugada syndrome", "option3": "Other specified cardiac arrhythmias", "option4": "Cardiac arrest",
           "question": "What is the description of the medical code I49.81 in ICD10CM?\nA. Ventricular bigeminy\nB. Brugada syndrome",
           "vocab": "ICD10CM", "level": "hard"}
    return {**row, **override}


def test_medconceptsqa_item_and_inconsistent_rows() -> None:
    item = lp.medconceptsqa_item(_mcqa_row(), level="hard", split="test", fy2026={"I49.8"}, fy2027={"I49.81"})
    assert item["context"] == "What is the description of the medical code I49.81 in ICD10CM?"
    assert item["answer"] == 1 and item["terms"] == [{"surface": "I49.81", "concept": "I49.81"}]
    assert item["meta"]["in_fy2026"] is False and item["meta"]["in_fy2027"] is True
    with pytest.raises(ValueError):
        lp.medconceptsqa_item(_mcqa_row(answer_id="A"), level="hard", split="test")
    with pytest.raises(ValueError):
        lp.medconceptsqa_item(_mcqa_row(option3="brugada syndrome"), level="hard", split="test")


# -- OET ------------------------------------------------------------------------------------------------------------------------


def _oet_row(concept: str, mention: str, parents: str, children: str = "", synonyms: str | None = None) -> dict:
    pairs = "|".join(f"{p}-{c or 'SCTID_NULL'}" for p in parents.split("|") for c in (children.split("|") if children else [""]))
    row = {"context_left": "left ", "mention": mention, "context_right": " right", "label_concept_UMLS": "C0000001",
           "label_concept": "SCTID-less", "label_concept_ori": concept, "label": "", "parents_concept": parents,
           "parents": "Parent (disorder)", "children_concept": children, "children": "", "parents-children_concept": pairs,
           "label_id": 9, "label_title": "NIL"}
    if synonyms is not None:
        row["synonyms"] = synonyms
    return row


def test_oet_items_group_mentions_and_positions() -> None:
    rows = [_oet_row("700", "renal failure", "100"), _oet_row("700", "kidney failure", "100"),
            _oet_row("701", "Lynch syndrome", "200|[EX.](<1> <2>)", children="300", synonyms="Lynch syndrome|HNPCC")]
    items = {i["record"]: i for i in lp.oet_items(rows, set_name="oet", split="test", labels={"700": "Renal failure (disorder)"})}
    assert items["700"]["name"] == "Renal failure (disorder)" and items["700"]["evidence"] == {"medmentions_mentions": 2}
    assert items["700"]["meta"]["positions"] == [["100", None]] and items["700"]["aliases"] == ["renal failure", "kidney failure"]
    assert items["701"]["gold_parents"] == ["200", "[EX.](<1> <2>)"] and ["child", "300"] in items["701"]["gold_relations"]
    assert items["701"]["meta"]["complex_parents"] == ["[EX.](<1> <2>)"] and items["701"]["name"] == "Lynch syndrome"


def test_owl_labels_and_oet_snapshot(tmp_path: Path) -> None:
    owl = ('Declaration(Class(<http://snomed.info/id/700>))\n'
           'AnnotationAssertion(rdfs:label <http://snomed.info/id/700> "Renal failure (disorder)"@en)\n'
           'AnnotationAssertion(rdfs:label :701 "Lynch \\"syndrome\\""@en)\n')
    with zipfile.ZipFile(tmp_path / "o.zip", "w") as archive:
        archive.writestr("x.owl", owl)
    with zipfile.ZipFile(tmp_path / "o.zip") as archive:
        assert lp.owl_labels(archive, "x.owl", {"700", "701"}) == {"700": "Renal failure (disorder)", "701": 'Lynch "syndrome"'}
    nodes, triples, candidates = lp.oet_snapshot(
        [{"idx": "100", "title": "parent (disorder)", "synonyms": "parent|p"}, {"idx": "300", "title": "child", "synonyms": ""}],
        [{"parent_idx": "100", "child_idx": "300"}, {"parent_idx": "100", "child_idx": "SCTID_NULL"},
         {"parent_idx": "[EX.](<1>)", "child_idx": "300", "parent": "something"}])
    assert {n["id"]: n["kind"] for n in nodes} == {"100": "concept", "300": "concept", "[EX.](<1>)": "complex"}
    assert ("300", "parent", "100") in triples and len(candidates) == 3 and candidates[1]["child"] is None


# -- TaxoExpan / TMN ---------------------------------------------------------------------------------------------------------------


class _Graph:      # pickled under this test module; read back as a stub
    def __init__(self) -> None:
        self.payload = [1, 2, 3]


class _Exploit:
    def __reduce__(self):
        return (print, ("EXECUTED",))


def test_stub_unpickler_executes_nothing(tmp_path: Path, capsys) -> None:
    data = {"name": "toy", "g_full": _Graph(), "exploit": _Exploit(),
            "vocab": ["entity||entity.n.01@@@0", "thing||thing.n.01@@@1", "wug||test.test.1@@@2", "dax||train.withdef.3@@@3"],
            "train_node_ids": [0, 1], "validation_node_ids": [3], "test_node_ids": [2]}
    path = tmp_path / "toy.pickle.bin"
    path.write_bytes(pickle.dumps(data))
    assert lp.taxoexpan_split(path) == {"train": ["entity.n.01", "thing.n.01"], "validation": ["train.withdef.3"],
                                        "test": ["test.test.1"]}
    assert "EXECUTED" not in capsys.readouterr().out


TOY_TERMS = {"entity.n.01": "entity", "animal.n.01": "animal", "dog.n.01": "dog", "cat.n.01": "cat",
             "puppy.n.01": "puppy", "wug": "wug", "plant.n.01": "plant"}
TOY_EDGES = [("entity.n.01", "animal.n.01"), ("entity.n.01", "plant.n.01"), ("animal.n.01", "dog.n.01"),
             ("animal.n.01", "cat.n.01"), ("dog.n.01", "puppy.n.01"), ("animal.n.01", "wug")]


def test_before_taxonomy_bridges_removed_nodes() -> None:
    kept = set(TOY_TERMS) - {"dog.n.01"}
    edges = lp.before_taxonomy(TOY_EDGES, kept)
    assert ("animal.n.01", "puppy.n.01") in edges and all("dog.n.01" not in e for e in edges)
    parents = {"puppy.n.01": ["dog.n.01"], "dog.n.01": ["animal.n.01"]}
    assert lp.kept_relatives("puppy.n.01", parents, kept) == ["animal.n.01"]


def test_taxo_items_completion_and_leakage() -> None:
    split = {"train": ["entity.n.01", "animal.n.01", "cat.n.01", "puppy.n.01", "plant.n.01"], "validation": ["wug"],
             "test": ["dog.n.01"]}
    items, before = lp.taxo_items("toy", TOY_TERMS, TOY_EDGES, split, completion=True, definitions={"dog.n.01": "a canine"})
    test = next(i for i in items if i["meta"]["split"] == "test")
    assert test["gold_parents"] == ["animal.n.01"] and ["child", "puppy.n.01"] in test["gold_relations"]
    assert test["meta"]["positions"] == [["animal.n.01", "puppy.n.01"]] and test["definition"] == "a canine"
    report = lp.leakage_report(items, TOY_TERMS, TOY_EDGES, set(split["train"]))
    assert report["test"]["with_training_sibling"] == 1 and report["test"]["mean_training_siblings"] == 1.0
    assert report["dev"]["wordnet_synsets"] == 0 and report["test"]["wordnet_synsets"] == 1
    expansion, _ = lp.taxo_items("toy", TOY_TERMS, TOY_EDGES, split, completion=False, definitions={})
    assert all(not any(r[0] == "child" for r in i["gold_relations"]) for i in expansion)


def test_tmn_split_is_seeded_and_excludes_roots() -> None:
    terms = {f"n{i}": f"n{i}" for i in range(50)}
    edges = [("n0", f"n{i}") for i in range(1, 50)]
    first = lp.tmn_split(terms, edges, seed=3, size=5)
    assert first == lp.tmn_split(terms, edges, seed=3, size=5)
    assert len(first["validation"]) == 5 and len(first["test"]) == 5 and "n0" in first["train"]
    assert not set(first["validation"]) & set(first["test"])


# -- author decisions of 2026-10-08: primary / secondary roles, committed ICD snapshot ------------------------------------------


def test_icd_build_writes_the_committed_snapshot(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    releases = {"fy2026": (ORDER_2026, [("I30-I5A", "Other forms of heart disease", ["I42", "I43"]),
                                        ("K65-K68", "Diseases of peritoneum", ["K65"])], 2026),
                "fy2027": (ORDER_2027, [("I30-I5A", "Other forms of heart disease", ["I42", "I43"]),
                                        ("K65-K6A", "Diseases of peritoneum and pelvis", ["K65", "K6A"])], 2027)}
    for name, (order, blocks, year) in releases.items():
        for (archive, member), text in ((icd.ORDER_FILES[name], "\r\n".join(_order_line(i, *r) for i, r in enumerate(order, 1))),
                                        (icd.TABULAR_FILES[name], _tabular(year, blocks))):
            with zipfile.ZipFile(raw / archive, "a") as handle:
                handle.writestr(member, text)
    manifest = icd.build(tmp_path / "out", tmp_path / "local", raw=raw, seed=1)
    assert manifest["snapshot"]["committed"] and manifest["snapshot"]["path"].endswith(icd.SNAPSHOT_DIR)
    nodes, edges = ld.load_snapshot(tmp_path / "out" / icd.SNAPSHOT_DIR)
    assert "I42.0" in nodes and "K6A" not in nodes and not (tmp_path / "local" / "snapshot-fy2026").exists()
    assert not icd.SNAPSHOT_DIR.startswith("snapshot-")          # the ignore rule `**/snapshot-*/` must not catch it
    files = manifest["files"]
    assert manifest["stats"]["new_codes"] == 7 and files["placement-test"]["items"] + files["placement-dev"]["items"] == 7


def _oet_zip(path: Path) -> Path:
    def lines(rows: list[dict]) -> str:
        return "\n".join(json.dumps(r) for r in rows) + "\n"

    with zipfile.ZipFile(path, "w") as archive:
        for part in lp.OET_PARTS.values():
            kind = part.split("-")[-1]
            base = f"{part}/mention-level-(concept-placement)"
            archive.writestr(f"{base}/valid-NIL.jsonl", lines([_oet_row("700", "renal failure", "100"),
                                                                _oet_row("701", "kidney stone", "200")]))
            archive.writestr(f"{base}/test-NIL.jsonl", lines([_oet_row("700", "renal failure", "100"),
                                                               _oet_row("702", "new syndrome", "100", children="200")]))
            archive.writestr(f"{part}/ontology/SNOMEDCT-US-20170301-{kind}-final.owl",
                             'AnnotationAssertion(rdfs:label <http://snomed.info/id/702> "New syndrome (disorder)"@en)\n')
            archive.writestr(f"{part}/ontology/SNOMEDCT-US-20140901-{kind}_syn_attr_hyp-all.jsonl",
                             lines([{"idx": "100", "title": "disorder (disorder)", "synonyms": "disorder", "text": ""},
                                    {"idx": "200", "title": "stone (disorder)", "synonyms": "stone", "text": ""}]))
            archive.writestr(f"{part}/ontology/SNOMEDCT-US-20140901-{kind}-edges-all.jsonl",
                             lines([{"parent_idx": "100", "child_idx": "200", "parent": "disorder", "child": "stone"},
                                    {"parent_idx": "200", "child_idx": "SCTID_NULL", "parent": "stone", "child": "NULL"}]))
    return path


def test_oet_primary_test_is_concept_disjoint(tmp_path: Path) -> None:
    manifest = lp.build_oet(tmp_path / "out", tmp_path / "local", zip_path=_oet_zip(tmp_path / "oet.zip"))
    base = tmp_path / "local" / "disease"
    assert [i["record"] for i in ld.read_jsonl(base / "placement-test.jsonl")] == ["702"]
    assert [i["record"] for i in ld.read_jsonl(base / "placement-test-mention-split.jsonl")] == ["700", "702"]
    assert [i["record"] for i in ld.read_jsonl(base / "placement-dev.jsonl")] == ["700", "701"]
    counts = manifest["sets"]["oet-snomed-disease"]["concepts"]
    assert counts == {"dev": 2, "test (primary, concept-disjoint)": 1, "test-mention-split (secondary)": 2}
    assert next(ld.read_jsonl(base / "placement-test.jsonl"))["name"] == "New syndrome (disorder)"
    assert not list((tmp_path / "out").glob("*.jsonl"))                     # only the manifest goes to the repository


def test_taxo_roles_primary_and_secondary(tmp_path: Path, monkeypatch) -> None:
    raw = tmp_path / "raw" / "toy"
    raw.mkdir(parents=True)
    terms = {"entity.n.01": "entity", "animal.n.01": "animal", "dog.n.01": "dog", "cat.n.01": "cat", "puppy.n.01": "puppy",
             "plant.n.01": "plant", "tree.n.01": "tree", "test.test.1": "wug", "train.withdef.2": "dax"}
    edges = [("entity.n.01", "animal.n.01"), ("entity.n.01", "plant.n.01"), ("animal.n.01", "dog.n.01"),
             ("animal.n.01", "cat.n.01"), ("dog.n.01", "puppy.n.01"), ("plant.n.01", "tree.n.01"),
             ("animal.n.01", "test.test.1"), ("plant.n.01", "train.withdef.2")]
    (raw / "wordnet_noun.terms").write_text("".join(f"{t}\t{s}||{t}\n" for t, s in terms.items()))
    (raw / "wordnet_noun.taxo").write_text("".join(f"{p}\t{c}\n" for p, c in edges))
    order = list(terms)
    (raw / "wordnet_noun.fasttext_mode4.pickle.bin").write_bytes(pickle.dumps(
        {"vocab": [f"{terms[t]}||{t}@@@{i}" for i, t in enumerate(order)], "train_node_ids": list(range(7)),
         "validation_node_ids": [8], "test_node_ids": [7]}))
    monkeypatch.setattr(lp, "TAXO_SETS", {"noun": "toy/wordnet_noun"})
    manifest = lp.build_taxo(tmp_path / "out", tmp_path / "local", raw=tmp_path / "raw", tmn_size=2)
    sets = manifest["sets"]
    assert sets["taxoexpan-semeval-noun"]["role"] == "primary" and sets["tmn-wordnet-noun"]["role"] == "secondary"
    assert "not comparable to published TMN numbers" in sets["tmn-wordnet-noun"]["label"]
    test = next(ld.read_jsonl(tmp_path / "local" / "tmn-wordnet-noun" / "placement-test.jsonl"))
    assert test["meta"]["role"] == "secondary" and "not comparable" in test["meta"]["label"]
    official = next(ld.read_jsonl(tmp_path / "local" / "taxoexpan-semeval-noun" / "placement-test.jsonl"))
    assert official["record"] == "test.test.1" and official["gold_parents"] == ["animal.n.01"]
    assert official["meta"]["role"] == "primary"
