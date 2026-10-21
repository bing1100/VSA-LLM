"""E14 RD-13 (`e14_retrieve`): the Wikipedia cache round-trip through a fake API, the disambiguation rule, the text
truncation, and the `retrieved` condition of `e14_reading`. No network."""

import json
from pathlib import Path

import pytest

from vsa_embed.experiments import e14_reading as rd
from vsa_embed.experiments import e14_retrieve as rt

PAGES = {
    1: {"title": "Lipoic acid", "revid": 11, "timestamp": "2026-09-16T11:36:30Z", "pageprops": {},
        "extract": "Lipoic acid is an organosulfur compound derived from caprylic acid. It is a cofactor.\n\n"
                   "== Function ==\nIt is essential for aerobic metabolism. Only the (R)-enantiomer exists in nature.\n\n"
                   "== See also ==\nDihydrolipoic acid\n"},
    2: {"title": "Lipoic (disambiguation)", "revid": 22, "timestamp": "2026-01-01T00:00:00Z", "pageprops": {"disambiguation": ""},
        "extract": "Lipoic may refer to: lipoic acid; a village."},
    3: {"title": "Lipoic Village", "revid": 33, "timestamp": "2026-01-01T00:00:00Z", "pageprops": {},
        "extract": "Lipoic Village is a village in the hills. It has a church and a school."},
    4: {"title": "Thioctic acid", "revid": 44, "timestamp": "2026-01-01T00:00:00Z", "pageprops": {},
        "extract": "Thioctic acid is another name for a chemical compound used as a drug."},
}
SEARCH = {"(R)-lipoic acid": [2, 3, 1], "thioctic acid": [4, 1]}


def fake_get(calls: list) -> callable:
    def get(params: dict) -> dict:
        calls.append(dict(params))
        if params.get("list") == "search":
            ids = SEARCH.get(params["srsearch"], [])
            return {"query": {"search": [{"pageid": i, "title": PAGES[i]["title"], "wordcount": 5} for i in ids]}}
        pid = int(params["pageids"])
        page = PAGES[pid]
        return {"query": {"pages": [{"pageid": pid, "title": page["title"], "extract": page["extract"], "pageprops": page["pageprops"],
                                     "revisions": [{"revid": page["revid"], "timestamp": page["timestamp"]}]}]}}
    return get


def test_cache_round_trip_is_offline_and_keyed_by_revision(tmp_path: Path) -> None:
    calls: list = []
    client = rt.WikiClient(tmp_path, rps=1000, get=fake_get(calls), log=lambda m: None)
    first = client.search("(R)-lipoic acid")
    page = client.page(1)
    assert [h["pageid"] for h in first] == [2, 3, 1] and page["revid"] == 11 and page["licence"] == rt.LICENCE
    n = len(calls)
    assert client.search("(R)-lipoic acid") == first and client.page(1) is page and len(calls) == n      # cached: no request
    again = rt.WikiClient(tmp_path, rps=1000, get=fake_get(calls), log=lambda m: None)                   # reloaded from disk
    assert again.page(1)["revid"] == 11 and again.search("(R)-lipoic acid") == first and len(calls) == n
    rows = [json.loads(l) for l in (tmp_path / "pages.jsonl").read_text().splitlines()]
    assert rows[0]["pageid"] == 1 and rows[0]["revid"] == 11 and rows[0]["fetched_at"].endswith("Z")
    assert (tmp_path / "searches.jsonl").exists()                   # requests.jsonl is written by the HTTP transport only


def test_disambiguation_rule_prefers_exact_titles_then_chemistry_cues(tmp_path: Path) -> None:
    calls: list = []
    client = rt.WikiClient(tmp_path, rps=1000, get=fake_get(calls), log=lambda m: None)
    record = {"name": "(R)-lipoic acid", "synonyms": ["thioctic acid", "C8H14O2S2", "LIPOIC ACID"]}
    result = rt.retrieve_concept(client, "(R)-lipoic acid", record, k=2)
    titles = [c["title"] for c in result["candidates"]]
    assert "Lipoic (disambiguation)" not in titles and result["dropped"] == 1            # disambiguation pages are dropped
    assert titles[0] == "Thioctic acid" and result["candidates"][0]["reason"] == "exact title"   # a synonym's exact title first
    assert titles[1] == "Lipoic acid" and result["candidates"][1]["reason"].startswith("cue")    # then the chemistry cue
    assert titles[-1] == "Lipoic Village" and result["candidates"][-1]["cue"] is None
    assert result["primary"] == 4 and result["plausible"] and result["pages"] == [4, 1]
    assert result["queries"] == ["(R)-lipoic acid", "thioctic acid"]                       # stopped once k plausible pages were in hand
    assert rt.queries_for("(R)-lipoic acid", record) == ["(R)-lipoic acid", "thioctic acid"]   # formulae and all-caps codes are not queries


def test_rank_candidates_records_reasons() -> None:
    cands = [{"pageid": 1, "title": "X", "extract": "", "query_rank": 0, "result_rank": 0},
             {"pageid": 2, "title": "Y", "extract": "Y is a town.", "query_rank": 0, "result_rank": 1},
             {"pageid": 3, "title": "Z", "extract": "Z is an ester.", "query_rank": 1, "result_rank": 0}]
    ranked = rt.rank_candidates(cands, ["q"])
    assert [c["pageid"] for c in ranked] == [3, 2] and ranked[0]["rank"] == 0 and ranked[0]["cue"] == "ester"
    assert ranked[1]["reason"] == "search rank only"


def test_truncation_keeps_whole_sentences_within_the_budget() -> None:
    paragraphs = rt.clean_paragraphs(PAGES[1]["extract"])
    assert paragraphs == ["Lipoic acid is an organosulfur compound derived from caprylic acid. It is a cofactor.",
                          "It is essential for aerobic metabolism. Only the (R)-enantiomer exists in nature."]   # headings and "See also" gone
    assert rt.intro_of(PAGES[1]["extract"]).endswith("It is a cofactor.")
    text, full = rt.retrieved_text("(R)-lipoic acid", PAGES[1]["extract"], words=14)
    assert text == "(R)-lipoic acid: Lipoic acid is an organosulfur compound derived from caprylic acid. It is a cofactor." and full == 26
    assert rt.truncate_words(paragraphs, 1000) == "\n".join(paragraphs)
    assert rt.truncate_words(paragraphs, 3) == "Lipoic acid is"                              # the first sentence is cut, never dropped
    assert rt.truncate_words(paragraphs, 20).count("\n") == 1                                 # a second paragraph starts on a new line
    assert rt.clean_paragraphs("Limonene () is a liquid.") == ["Limonene is a liquid."]


def test_retrieved_condition_uses_the_retrieved_text() -> None:
    frames, contexts, _ = rd.condition_inputs("retrieved", ["a", "b"], {}, {"a": "A", "b": "B"}, None, {"a": "def a"}, {}, seed=0,
                                              retrieved={"a": "wiki a"})
    assert frames == {"a": None, "b": None} and contexts == {"a": "wiki a"}
    with pytest.raises(KeyError):
        rd.condition_inputs("retrieved", ["a"], {}, {"a": "A"}, None, {}, {}, seed=0)
    assert "retrieved" in rd.RD13_CONDITIONS and all(a != b for a, b in rd.PRIMARY_R)
