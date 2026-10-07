"""Survey: parse Wiktionary neologisms (definitions, earliest quotation year), WordNet membership, general-text
occurrences in FineWeb-Edu (C3 stream order, documents [0, LIMIT)) and in the T7 PubMed 2026 text."""
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, "src")
from vsa_embed.ontologies.mesh_novel import count_general_mentions, mention_key

def main():
    RAW = Path(sys.argv[1])
    LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 300_000
    rows = [json.loads(line) for line in (RAW / "pages.jsonl").read_text().splitlines() if line.strip()]
    YEAR = re.compile(r"(?:\|\s*(?:year|date)\s*=\s*[^|}]*?\b((?:1[5-9]|20)\d\d)\b)|(?:'''((?:1[5-9]|20)\d\d)''')")
    POS = {"Noun", "Verb", "Adjective", "Adverb", "Interjection", "Phrase", "Proper noun", "Prepositional phrase"}


    def english(text: str) -> str:
        m = re.search(r"^==English==\s*$", text, re.M)
        if not m:
            return ""
        rest = text[m.end():]
        n = re.search(r"^==[^=].*==\s*$", rest, re.M)
        return rest[:n.start()] if n else rest


    def clean(definition: str) -> str:
        d = re.sub(r"\{\{(?:lb|lbl|label|q|qualifier|gloss|i)\|[^}]*\}\}", "", definition)
        d = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", d)
        d = re.sub(r"\{\{[^}]*\}\}", "", d)
        d = re.sub(r"<[^>]+>", "", d)
        return " ".join(d.replace("'''", "").replace("''", "").split()).strip(" .;:")


    parsed = []
    for row in rows:
        section = english(row.get("wikitext") or "")
        if not section:
            continue
        pos, defs, years = None, [], []
        for line in section.splitlines():
            h = re.match(r"^===+\s*([^=]+?)\s*===+\s*$", line)
            if h:
                pos = h.group(1) if h.group(1) in POS else pos
            elif line.startswith("# ") and pos:
                text = clean(line[2:])
                if text:
                    defs.append({"pos": pos, "definition": text})
            if line.startswith("#*") or "{{quote" in line or "{{defdate" in line:
                years += [int(a or b) for a, b in YEAR.findall(line)]
        parsed.append({"title": row["title"], "revid": row.get("revid"), "definitions": defs,
                       "first_year": min(years) if years else None, "quotation_years": sorted(set(years))})
    recent = [p for p in parsed if p["first_year"] and p["first_year"] >= 2023]
    dated = [p for p in parsed if p["first_year"]]
    print("pages", len(rows), "English with definitions", sum(1 for p in parsed if p["definitions"]), "dated", len(dated))
    by_decade = collections.Counter((p["first_year"] // 10) * 10 for p in dated)
    print("first quotation decade", dict(sorted(by_decade.items())))
    print("first quotation >= 2023:", len(recent), "; >= 2020:", sum(1 for p in dated if p["first_year"] >= 2020))
    from nltk.corpus import wordnet as wn
    in_wordnet = sum(1 for p in parsed if wn.synsets(p["title"].replace(" ", "_")))
    print("titles that are WordNet lemmas:", in_wordnet)
    multi = sum(1 for p in parsed if " " in p["title"] or "-" in p["title"])
    print("multiword/hyphenated titles:", multi)
    keys = sorted({mention_key(p["title"]) for p in parsed if p["definitions"]})
    shards = ["/home/bhux/data/vsa-llm/fineweb-edu/sample/10BT/002_00000.parquet", "/home/bhux/data/vsa-llm/fineweb-edu/sample/10BT/000_00000.parquet"]
    counts, documents = count_general_mentions(shards, keys, skip=0, limit=LIMIT, workers=4)
    recent_keys = {mention_key(p["title"]) for p in recent if p["definitions"]}
    print(f"FineWeb-Edu documents read {documents}: titles with >=1 occurrence {sum(1 for k in keys if counts[k])}, "
          f">=5: {sum(1 for k in keys if counts[k] >= 5)}; occurrences {sum(counts.values())}")
    print(f"  recent (>=2023) titles {len(recent_keys)}: with >=1 occurrence {sum(1 for k in recent_keys if counts[k])}, "
          f">=5 {sum(1 for k in recent_keys if counts[k] >= 5)}, occurrences {sum(counts[k] for k in recent_keys)}")
    for p in recent[:12]:
        print("  ", p["title"], p["first_year"], counts[mention_key(p["title"])], "|", (p["definitions"] or [{}])[0].get("definition", "")[:100])
    (RAW / "parsed.jsonl").write_text("".join(json.dumps(p) + "\n" for p in parsed))
    (RAW / "fineweb_counts.json").write_text(json.dumps({"documents": documents, "limit": LIMIT, "counts": {k: counts[k] for k in keys}}))


if __name__ == "__main__":
    main()
