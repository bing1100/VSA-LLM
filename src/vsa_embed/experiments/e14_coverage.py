"""E14 RD — the CPU side of reading from real text (pre-registration `experiments/e14-logic/preregistration.md` §7; plan
§8): what the text states, what the dictionary can find, whether reading snowballs, and the structured part of a frame.

- `audit` (RD-1a) — per relation: gold edges, the share *stated* in the definition (partner found by E11's concept
  finder) and the share each E11 reader recovered, from E11 `frames.jsonl` files (any track).
- `audit-teacher` (RD-1b) — Claude lists each statement a definition makes about its headword and labels it expressible
  by one of the track's relations, expressible but with a partner the dictionary lacks, or not expressible (with a
  category). Open-licence text only (`e11.teacher_allowed`); cached per prompt; stops at `--max-usd`.
- `dictionary` (RD-2) — T4: the concept finder rebuilt with ChEBI 255 synonyms (and hyphen/case variants) as aliases of
  the atoms; the share of gold partners found in the held-out definitions, before and after; and the shrinking curve
  (25/50/75% of the atoms' surfaces removed, seeded).
- `bootstrap` (RD-5) — a reading simulation over ChEBI definitions of T4 entries: start with a share of the atoms
  linkable, read every definition with the finder over the current dictionary and the type prior's relation (with a
  simulated relation error rate), add each read entry's own atom to the dictionary, repeat; per round the dictionary size,
  partner recall, edge precision and recall.
- `structure` (RD-6) — T4: `contains_element` from the ChEBI formula, `charge` from the ChEBI charge, `branch` from the
  root of the is_a ancestry; recall and precision against the gold frames of the held-out terms.

ChEBI is read from `~/data/vsa-llm/tracks/t4-chemistry/raw/chebi-255.obo.gz` (CC BY 4.0) and cached as
`~/data/vsa-llm/e14/chebi-255-records.json.gz`.

    python -m vsa_embed.experiments.e14_coverage audit --frames F.jsonl ... --output DIR
    python -m vsa_embed.experiments.e14_coverage audit-teacher --items DIR --output DIR [--limit 100] [--max-usd 7.5]
    python -m vsa_embed.experiments.e14_coverage dictionary --items DIR --output DIR
    python -m vsa_embed.experiments.e14_coverage bootstrap --output DIR [--entries 5000] [--rounds 5]
    python -m vsa_embed.experiments.e14_coverage structure --items DIR --output DIR
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .. import read_to_learn as rtl
from . import e11_read_to_learn as e11
from .e5_common import write_json

DATA = Path("~/data/vsa-llm").expanduser()
CHEBI_OBO = DATA / "tracks/t4-chemistry/raw/chebi-255.obo.gz"
CHEBI_CACHE = DATA / "e14/chebi-255-records.json.gz"
ROOTS = {"CHEBI:24431": "chemical entity", "CHEBI:50906": "role", "CHEBI:36342": "subatomic particle"}


# -- ChEBI ------------------------------------------------------------------------------------------------------------

def parse_obo(path: Path) -> dict[str, dict[str, Any]]:
    """ChEBI OBO → id → {name, definition, synonyms, formula, charge, is_a, stars}."""
    out: dict[str, dict[str, Any]] = {}
    current: dict[str, Any] | None = None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                current = {"synonyms": [], "is_a": []}
                continue
            if line.startswith("[") or current is None:
                if line.startswith("[") and line != "[Term]":
                    current = None
                continue
            if not line:
                if "id" in current and not current.get("obsolete"):
                    out[current.pop("id")] = current
                current = None
                continue
            key, _, value = line.partition(": ")
            if key == "id":
                current["id"] = value
            elif key == "name":
                current["name"] = value
            elif key == "def":
                m = re.match(r'"(.*)"', value)
                current["definition"] = re.sub(r"<[^>]+>", "", m.group(1)) if m else value
            elif key == "synonym":
                m = re.match(r'"(.*?)"', value)
                if m:
                    current["synonyms"].append(re.sub(r"<[^>]+>", "", m.group(1)))
            elif key == "is_a":
                current["is_a"].append(value.split(" ")[0])
            elif key == "is_obsolete" and value == "true":
                current["obsolete"] = True
            elif key == "subset" and value.endswith("_STAR"):
                current["stars"] = int(value.split("_")[0]) if value[0].isdigit() else None
            elif key == "property_value":
                m = re.match(r'(\S+) "(.*?)"', value)
                if m:
                    prop, val = m.group(1), m.group(2)
                    if prop.endswith(("/formula", ":generalized_empirical_formula", ":formula")):
                        current["formula"] = val
                    elif prop.endswith(("/charge", ":charge")):
                        try:
                            current["charge"] = int(val)
                        except ValueError:
                            pass
    if current and "id" in current:
        out[current.pop("id")] = current
    return out


def chebi_records(path: Path = CHEBI_OBO, cache: Path = CHEBI_CACHE) -> dict[str, dict[str, Any]]:
    if cache.exists():
        with gzip.open(cache, "rt") as handle:
            return json.load(handle)
    records = parse_obo(path)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache, "wt") as handle:
        json.dump(records, handle)
    return records


def atom_chebi_id(name: str) -> str | None:
    return name.split(":", 1)[1] if name.startswith("chebi:") else None


def variants(text: str) -> set[str]:
    """Hyphen, case and spacing variants of a surface ("anti-oxidant" ↔ "antioxidant" ↔ "anti oxidant")."""
    base = text.strip()
    out = {base, base.lower()}
    if "-" in base:
        out |= {base.replace("-", ""), base.replace("-", " ")}
    return {v for v in out if v}


# -- shared: headword spans, gold, found ----------------------------------------------------------------------------------

def headword_spans(text: str, surfaces: Iterable[str]) -> list[tuple[int, int]]:
    spans = []
    for s in surfaces:
        if not s:
            continue
        for m in re.finditer(re.escape(s), text, flags=re.IGNORECASE):
            spans.append((m.start(), m.end()))
    return spans


def text_variants(text: str) -> list[str]:
    """The text, with hyphens between letters removed ("anti-oxidant" → "antioxidant"), and with them as spaces."""
    joined = re.sub(r"(?<=[A-Za-z])-(?=[a-z])", "", text)
    spaced = re.sub(r"(?<=[A-Za-z])-(?=[a-z])", " ", text)
    return list(dict.fromkeys([text, joined, spaced]))


def found_atoms(lexicon: rtl.FillerLexicon, text: str, exclude: Sequence[tuple[int, int]], *, headwords: Sequence[str] = (),
                dehyphenate: bool = False) -> set[int]:
    if not dehyphenate:
        return {a for m in lexicon.find(text, exclude) for a in m.atoms}
    out: set[int] = set()
    for variant in text_variants(text):
        spans = headword_spans(variant, [v for h in headwords for v in text_variants(h)]) if headwords else exclude
        out |= {a for m in lexicon.find(variant, spans) for a in m.atoms}
    return out


def lexicon_pairs(lexicon: rtl.FillerLexicon) -> list[tuple[str, int]]:
    return [(alias, a) for alias, atoms in lexicon.surfaces.items() for a in atoms]


# -- RD-1a ------------------------------------------------------------------------------------------------------------------

def audit_frames(path: Path, readers: Sequence[str] = ("stated", "linker", "host", "typeprior", "teacher")) -> dict[str, Any]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    by: dict[tuple[str, str], dict[str, Any]] = defaultdict(dict)
    for r in rows:
        by[(r["style"], r["concept"])][r["reader"]] = r
    out: dict[str, Any] = {}
    for style in sorted({s for s, _ in by}):
        gold: Counter = Counter()
        hit = {r: Counter() for r in readers}
        concepts = 0
        for (s, c), m in by.items():
            if s != style or "oracle" not in m:
                continue
            concepts += 1
            g = {tuple(e) for e in (m["oracle"].get("frame") or [])}
            sets = {r: {tuple(e) for e in ((m.get(r) or {}).get("frame") or [])} for r in readers}
            for e in g:
                gold[e[0]] += 1
                for r in readers:
                    hit[r][e[0]] += e in sets[r]
        per = {rel: {"gold": n, **{r: hit[r][rel] / n for r in readers if any(hit[r].values()) or r == "stated"}}
               for rel, n in gold.most_common()}
        total = sum(gold.values())
        out[style] = {"concepts": concepts, "gold_edges": total, "relations": per,
                      "all": {r: sum(hit[r].values()) / total for r in readers if total and (any(hit[r].values()) or r == "stated")}}
    return out


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    doc = {"files": {}}
    lines = ["# E14 RD-1a — what the definitions state (per relation; share of gold edges)", ""]
    for f in args.frames:
        result = audit_frames(Path(f))
        doc["files"][str(f)] = result
        for style, r in result.items():
            readers = list(r["all"])
            lines += [f"## `{f}` — style `{style}` ({r['concepts']} terms, {r['gold_edges']} gold edges)", "",
                      "| relation | gold | " + " | ".join(readers) + " |", "|---|---:|" + "---:|" * len(readers)]
            for rel, v in r["relations"].items():
                lines.append(f"| {rel} | {v['gold']} | " + " | ".join(f"{v.get(x, 0):.3f}" for x in readers) + " |")
            lines.append("| **all** | " + f"{r['gold_edges']} | " + " | ".join(f"**{r['all'][x]:.3f}**" for x in readers) + " |")
            lines.append("")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "audit.json", doc)
    (out / "audit.md").write_text("\n".join(lines) + "\n")
    return doc


# -- RD-1b ------------------------------------------------------------------------------------------------------------------

CATEGORIES = ("expressible", "expressible_partner_missing", "structural", "quantitative", "provenance", "property_other", "other")


def audit_schema(relations: Sequence[str]) -> dict[str, Any]:
    statement = {"type": "object", "required": ["text", "relation", "partner", "category"], "properties": {
        "text": {"type": "string"}, "relation": {"type": "string", "enum": [*relations, "none"]},
        "partner": {"type": "string"}, "category": {"type": "string", "enum": list(CATEGORIES)}}}
    return {"type": "object", "required": ["definitions"], "properties": {"definitions": {"type": "array", "items": {
        "type": "object", "required": ["id", "statements"], "properties": {
            "id": {"type": "string"}, "statements": {"type": "array", "items": statement}}}}}}


def audit_prompt(batch: Sequence[dict[str, Any]], relations: Sequence[str]) -> str:
    rel = "\n".join(f"- {r}: {r.replace('_', ' ')}" for r in relations)
    body = "\n\n".join(f"Definition {d['id']} (headword \"{d['headword']}\"):\n{d['text']}" for d in batch)
    return (
        "You are auditing what dictionary definitions say about their headword, against a fixed set of relations.\n"
        "For each definition, split it into atomic statements about the headword (one fact each). For each statement give:\n"
        "- text: the statement in a few words;\n"
        "- relation: the one relation below that expresses it, or \"none\";\n"
        "- partner: the other concept, copied as written in the definition (\"\" if none);\n"
        "- category: \"expressible\" (a relation below expresses it and the partner is a concept), "
        "\"expressible_partner_missing\" (a relation fits but the partner is not a nameable concept), \"structural\" "
        "(molecular structure, substitution, configuration), \"quantitative\" (numbers, masses, counts), \"provenance\" "
        "(source organism, where found, who uses it), \"property_other\" (a property no relation below covers), or \"other\".\n\n"
        f"Relations:\n{rel}\n\n{body}\n\nAnswer only with the JSON object."
    )


def run_audit_teacher(args: argparse.Namespace) -> dict[str, Any]:
    from ..judging import claude_cli_runner
    read_set = e11.load_read_set(args.items)
    if not e11.teacher_allowed(read_set):
        raise SystemExit("open-licence text only (decision 58)")
    track = read_set.manifest["track"]
    ctx = e11.track_context(track)
    relations = list(ctx.typing.relation_names)
    style = args.style or (e11.PRIMARY_STYLE.get(track) if e11.PRIMARY_STYLE.get(track) in read_set.styles else read_set.styles[0])
    concepts = {c["concept"]: c for c in read_set.concepts}
    defs = [{"id": d["concept"], "headword": d.get("headword") or concepts[d["concept"]]["surface"], "text": d["text"]}
            for d in read_set.definitions if d["style"] == style]
    random.Random(args.seed).shuffle(defs)
    defs = defs[:args.limit]
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    schema = audit_schema(relations)
    spent, calls, results = 0.0, 0, {}
    for start in range(0, len(defs), args.batch):
        batch = defs[start:start + args.batch]
        prompt = audit_prompt(batch, relations)
        key = hashlib.sha256(json.dumps({"m": args.model, "p": prompt}).encode()).hexdigest()
        path = cache / f"{key}.json"
        if path.exists():
            record = json.loads(path.read_text())
        else:
            if spent >= args.max_usd:
                break
            envelope = claude_cli_runner(prompt, schema, args.model)
            calls += 1
            spent += float(envelope.get("total_cost_usd") or 0)
            record = {"verdict": envelope.get("structured_output"), "cost_usd": envelope.get("total_cost_usd")}
            if isinstance(record["verdict"], dict):
                path.write_text(json.dumps(record) + "\n")
        for d in (record.get("verdict") or {}).get("definitions", []):
            results[d.get("id")] = d.get("statements", [])
    # partner in the dictionary? (the finder's resolve)
    tally: Counter = Counter()
    rel_tally: Counter = Counter()
    linkable = 0
    expressible = 0
    for cid, statements in results.items():
        for s in statements:
            tally[s["category"]] += 1
            if s["category"] == "expressible":
                expressible += 1
                rel_tally[s["relation"]] += 1
                linkable += bool(s.get("partner") and ctx.fillers.resolve(s["partner"]))
    total = sum(tally.values())
    doc = {"track": track, "style": style, "definitions": len(results), "statements": total, "calls": calls, "spent_usd": spent,
           "categories": {k: tally[k] / total for k in CATEGORIES} if total else {},
           "expressible_relations": dict(rel_tally.most_common()),
           "expressible_partner_in_dictionary": linkable / expressible if expressible else None}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "audit-teacher.json", {**doc, "statements_by_definition": results})
    lines = [f"# E14 RD-1b — what {track} definitions say, and what the ontology can express ({len(results)} definitions, "
             f"{total} statements; ${spent:.2f})", "", "| category | share |", "|---|---:|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in doc["categories"].items()]
    lines += ["", f"Expressible statements whose partner the dictionary resolves: "
              f"{doc['expressible_partner_in_dictionary'] if doc['expressible_partner_in_dictionary'] is None else round(doc['expressible_partner_in_dictionary'], 3)}",
              "", "Expressible statements by relation: " + ", ".join(f"{k} {v}" for k, v in rel_tally.most_common())]
    (out / "audit-teacher.md").write_text("\n".join(lines) + "\n")
    return doc


# -- RD-2 ------------------------------------------------------------------------------------------------------------------

def heldout_tasks(read_set: e11.ReadSet, style: str) -> list[dict[str, Any]]:
    concepts = {c["concept"]: c for c in read_set.concepts}
    out = []
    for d in read_set.definitions:
        if d["style"] != style:
            continue
        c = concepts[d["concept"]]
        out.append({"concept": c["concept"], "text": d["text"], "frame": c["frame"],
                    "headwords": [c["surface"], c.get("item_surface") or "", d.get("headword") or ""]})
    return out


def partner_recall(lexicon: rtl.FillerLexicon, tasks: Sequence[dict[str, Any]], atom_id: dict[str, int],
                   relations: set[str] | None = None, *, dehyphenate: bool = False) -> dict[str, float]:
    gold_n = found_n = 0
    per_rel: Counter = Counter(); per_rel_found: Counter = Counter()
    for t in tasks:
        exclude = headword_spans(t["text"], t["headwords"])
        found = found_atoms(lexicon, t["text"], exclude, headwords=t["headwords"], dehyphenate=dehyphenate)
        for r, a in t["frame"]:
            if relations and r not in relations or a not in atom_id:
                continue
            gold_n += 1; per_rel[r] += 1
            if atom_id[a] in found:
                found_n += 1; per_rel_found[r] += 1
    return {"recall": found_n / gold_n if gold_n else 0.0, "gold": gold_n,
            "by_relation": {r: per_rel_found[r] / n for r, n in per_rel.most_common()}}


def run_dictionary(args: argparse.Namespace) -> dict[str, Any]:
    read_set = e11.load_read_set(args.items)
    track = read_set.manifest["track"]
    if track != "t4":
        raise SystemExit("RD-2 tier P is the T4 (ChEBI) dictionary")
    ctx = e11.track_context(track)
    style = args.style or "chebi"
    tasks = heldout_tasks(read_set, style)
    atom_names = list(ctx.ontology["atomic_names"])
    atom_id = {n: i for i, n in enumerate(atom_names)}
    base = lexicon_pairs(ctx.fillers)
    records = chebi_records()
    synonyms = []
    for i, name in enumerate(atom_names):
        cid = atom_chebi_id(name)
        rec = records.get(cid) if cid else None
        if rec:
            for s in [rec.get("name", "")] + rec.get("synonyms", []):
                synonyms += [(v, i) for v in variants(s)]
    base_variants = [(v, a) for s, a in base for v in variants(s)]
    levels = {"base": base, "base+variants": base + base_variants, "base+synonyms": base + synonyms,
              "base+synonyms+variants": base + base_variants + synonyms}
    doc: dict[str, Any] = {"track": track, "style": style, "terms": len(tasks), "levels": {}, "shrink": {}}
    for name, pairs in levels.items():
        lex = rtl.FillerLexicon(pairs, atom_names)
        doc["levels"][name] = {"surfaces": len(lex), **partner_recall(lex, tasks, atom_id)}
        if name == "base+synonyms+variants":
            doc["levels"][name + "+dehyphenated text"] = {"surfaces": len(lex), **partner_recall(lex, tasks, atom_id, dehyphenate=True)}
    for keep in (0.25, 0.5, 0.75):
        values = []
        for seed in (1, 2, 3):
            rng = random.Random(seed)
            atoms = sorted({a for _, a in base})
            kept = set(rng.sample(atoms, int(keep * len(atoms))))
            lex = rtl.FillerLexicon([(s, a) for s, a in base if a in kept], atom_names)
            values.append(partner_recall(lex, tasks, atom_id)["recall"])
        doc["shrink"][f"{keep:g}"] = {"mean": float(np.mean(values)), "seeds": values}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "dictionary.json", doc)
    lines = [f"# E14 RD-2 — dictionary breadth (T4 held-out, {len(tasks)} ChEBI definitions)", "",
             "| dictionary | surfaces | gold partners found |", "|---|---:|---:|"]
    lines += [f"| {k} | {v['surfaces']} | {v['recall']:.3f} |" for k, v in doc["levels"].items()]
    lines += ["", "Shrinking curve (base surfaces of a random share of the atoms kept; 3 seeds): "
              + "; ".join(f"{k}: {v['mean']:.3f}" for k, v in doc["shrink"].items()) + f"; 1: {doc['levels']['base']['recall']:.3f}", "",
              "By relation (base → synonyms+variants): " + ", ".join(
                  f"{r} {doc['levels']['base']['by_relation'][r]:.2f}→{doc['levels']['base+synonyms+variants']['by_relation'].get(r, 0):.2f}"
                  for r in doc["levels"]["base"]["by_relation"])]
    (out / "dictionary.md").write_text("\n".join(lines) + "\n")
    return doc


# -- RD-5 ------------------------------------------------------------------------------------------------------------------

def entry_tasks(ctx: e11.TrackContext, records: dict[str, dict[str, Any]], *, limit: int, seed: int) -> list[dict[str, Any]]:
    """T4 entries with one ChEBI concept, a definition and a frame (a seeded sample of `limit`)."""
    o = ctx.ontology
    names = o["concept_names"]
    atom_id = {n: i for i, n in enumerate(o["atomic_names"])}
    held = {int(e) for e in o.get("heldout_entries", ())}
    pool = []
    for e, concepts in enumerate(ctx.table.entry_concepts):
        if len(concepts) != 1 or e in held:
            continue
        cid = names[concepts[0]]
        rec = records.get(cid)
        frame = ctx.frame(e)
        if not rec or not rec.get("definition") or not frame:
            continue
        pool.append({"entry": e, "chebi": cid, "text": rec["definition"], "frame": frame, "own_atom": atom_id.get(f"chebi:{cid}"),
                     "headwords": [rec.get("name", "")]})
    random.Random(seed).shuffle(pool)
    return pool[:limit]


def simulate_bootstrap(ctx: e11.TrackContext, tasks: Sequence[dict[str, Any]], *, start: float, relation_accuracy: float | None,
                       rounds: int, seed: int) -> list[dict[str, Any]]:
    atom_names = list(ctx.ontology["atomic_names"])
    base = lexicon_pairs(ctx.fillers)
    textual = sorted({a for _, a in base})
    rng = random.Random(seed)
    known = set(rng.sample(textual, int(start * len(textual))))
    full = rtl.FillerLexicon(base, atom_names)
    ceiling_found = {t["entry"]: found_atoms(full, t["text"], headword_spans(t["text"], t["headwords"])) for t in tasks}
    stated_total = sum(sum(a in ceiling_found[t["entry"]] for _, a in t["frame"]) for t in tasks)
    gold_total = sum(len(t["frame"]) for t in tasks)
    history = []
    for r in range(1, rounds + 1):
        lex = rtl.FillerLexicon([(s, a) for s, a in base if a in known], atom_names)
        hits = edges = found_gold = 0
        read_entries = set()
        for t in tasks:
            atoms = found_atoms(lex, t["text"], headword_spans(t["text"], t["headwords"]))
            gold = set(t["frame"])
            frame = []
            for a in sorted(atoms):
                rel = ctx.typing.prior(a)
                if rel is None:
                    continue
                if relation_accuracy is not None and random.Random(f"{seed}:{t['entry']}:{a}").random() > relation_accuracy:
                    others = [x for x in ctx.typing.candidates(a) if x != rel]
                    rel = others[0] if others else rel
                frame.append((rel, a))
            edges += len(frame)
            hits += sum(e in gold for e in frame)
            found_gold += sum(a in atoms for _, a in gold)
            if frame:
                read_entries.add(t["entry"])
        history.append({"round": r, "dictionary_share": len(known) / len(textual), "edges": edges,
                        "precision": hits / edges if edges else None, "recall": hits / gold_total if gold_total else None,
                        "partner_recall_of_ceiling": found_gold / stated_total if stated_total else None,
                        "entries_read": len(read_entries)})
        own = {t["own_atom"] for t in tasks if t["entry"] in read_entries and t["own_atom"] is not None}
        known |= own & set(textual)
    return history


def run_bootstrap(args: argparse.Namespace) -> dict[str, Any]:
    ctx = e11.track_context("t4")
    records = chebi_records()
    tasks = entry_tasks(ctx, records, limit=args.entries, seed=args.seed)
    doc: dict[str, Any] = {"entries": len(tasks), "rounds": args.rounds, "runs": []}
    for start in (0.25, 0.5, 0.75):
        for accuracy in (None, 0.6, 0.95):
            for seed in (1, 2, 3):
                hist = simulate_bootstrap(ctx, tasks, start=start, relation_accuracy=accuracy, rounds=args.rounds, seed=seed)
                doc["runs"].append({"start": start, "relation_accuracy": accuracy, "seed": seed, "history": hist})
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "bootstrap.json", doc)
    lines = [f"# E14 RD-5 — does reading pick up steam? (simulation over {len(tasks)} T4 ChEBI definitions)", "",
             "Relation accuracy `type prior` = the type prior's own choice; 0.6 / 0.95 = a simulated reader. Mean of 3 seeds.", "",
             "| start | relation | round | dictionary | partners found (of the full-dictionary ceiling) | edge precision | edge recall (all gold) |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    grouped: dict[tuple, list] = defaultdict(list)
    for run in doc["runs"]:
        grouped[(run["start"], run["relation_accuracy"])].append(run["history"])
    for (start, acc), hs in grouped.items():
        for i in range(args.rounds):
            m = lambda k: np.mean([h[i][k] for h in hs if h[i][k] is not None])  # noqa: E731
            lines.append(f"| {start:g} | {'type prior' if acc is None else acc} | {i + 1} | {m('dictionary_share'):.3f} | "
                         f"{m('partner_recall_of_ceiling'):.3f} | {m('precision'):.3f} | {m('recall'):.3f} |")
    (out / "bootstrap.md").write_text("\n".join(lines) + "\n")
    return doc


# -- RD-6 ------------------------------------------------------------------------------------------------------------------

ELEMENT = re.compile(r"([A-Z][a-z]?)")


def formula_elements(formula: str) -> set[str]:
    return {e for e in ELEMENT.findall(formula or "") if e not in {"R", "X"}}


def chebi_root(cid: str, records: dict[str, dict[str, Any]], cache: dict[str, str | None]) -> str | None:
    if cid in cache:
        return cache[cid]
    seen, frontier, root = set(), [cid], None
    while frontier and root is None:
        nxt = []
        for c in frontier:
            if c in ROOTS:
                root = ROOTS[c]
                break
            for p in (records.get(c) or {}).get("is_a", []):
                if p not in seen:
                    seen.add(p); nxt.append(p)
        frontier = nxt
    cache[cid] = root
    return root


def structure_frame(cid: str, records: dict[str, dict[str, Any]], atom_id: dict[str, int], roots: dict[str, str | None]) -> list[tuple[str, str]]:
    rec = records.get(cid) or {}
    out = [("contains_element", f"element:{e}") for e in sorted(formula_elements(rec.get("formula", "")))
           if f"element:{e}" in atom_id]
    if "charge" in rec:
        c = rec["charge"]
        out.append(("charge", "charge:" + ("negative" if c < 0 else "positive" if c > 0 else "neutral")))
    root = chebi_root(cid, records, roots)
    if root and f"branch:{root}" in atom_id:
        out.append(("branch", f"branch:{root}"))
    return out


def run_structure(args: argparse.Namespace) -> dict[str, Any]:
    read_set = e11.load_read_set(args.items)
    if read_set.manifest["track"] != "t4":
        raise SystemExit("RD-6 tier P reads T4 structure from ChEBI")
    ctx = e11.track_context("t4")
    records = chebi_records()
    atom_id = {n: i for i, n in enumerate(ctx.ontology["atomic_names"])}
    structural = {"contains_element", "charge", "branch"}
    roots: dict[str, str | None] = {}
    gold_n: Counter = Counter(); hit: Counter = Counter(); pred_n: Counter = Counter(); all_gold = 0
    for c in read_set.concepts:
        cid = c.get("source_concept")
        gold = {tuple(e) for e in c["frame"]}
        all_gold += len(gold)
        pred = set(structure_frame(cid, records, atom_id, roots)) if cid else set()
        for e in gold:
            if e[0] in structural:
                gold_n[e[0]] += 1
                hit[e[0]] += e in pred
        for e in pred:
            pred_n[e[0]] += 1
    doc = {"terms": len(read_set.concepts), "gold_edges": all_gold,
           "structural_share_of_gold": sum(gold_n.values()) / all_gold if all_gold else None,
           "by_relation": {r: {"gold": gold_n[r], "recall": hit[r] / gold_n[r] if gold_n[r] else None,
                               "precision": hit[r] / pred_n[r] if pred_n[r] else None} for r in sorted(structural)},
           "recall_structural": sum(hit.values()) / sum(gold_n.values()) if gold_n else None}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "structure.json", doc)
    lines = [f"# E14 RD-6 — the structured part of the T4 frame, read from ChEBI fields ({doc['terms']} held-out terms)", "",
             f"Structural relations are {doc['structural_share_of_gold']:.3f} of the gold edges; recovered "
             f"{doc['recall_structural']:.3f} of them.", "", "| relation | gold | recall | precision |", "|---|---:|---:|---:|"]
    lines += [f"| {r} | {v['gold']} | {v['recall'] if v['recall'] is None else round(v['recall'], 3)} | "
              f"{v['precision'] if v['precision'] is None else round(v['precision'], 3)} |" for r, v in doc["by_relation"].items()]
    (out / "structure.md").write_text("\n".join(lines) + "\n")
    return doc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("audit"); a.add_argument("--frames", nargs="+", required=True); a.add_argument("--output", type=Path, required=True)
    t = sub.add_parser("audit-teacher")
    t.add_argument("--items", type=Path, required=True); t.add_argument("--output", type=Path, required=True)
    t.add_argument("--style", default=None); t.add_argument("--limit", type=int, default=100); t.add_argument("--batch", type=int, default=5)
    t.add_argument("--max-usd", type=float, default=7.5); t.add_argument("--model", default="claude-opus-5-5")
    t.add_argument("--cache", type=Path, default=Path("experiments/e14-logic/teacher-cache")); t.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("dictionary"); d.add_argument("--items", type=Path, required=True); d.add_argument("--output", type=Path, required=True)
    d.add_argument("--style", default=None)
    b = sub.add_parser("bootstrap"); b.add_argument("--output", type=Path, required=True)
    b.add_argument("--entries", type=int, default=5000); b.add_argument("--rounds", type=int, default=5); b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("structure"); s.add_argument("--items", type=Path, required=True); s.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    {"audit": run_audit, "audit-teacher": run_audit_teacher, "dictionary": run_dictionary, "bootstrap": run_bootstrap,
     "structure": run_structure}[args.command](args)


if __name__ == "__main__":
    main()
