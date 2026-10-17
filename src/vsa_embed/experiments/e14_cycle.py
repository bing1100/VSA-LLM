"""E14 RD-11 — multi-round active reading with relation discovery (pre-registration §11; the author's read → reflect →
re-read cycle). Tier P: the active rounds with Claude as the reader; the interleaved passive (gradient) rounds are tier F.

Per round r = 0, 1, 2:
1. **Read twice:** every definition is read twice (the relation list in two different orders). A fact is a partner
   copied from the text plus a listed relation, or `NEW` (a proposed relation name and description) when nothing fits.
2. **Consolidate:** facts proposed in both passes (same relation, same resolved partner) are kept.
3. **Reflect:** one call clusters the round's consolidated `NEW` proposals into at most 6 relation types; a type with at
   least `--support` proposals is accepted and listed (with its description) in the next round.

Measures (T4: gold frames; 4 relations hidden at the start): edges per definition, visible-relation precision and
recall, gold recall over all relations (accepted new relations mapped to the hidden gold relation of best edge Jaccard),
each hidden relation's best Jaccard, the reachable ceiling (gold edges whose partner the finder sees in the text), the
`NEW` share and unresolved partners. Open-licence text only; cached per prompt; stops at `--max-usd`.

    python -m vsa_embed.experiments.e14_cycle run --items DIR --output OUT [--limit 200] [--rounds 3] [--hide a,b,c] [--max-usd 7.5]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

from . import e11_read_to_learn as e11
from .e5_common import write_json

T4_HIDDEN = ("has_functional_parent", "is_conjugate_acid_of", "is_conjugate_base_of", "has_part")


def describe(ctx: e11.TrackContext, name: str) -> str:
    spec = getattr(ctx.lexicon, "templates", {}).get(name) if ctx.lexicon is not None else None
    example = spec.statement.replace("{x}", "X").replace("{y}", "Y") if spec is not None and getattr(spec, "statement", None) else None
    return name.replace("_", " ") + (f" (e.g. \"{example}\")" if example else "")


class Teacher:
    def __init__(self, cache: Path, model: str, max_usd: float) -> None:
        self.cache, self.model, self.max_usd = Path(cache), model, float(max_usd)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.spent, self.calls, self.cached = 0.0, 0, 0

    def ask(self, prompt: str, schema: dict[str, Any]) -> dict[str, Any] | None:
        from ..judging import claude_cli_runner
        key = hashlib.sha256(json.dumps({"m": self.model, "p": prompt, "s": schema}, sort_keys=True).encode()).hexdigest()
        path = self.cache / f"{key}.json"
        if path.exists():
            self.cached += 1
            return json.loads(path.read_text())
        if self.spent >= self.max_usd:
            return None
        for _ in range(2):
            try:
                env = claude_cli_runner(prompt, schema, self.model)
            except Exception:                                        # noqa: BLE001 — a failed call is retried once, then skipped
                continue
            self.calls += 1
            self.spent += float(env.get("total_cost_usd") or 0)
            out = env.get("structured_output")
            if isinstance(out, dict):
                path.write_text(json.dumps(out) + "\n")
                return out
        return None


def read_schema(relations: Sequence[str]) -> dict[str, Any]:
    fact = {"type": "object", "required": ["partner", "relation", "new_name", "new_description"], "properties": {
        "partner": {"type": "string"}, "relation": {"type": "string", "enum": [*relations, "NEW"]},
        "new_name": {"type": "string"}, "new_description": {"type": "string"}}}
    return {"type": "object", "required": ["definitions"], "properties": {"definitions": {"type": "array", "items": {
        "type": "object", "required": ["id", "facts"], "properties": {"id": {"type": "string"}, "facts": {"type": "array", "items": fact}}}}}}


def read_prompt(batch: Sequence[dict[str, Any]], relations: dict[str, str], order: Sequence[str], second: bool) -> str:
    rel = "\n".join(f"- {r}: {relations[r]}" for r in order)
    body = "\n\n".join(f"Definition {d['id']} (headword \"{d['headword']}\"):\n{d['text']}" for d in batch)
    again = ("This is a second, careful reading: check every fact against the text, and consider every listed relation "
             "before choosing NEW.\n") if second else ""
    return ("Read each definition and list the facts it states about its headword. For each fact give:\n"
            "- partner: the other concept, copied exactly as written in the text;\n"
            "- relation: the listed relation that expresses the fact; or NEW if none of them fits;\n"
            "- new_name / new_description: only for NEW — a short snake_case relation name and a one-line description "
            "(otherwise empty strings).\n"
            "State only facts the text supports; skip facts with no partner concept.\n" + again +
            f"\nRelations:\n{rel}\n\n{body}\n\nAnswer only with the JSON object.")


def reflect_schema() -> dict[str, Any]:
    return {"type": "object", "required": ["relations", "duplicates"], "properties": {
        "relations": {"type": "array", "items": {"type": "object", "required": ["name", "description", "members"], "properties": {
            "name": {"type": "string"}, "description": {"type": "string"}, "members": {"type": "array", "items": {"type": "integer"}}}}},
        "duplicates": {"type": "array", "items": {"type": "object", "required": ["id", "relation"], "properties": {
            "id": {"type": "integer"}, "relation": {"type": "string"}}}}}}


def reflect_prompt(proposals: Sequence[dict[str, Any]], relations: dict[str, str]) -> str:
    rel = "\n".join(f"- {r}: {d}" for r, d in relations.items())
    items = "\n".join(f"{i}. headword \"{p['headword']}\" — partner \"{p['partner']}\" — proposed {p['new_name']}: {p['new_description']}"
                      for i, p in enumerate(proposals))
    return ("Below are facts that readers could not express with the existing relations, each with a proposed new relation. "
            "Reflect on them: group proposals that express the same kind of relation into at most 6 new relation types "
            "(give each a snake_case name, a one-line description, and the numbers of its member proposals). List a proposal "
            "under duplicates if an existing relation already expresses it. Leave out one-off proposals.\n\n"
            f"Existing relations:\n{rel}\n\nProposals:\n{items}\n\nAnswer only with the JSON object.")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def resolve(ctx: e11.TrackContext, partner: str) -> int | None:
    atoms = ctx.fillers.resolve(partner) if partner else ()
    return int(atoms[0]) if atoms else None


def run(args: argparse.Namespace) -> dict[str, Any]:
    read_set = e11.load_read_set(args.items)
    if not e11.teacher_allowed(read_set):
        raise SystemExit("open-licence text only (decision 58)")
    track = read_set.manifest["track"]
    ctx = e11.track_context(track)
    style = args.style or (e11.PRIMARY_STYLE.get(track) if e11.PRIMARY_STYLE.get(track) in read_set.styles else read_set.styles[0])
    concepts = {c["concept"]: c for c in read_set.concepts}
    defs = [{"id": d["concept"], "headword": d.get("headword") or concepts[d["concept"]]["surface"], "text": d["text"]}
            for d in read_set.definitions if d["style"] == style and d["concept"] in concepts]
    random.Random(args.seed).shuffle(defs)
    defs = defs[: args.limit]
    names = list(ctx.ontology["relation_names"])
    hidden = [h for h in (args.hide.split(",") if args.hide else ()) if h in names]
    textual = [r for r in names if r not in {"contains_element", "charge", "branch"}]     # structure-only relations (RD-6) are not read
    relations = {r: describe(ctx, r) for r in textual if r not in hidden}
    teacher = Teacher(Path(args.cache), args.model, args.max_usd)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    gold: dict[str, set[tuple[str, int]]] = {}
    ceiling: dict[str, set[tuple[str, int]]] = {}
    for d in defs:
        frame = ctx.ids(concepts[d["id"]]["frame"]) or []
        g = {(names[r], int(a)) for r, a in frame if names[r] in textual}
        gold[d["id"]] = g
        found = {a for m in ctx.fillers.find(d["text"]) for a in m.atoms}
        ceiling[d["id"]] = {(r, a) for r, a in g if a in found}
    rounds: list[dict[str, Any]] = []
    all_facts = []
    for rnd in range(args.rounds):
        t0 = time.monotonic()
        passes = []
        for p in range(2):
            order = sorted(relations)
            random.Random(f"{args.seed}:{rnd}:{p}").shuffle(order)
            facts = defaultdict(list)
            for start in range(0, len(defs), args.batch):
                batch = defs[start:start + args.batch]
                ans = teacher.ask(read_prompt(batch, relations, order, second=p == 1), read_schema(list(relations)))
                for item in (ans or {}).get("definitions", []):
                    for f in item.get("facts", []):
                        facts[item.get("id")].append(f)
            passes.append(facts)
        consolidated: dict[str, list[dict[str, Any]]] = defaultdict(list)
        single: dict[str, list[dict[str, Any]]] = defaultdict(list)
        new_props: list[dict[str, Any]] = []
        n_new = n_all = n_unres = 0
        for d in defs:
            keyed = []
            for facts in passes:
                ks = {}
                for f in facts.get(d["id"], []):
                    atom = resolve(ctx, f.get("partner", ""))
                    rel = f.get("relation")
                    key = (rel if rel != "NEW" else "NEW", atom if atom is not None else norm(f.get("partner", "")))
                    ks[key] = {**f, "atom": atom}
                keyed.append(ks)
            for key, f in keyed[0].items():
                n_all += 1
                n_new += key[0] == "NEW"
                n_unres += f["atom"] is None
                single[d["id"]].append(f)
                if key in keyed[1]:
                    consolidated[d["id"]].append(f)
                    if key[0] == "NEW":
                        new_props.append({"concept": d["id"], "headword": d["headword"], "partner": f.get("partner", ""),
                                          "new_name": f.get("new_name", ""), "new_description": f.get("new_description", ""), "atom": f["atom"]})
        # metrics against gold
        def edges(facts_by: dict[str, list[dict[str, Any]]], rel_filter=None) -> set[tuple[str, str, int]]:
            return {(c, f["relation"], f["atom"]) for c, fs in facts_by.items() for f in fs
                    if f["atom"] is not None and f["relation"] != "NEW" and (rel_filter is None or f["relation"] in rel_filter)}
        G = {(c, r, a) for c, es in gold.items() for r, a in es}
        Gvis = {e for e in G if e[1] not in hidden}
        Gceil = {(c, r, a) for c, es in ceiling.items() for r, a in es}
        def pr(P, Gs):
            tp = len(P & Gs)
            return (tp / len(P) if P else None, tp / len(Gs) if Gs else None)
        vis_single = {e for e in edges(single) if e[1] in names}
        vis_cons = {e for e in edges(consolidated) if e[1] in names}
        p1, r1 = pr(vis_single, Gvis)
        pc, rc = pr(vis_cons, Gvis)
        # accepted new relations (from earlier rounds) mapped to hidden gold relations by edge Jaccard over (concept, atom)
        new_rel_edges: dict[str, set[tuple[str, int]]] = defaultdict(set)
        for c, fs in consolidated.items():
            for f in fs:
                if f["relation"] not in names and f["relation"] != "NEW" and f["atom"] is not None:
                    new_rel_edges[f["relation"]].add((c, f["atom"]))
        mapping = {}
        for h in hidden:
            gh = {(c, a) for c, r, a in G if r == h}
            best = max(((n, len(e & gh) / max(1, len(e | gh))) for n, e in new_rel_edges.items()), key=lambda x: x[1], default=(None, 0.0))
            mapping[h] = {"relation": best[0], "jaccard": best[1], "gold_edges": len(gh)}
        recovered = {e for e in vis_cons if e in G}
        for h, m in mapping.items():
            if m["relation"] and m["jaccard"] >= 0.2:
                recovered |= {(c, h, a) for c, a in new_rel_edges[m["relation"]] if (c, h, a) in G}
        metrics = {"round": rnd, "relations": len(relations), "facts_per_definition_single": n_all / max(1, len(defs)),
                   "facts_per_definition_consolidated": sum(len(v) for v in consolidated.values()) / max(1, len(defs)),
                   "new_share": n_new / max(1, n_all), "unresolved_share": n_unres / max(1, n_all),
                   "visible_precision_single": p1, "visible_recall_single": r1, "visible_precision_consolidated": pc,
                   "visible_recall_consolidated": rc, "gold_recall_all": len(recovered) / max(1, len(G)),
                   "ceiling_recall": len(Gceil) / max(1, len(G)), "hidden_mapping": mapping, "seconds": round(time.monotonic() - t0, 1)}
        # reflect: cluster NEW proposals into relation types for the next round
        accepted = []
        if rnd + 1 < args.rounds and new_props:
            sample = new_props[: args.max_proposals]
            ans = teacher.ask(reflect_prompt(sample, relations), reflect_schema())
            for rt in (ans or {}).get("relations", []):
                members = [m for m in rt.get("members", []) if isinstance(m, int) and 0 <= m < len(sample)]
                name = re.sub(r"[^a-z0-9_]", "", rt.get("name", "").lower().replace(" ", "_"))
                if len(members) >= args.support and name and name not in relations and name not in {"NEW"}:
                    relations[name] = rt.get("description", name)
                    accepted.append({"name": name, "description": rt.get("description", ""), "support": len(members)})
        metrics["accepted"] = accepted
        metrics["teacher_spent_usd"] = round(teacher.spent, 3)
        rounds.append(metrics)
        all_facts += [{"round": rnd, "concept": c, **{k: v for k, v in f.items()}} for c, fs in consolidated.items() for f in fs]
        print(json.dumps({k: v for k, v in metrics.items() if k != "hidden_mapping"}, default=str), flush=True)
        write_json(out / "rounds.json", {"track": track, "style": style, "definitions": len(defs), "hidden": hidden, "rounds": rounds,
                                          "teacher": {"spent_usd": teacher.spent, "calls": teacher.calls, "cached": teacher.cached}})
    with (out / "facts.jsonl").open("w") as handle:
        for f in all_facts:
            handle.write(json.dumps(f, default=str) + "\n")
    f2 = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    lines = [f"# E14 RD-11 — multi-round active reading with relation discovery ({track}, {len(defs)} definitions, style `{style}`)", "",
             f"Hidden at the start: {', '.join(hidden) or 'none'}. Teacher ${teacher.spent:.2f} ({teacher.calls} calls, {teacher.cached} cached).", "",
             "| round | relations | facts/def (single → consolidated) | NEW share | unresolved | visible P / R (single) | visible P / R (consolidated) | gold recall, all relations | ceiling | accepted next |",
             "|---:|---:|---|---:|---:|---|---|---:|---:|---|"]
    for m in rounds:
        lines.append(f"| {m['round']} | {m['relations']} | {m['facts_per_definition_single']:.2f} → {m['facts_per_definition_consolidated']:.2f} | "
                     f"{m['new_share']:.3f} | {m['unresolved_share']:.3f} | {f2(m['visible_precision_single'])} / {f2(m['visible_recall_single'])} | "
                     f"{f2(m['visible_precision_consolidated'])} / {f2(m['visible_recall_consolidated'])} | {m['gold_recall_all']:.3f} | "
                     f"{m['ceiling_recall']:.3f} | {', '.join(a['name'] + ' (' + str(a['support']) + ')' for a in m['accepted']) or '—'} |")
    if hidden:
        lines += ["", "Hidden gold relations — best-matching discovered relation (edge Jaccard) by round:", ""]
        for h in hidden:
            lines.append(f"- {h}: " + "; ".join(f"r{m['round']}: {m['hidden_mapping'][h]['relation'] or '—'} ({m['hidden_mapping'][h]['jaccard']:.2f})"
                                               for m in rounds))
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return {"rounds": rounds}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--items", type=Path, required=True); r.add_argument("--output", type=Path, required=True)
    r.add_argument("--style", default=None); r.add_argument("--limit", type=int, default=200); r.add_argument("--rounds", type=int, default=3)
    r.add_argument("--hide", default=""); r.add_argument("--batch", type=int, default=8); r.add_argument("--support", type=int, default=8)
    r.add_argument("--max-proposals", type=int, default=250); r.add_argument("--max-usd", type=float, default=7.5)
    r.add_argument("--model", default="claude-opus-5-5"); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--cache", type=Path, default=Path("experiments/e14-logic/teacher-cache/rd11"))
    run(parser.parse_args(argv))


if __name__ == "__main__":
    main()
