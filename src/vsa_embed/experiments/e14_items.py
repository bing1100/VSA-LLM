"""E14 item sets for the model-reading experiments (plan §§4–6; pre-registration §§4–5): every item carries its question,
options, gold answer and, per condition, the text placed before the question (precomputed from the exact reference and the
VSA store), so scoring is one pass per host.

Sets (`build --sets`, default all; seed 1 by default):
- `taxonomy` — LG-B1/B2/B3/B7 and LG-C3: 6 depths × 200 questions (true/false), plus a cloze variant on true items.
- `prop-rp`, `prop-lp` — LG-B4: SimpleLogic-style propositional theories, 600 per sampler.
- `composition` — LG-B5: CLUTRR-style chains, k = 2–6, 100 each.
- `rulebase` — LG-B6: taxonomy theories with about 10 and about 50 rules, depths 1/3/5, 100 each.
- `fewshot` — LG-C4: 4 rule kinds × k ∈ {1, 3, 5, 10} × 100.
- `edits` — LG-C5: 300 class edits × 3 hops plus 300 locality questions.
- `trees` — LG-D: 600 trees (depth 2/3/4/6), 300 lists (length 3/6/12), 200 tree edits.

Item files are written gzipped to `--output` (default `~/data/vsa-llm/e14/items/`); a manifest with counts, conditions and
SHA-256 goes to `experiments/e14-logic/items/manifest-<seed>.json`.

    python -m vsa_embed.experiments.e14_items build [--seed 1] [--sets taxonomy,trees] [--scale 1.0]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any, Callable, Sequence

from ..logic import reference as R
from ..logic import store as S
from ..logic import world as W

OUT = Path("~/data/vsa-llm/e14/items").expanduser()
MANIFEST_DIR = Path("experiments/e14-logic/items")
TF = [" True", " False"]


def _lines(rows: Sequence[str]) -> str:
    return "\n".join(f"- {r}" for r in rows)


# -- taxonomy (B1–B3, B6, B7, C3, C5) ------------------------------------------------------------------------------------

def taxonomy_text(item: dict[str, Any], rules: Sequence[dict[str, Any]] | None = None) -> str:
    facts = [W.taxonomy_fact_text(f) for f in item["facts"]]
    rules = item["rules"] if rules is None else rules
    return "Facts:\n" + _lines(facts) + "\nRules:\n" + _lines([W.taxonomy_rule_text(r) for r in rules])


def tax_question(item: dict[str, Any]) -> str:
    return f"Question: Is it true that {item['entity']} is a {item['query']}?\nAnswer (True or False):"


def retrieve_rules(item: dict[str, Any], k: int = 5) -> list[dict[str, Any]]:
    """Keyword retrieval baseline: rules ranked by word overlap with the question and the entity's facts."""
    words = set(re.findall(r"[a-z]+", (tax_question(item) + " " + " ".join(f["cls"] for f in item["facts"])).lower()))
    scored = sorted(item["rules"], key=lambda r: -len(words & {r["a"].lower(), r["b"].lower()}))
    return scored[:k]


def taxonomy_conditions(item: dict[str, Any], demo: dict[str, Any] | None, *, width: int, seed: int) -> dict[str, str]:
    ent = item["entity"]
    vsa = S.TaxonomyStore(item, width=width, seed=seed).execute()
    sym = S.taxonomy_symbolic(item)
    conds = {
        "none": "",
        "facts": "Facts:\n" + _lines([W.taxonomy_fact_text(f) for f in item["facts"]]),
        "text": taxonomy_text(item),
        "exec-vsa": vsa.text(ent),
        "exec-vsa-conclusion": vsa.text(ent, trace=False),
        "exec-vsa-noconf": vsa.text(ent, confidence=False),
        "exec-sym": sym.text(ent),
        "exec-shuffled": S.TaxonomyStore(item, width=width, seed=seed, shuffle=True).execute().text(ent),
        "exec-nobind": S.TaxonomyStore(item, width=width, seed=seed, binding=False).execute().text(ent),
        "exec-vsa-stressed": S.TaxonomyStore(item, width=width, seed=seed, extra_load=10).execute().text(ent),
        "text-retrieved": "Facts:\n" + _lines([W.taxonomy_fact_text(f) for f in item["facts"]]) + "\nRules:\n"
                          + _lines([W.taxonomy_rule_text(r) for r in retrieve_rules(item)]),
    }
    for p in (0.1, 0.3):
        noisy = S.TaxonomyStore(item, width=width, seed=seed, noise=p)
        conds[f"exec-vsa@{p:g}"] = noisy.execute().text(ent)
        conds[f"exec-sym@{p:g}"] = S.taxonomy_symbolic(noisy.symbolic_rules()).text(ent)
    if demo is not None:
        answer = "True" if demo["label"] else "False"
        conds["text+demo"] = (f"Example.\n{taxonomy_text(demo)}\n{tax_question(demo)} {answer}\n\nNow the real question.\n"
                              + taxonomy_text(item))
    return conds


def taxonomy_set(seed: int, *, scale: float, width: int, namer: W.Namer) -> list[dict[str, Any]]:
    world = W.taxonomy(seed, questions_per_depth=max(2, int(200 * scale)), namer=namer)["items"]
    demo_pool = [i for i in world if i["depth"] == 2]
    rng = random.Random(f"tax-demo:{seed}")
    out = []
    for item in world:
        demo = rng.choice([d for d in demo_pool if d["id"] != item["id"]])
        conds = taxonomy_conditions(item, demo, width=width, seed=seed)
        gold = 0 if item["label"] else 1
        base = {k: item[k] for k in ("id", "world", "depth", "label", "counterfactual", "entity", "query")}
        out.append({**base, "set": "taxonomy", "format": "tf", "question": tax_question(item), "options": TF, "gold": gold,
                    "contexts": conds})
        if item["label"]:                                              # B7 cloze on true items: the class among 4
            distract = [r["b"] for r in item["rules"] if r["kind"] == "sub" and r["b"] not in item["chain"]][:3]
            if len(distract) == 3:
                options = [" " + item["query"]] + [" " + d for d in distract]
                rng.shuffle(options)
                out.append({**base, "id": item["id"] + "-cloze", "set": "taxonomy", "format": "cloze",
                            "question": f"Question: {item['entity']} is a", "options": options,
                            "gold": options.index(" " + item["query"]),
                            "contexts": {k: conds[k] for k in ("none", "text", "exec-vsa", "exec-sym")}})
    return out


def rulebase_set(seed: int, *, scale: float, width: int, namer: W.Namer) -> list[dict[str, Any]]:
    out = []
    for size, chains in ((10, 2), (50, 22)):
        world = W.taxonomy(seed + size, questions_per_depth=max(2, int(100 * scale)), distractor_chains=chains, namer=namer)["items"]
        for item in world:
            if item["depth"] not in (1, 3, 5):
                continue
            vsa = S.TaxonomyStore(item, width=width, seed=seed).execute()
            conds = {"text": taxonomy_text(item), "exec-vsa": vsa.text(item["entity"]),
                     "text-retrieved": "Facts:\n" + _lines([W.taxonomy_fact_text(f) for f in item["facts"]]) + "\nRules:\n"
                                       + _lines([W.taxonomy_rule_text(r) for r in retrieve_rules(item)])}
            out.append({"id": f"{item['id']}-r{size}", "set": "rulebase", "world": "taxonomy", "rules": size, "depth": item["depth"],
                        "label": item["label"], "format": "tf", "question": tax_question(item), "options": TF,
                        "gold": 0 if item["label"] else 1, "contexts": conds})
    return out


def edits_set(seed: int, *, scale: float, width: int, namer: W.Namer) -> list[dict[str, Any]]:
    """The entity's class is edited to the bottom of a fresh chain. Propagation: "which is Brindle: a D_h or a C_h?" (hop h
    of the new chain against hop h of the old one; after the edit only D_h follows). Locality: a second entity of the old
    class, same two options, answer C_h. A two-way choice, so a yes-biased model gains nothing; the stale store (built before
    the edit) points to the wrong option on propagation."""
    rng = random.Random(f"edits:{seed}")
    base = W.taxonomy(seed + 7, questions_per_depth=max(2, int(50 * scale)), namer=namer)["items"]
    out = []
    for item in base:
        new_chain = [namer.word() for _ in range(4)]
        added = [{"kind": "sub", "a": new_chain[i], "b": new_chain[i + 1]} for i in range(3)]
        edited = {**item, "facts": [{"kind": "is", "entity": item["entity"], "cls": new_chain[0]}], "rules": item["rules"] + added}
        stale = {**item, "rules": item["rules"] + added}
        other = namer.name()
        for hop in (1, 2, 3):
            new_c, old_c = new_chain[hop], item["chain"][hop]
            options = [" " + new_c, " " + old_c]
            rng.shuffle(options)
            q_new, q_stale = {**edited, "query": new_c}, {**stale, "query": new_c}
            conds = {"none": "", "text-edited": taxonomy_text(edited),
                     "exec-vsa": S.TaxonomyStore(q_new, width=width, seed=seed).execute().text(item["entity"]),
                     "exec-stale": S.TaxonomyStore({**q_stale, "query": old_c}, width=width, seed=seed).execute().text(item["entity"])}
            out.append({"id": f"{item['id']}-edit-h{hop}", "set": "edits", "world": "taxonomy", "depth": hop, "kind": "propagation",
                        "format": "mc", "question": f"Question: Which is {item['entity']}: a{options[0]} or a{options[1]}?\nAnswer: a",
                        "options": options, "gold": options.index(" " + new_c), "contexts": conds})
        hop = rng.randint(1, 3)
        loc = {**item, "entity": other, "facts": [{"kind": "is", "entity": other, "cls": item["chain"][0]}], "rules": edited["rules"],
               "query": item["chain"][hop]}
        options = [" " + item["chain"][hop], " " + new_chain[hop]]
        rng.shuffle(options)
        text = S.TaxonomyStore(loc, width=width, seed=seed).execute().text(other)
        conds = {"none": "", "text-edited": taxonomy_text(loc), "exec-vsa": text, "exec-stale": text}
        out.append({"id": f"{item['id']}-local", "set": "edits", "world": "taxonomy", "depth": hop, "kind": "locality", "format": "mc",
                    "question": f"Question: Which is {other}: a{options[0]} or a{options[1]}?\nAnswer: a", "options": options,
                    "gold": options.index(" " + item["chain"][hop]), "contexts": conds})
    return out


# -- propositional (B4) ----------------------------------------------------------------------------------------------------

def prop_set(seed: int, sampler: str, *, scale: float, width: int, namer: W.Namer) -> list[dict[str, Any]]:
    world = W.propositional(seed, sampler=sampler, items_per_depth=max(2, int(120 * scale)), namer=namer)["items"]
    out = []
    for item in world:
        text = "Facts: it is " + ", ".join(item["facts"]) + ".\nRules:\n" + _lines([W.prop_rule_text(b, h) for b, h in item["rules"]])
        conds = {"none": "", "text": text, "exec-vsa": S.PropositionStore(item, width=width, seed=seed).execute().text("it"),
                 "exec-sym": S.proposition_symbolic(item).text("it"),
                 "exec-shuffled": S.PropositionStore(item, width=width, seed=seed, shuffle=True).execute().text("it")}
        out.append({"id": item["id"], "set": f"prop-{sampler}", "world": "propositional", "sampler": sampler, "depth": item["depth"],
                    "label": item["label"], "rule_count": item["rule_count"], "format": "tf",
                    "question": f"Question: Is it true that it is {item['query']}?\nAnswer (True or False):", "options": TF,
                    "gold": 0 if item["label"] else 1, "contexts": conds})
    return out


# -- composition (B5) ------------------------------------------------------------------------------------------------------

def composition_set(seed: int, *, scale: float, width: int, namer: W.Namer) -> list[dict[str, Any]]:
    world = W.composition(seed, items_per_k=max(2, int(100 * scale)), namer=namer)
    rules = [f"If X is the {a} of Y and Z is the {b} of X, then Z is the {c} of Y." for a, b, c in world["table"]]
    rng = random.Random(f"comp-demo:{seed}")
    out = []
    for item in world["items"]:
        facts = [f"{item['people'][i + 1]} is the {r} of {item['people'][i]}." for i, r in enumerate(item["relations"])]
        text = "Facts:\n" + _lines(facts) + "\nRules:\n" + _lines(rules)
        p0, pk = item["people"][0], item["people"][-1]
        conds = {"none": "", "facts": "Facts:\n" + _lines(facts), "text": text,
                 "exec-vsa": S.CompositionStore(world, item, width=width, seed=seed).execute().text(p0),
                 "exec-sym": S.composition_symbolic(world, item).text(p0),
                 "exec-nobind": S.CompositionStore(world, item, width=width, seed=seed, binding=False).execute().text(p0)}
        options = [" " + o for o in item["options"]]
        out.append({"id": item["id"], "set": "composition", "world": "composition", "depth": item["depth"],
                    "uses_heldout": item["uses_heldout"], "format": "mc", "question": f"Question: {pk} is the ___ of {p0}.\nAnswer:",
                    "options": options, "gold": options.index(" " + item["answer"]), "contexts": conds})
    rng.shuffle(out)
    return out


# -- few-shot rules (C4) ---------------------------------------------------------------------------------------------------

def fewshot_set(seed: int, *, scale: float, width: int, namer: W.Namer) -> list[dict[str, Any]]:
    world = W.fewshot(seed, queries=max(2, int(100 * scale)), namer=namer)["items"]
    out = []
    say = lambda t: f"{t[0]} {t[1]} {t[2]}"  # noqa: E731
    for item in world:
        examples = [", and ".join(say(p) for p in prem) + f", so {say(conc)}." for prem, conc in item["examples"]]
        premise = ", and ".join(say(p) for p in item["premises"])
        conds = {"none": f"{premise}.", "icl": "Examples:\n" + _lines(examples) + f"\nNow: {premise}.",
                 "exec-vsa": f"{premise}.\n" + S.MappingRule(item, width=1024, seed=seed).execute().text("pattern")}
        options = [" " + o for o in item["options"]]
        subject, rel = item["question"]
        out.append({"id": item["id"], "set": "fewshot", "world": "fewshot", "kind": item["kind"], "k": item["k"], "depth": item["k"],
                    "format": "mc", "question": f"Question: {subject} {rel} ___?\nAnswer:", "options": options,
                    "gold": options.index(" " + item["answer"]), "contexts": conds})
    return out


# -- trees and lists (LG-D) ------------------------------------------------------------------------------------------------

def record_text(item: dict[str, Any]) -> str:
    return f"{item['entity']}: " + "; ".join(f"{k} is {v}" for k, v in item["record"].items()) + "."


def list_text(item: dict[str, Any]) -> list[str]:
    rows = [f"{'If' if i == 0 else 'Else if'} {W.test_text(r['if'])}: {r['then']}." for i, r in enumerate(item["rules"])]
    return rows + [f"Otherwise: {item['default']}."]


def trees_set(seed: int, *, scale: float, width: int, namer: W.Namer) -> list[dict[str, Any]]:
    world = W.trees(seed, inputs_per_depth=max(2, int(150 * scale)), inputs_per_list=max(2, int(100 * scale)),
                    edits=max(2, int(200 * scale)), namer=namer)["items"]
    out = []
    for item in world:
        ent = item["entity"]
        policy = (["Policy:"] + W.tree_text(item["tree"])) if item["kind"] != "list" else (["Policy:"] + list_text(item))
        conds = {"none": record_text(item), "tree-text": record_text(item) + "\n" + "\n".join(policy)}
        if item["kind"] != "list":
            conds["tree-bullets"] = record_text(item) + "\nPolicy:\n" + "\n".join("- " + l.strip() for l in W.tree_text(item["tree"]))
        vsa = S.TreeStore(item, width=width, seed=seed).execute()
        conds["path-vsa"] = record_text(item) + "\n" + vsa.text(ent)
        conds["leaf-vsa"] = record_text(item) + "\n" + vsa.text(ent, trace=False)
        conds["path-sym"] = record_text(item) + "\n" + S.tree_symbolic(item).text(ent)
        if item["kind"] == "tree":
            conds["path-shuffled"] = record_text(item) + "\n" + S.TreeStore(item, width=width, seed=seed, shuffle=True).execute().text(ent)
        if item["kind"] == "edit":                                     # the path through the tree before the edit
            conds["path-stale"] = record_text(item) + "\n" + S.tree_symbolic({**item, "tree": item["base_tree"]}).text(ent)
            conds["tree-text"] = record_text(item) + "\n" + "\n".join(["Policy:"] + W.tree_text(item["tree"]))
        options = [" " + o for o in item["options"]]
        out.append({"id": item["id"], "set": "trees", "world": "trees", "kind": item["kind"], "depth": item["depth"],
                    "numeric": item.get("numeric", False), "format": "mc",
                    "question": f"Question: Under the policy, what happens to {ent}?\nAnswer:", "options": options,
                    "gold": options.index(" " + item["answer"]), "contexts": conds})
    return out


BUILDERS: dict[str, Callable[..., list[dict[str, Any]]]] = {
    "taxonomy": taxonomy_set, "rulebase": rulebase_set, "edits": edits_set,
    "prop-rp": lambda seed, **kw: prop_set(seed, "rp", **kw), "prop-lp": lambda seed, **kw: prop_set(seed, "lp", **kw),
    "composition": composition_set, "fewshot": fewshot_set, "trees": trees_set}


def write_set(rows: Sequence[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows).encode()
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as handle:
        handle.write(data)
    return hashlib.sha256(data).hexdigest()


def read_set(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


# -- LG-B verdict format (pre-registration §14): the same recall, the conclusion written as a verdict on the question's claim

VERDICT_SOURCES = ("exec-vsa", "exec-sym", "exec-vsa-conclusion")
_CONCLUSION = re.compile(r"^Conclusion: (?P<ent>.+?) is (?P<neg>not )?a (?P<cls>.+?)\.(?P<conf> \(\d\.\d\d\))?$")
_NO_CONCLUSION = re.compile(r"^Conclusion: no conclusion about whether (?P<ent>.+?) is a (?P<cls>.+?)\.(?P<conf> \(\d\.\d\d\))?$")


def verdict_text(context: str) -> str:
    """`context` with its conclusion line rewritten as a verdict: "Conclusion: X is not a Y. (0.74)" → "Conclusion: the claim
    "X is a Y" is False. (0.74)"; "X is a Y" → "… is True."; no conclusion → "… is unknown.". Every other line is kept."""
    out = []
    for line in context.split("\n"):
        m = _NO_CONCLUSION.match(line)
        if m:
            out.append(f'Conclusion: the claim "{m["ent"]} is a {m["cls"]}" is unknown.{m["conf"] or ""}')
            continue
        m = _CONCLUSION.match(line)
        if m:
            out.append(f'Conclusion: the claim "{m["ent"]} is a {m["cls"]}" is {"False" if m["neg"] else "True"}.{m["conf"] or ""}')
            continue
        out.append(line)
    return "\n".join(out)


def verdict_set(rows: Sequence[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    """True/false items of a set with `<source>-verdict` contexts added for each source in `VERDICT_SOURCES` (and the
    baselines `none`, `text` and the sources kept), set name `name`."""
    out = []
    for r in rows:
        if r.get("format") != "tf":
            continue
        keep = {c: r["contexts"][c] for c in ("none", "text", *VERDICT_SOURCES) if c in r["contexts"]}
        for c in VERDICT_SOURCES:
            if c in r["contexts"]:
                keep[f"{c}-verdict"] = verdict_text(r["contexts"][c])
        out.append({**r, "set": name, "contexts": keep})
    return out


def run_verdict(args: argparse.Namespace) -> dict[str, Any]:
    manifest = {"seed": args.seed, "from": str(args.output), "sets": {}}
    for base in ("taxonomy", "rulebase"):
        rows = read_set(Path(args.output) / f"{base}-s{args.seed}.jsonl.gz")
        name = f"{base}-verdict"
        new = verdict_set(rows, name)
        changed = sum(r["contexts"].get("exec-vsa-verdict") != r["contexts"].get("exec-vsa") for r in new if "exec-vsa" in r["contexts"])
        sha = write_set(new, Path(args.output) / f"{name}-s{args.seed}.jsonl.gz")
        manifest["sets"][name] = {"items": len(new), "sha256": sha, "conclusions_rewritten": changed,
                                  "conditions": sorted({c for r in new for c in r["contexts"]})}
        print(f"{name}: {len(new)} items, {changed} conclusions rewritten")
    Path(args.manifest_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.manifest_dir) / f"manifest-verdict-s{args.seed}.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    sets = [s.strip() for s in args.sets.split(",")] if args.sets else list(BUILDERS)
    namer = W.Namer(args.seed)
    manifest = {"seed": args.seed, "scale": args.scale, "width": args.width, "sets": {}}
    for name in sets:
        rows = BUILDERS[name](args.seed, scale=args.scale, width=args.width, namer=namer)
        path = Path(args.output) / f"{name}-s{args.seed}.jsonl.gz"
        digest = write_set(rows, path)
        conds = sorted({c for r in rows for c in r["contexts"]})
        manifest["sets"][name] = {"path": str(path), "items": len(rows), "sha256": digest, "conditions": conds}
        print(f"{name}: {len(rows)} items, {len(conds)} conditions -> {path}")
    out = Path(args.manifest_dir) / f"manifest-s{args.seed}{'-' + args.tag if args.tag else ''}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--sets", default=None)
    b.add_argument("--scale", type=float, default=1.0, help="item counts × scale (smoke: 0.02)")
    b.add_argument("--width", type=int, default=512)
    b.add_argument("--output", type=Path, default=OUT)
    b.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    b.add_argument("--tag", default="")
    v = sub.add_parser("verdict", help="LG-B verdict-format sets from the built taxonomy and rulebase sets (pre-registration §14)")
    v.add_argument("--seed", type=int, default=1)
    v.add_argument("--output", type=Path, default=OUT)
    v.add_argument("--manifest-dir", type=Path, default=MANIFEST_DIR)
    args = parser.parse_args(argv)
    {"build": run_build, "verdict": run_verdict}[args.command](args)


if __name__ == "__main__":
    main()
