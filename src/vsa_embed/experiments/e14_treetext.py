"""E14 addendum 9 (pre-registration `experiments/e14-logic/preregistration-addendum-9-treetext.md`): can a frozen host read a
decision tree described in prose and write it in the executor's structured form, accurately enough that executing the
decoded tree matches the gold tree?

Items (`build`, set `trees-text-v1`, seed 1): LG-D's tree generator (`world.random_tree`, 8 attributes, 6 outcomes, half the
trees with numeric attributes) at depths 2, 3 and 4, under its own seed key and namer; per tree 24 test records with the
exact leaf (`reference.run_tree`). Each tree is rendered to prose in two styles:
- `templated` — a numbered step list in pre-order, one fixed sentence per test
  ("Step 2: if its plonk is varn, send it to the drusk desk; otherwise, go to step 3."), unambiguous and regex-parseable;
- `paraphrased` — a small grammar: steps after the first in random order, varied connectives, negated tests with swapped
  branches ("unless", "when ... exceeds"), outcome and go-to phrasings varied, and a depth-1 subtree inlined as a nested
  clause of its parent's sentence with probability 1/2.
Two worked examples (depth 2 categorical, depth 3 numeric) come from a separate seed key and are held out of the item set.

Structured form: the executor's `world.tree_text` lines ("If its <attr> is <value>:" / "If its <attr> is at most <n>:",
the yes branch indented, "Otherwise:", the no branch, leaves as "then <action>."). `parse_tree` is tolerant (indentation,
case, punctuation and a few spellings are ignored; the if / otherwise / then grammar is what is parsed).

Reader (`read`): greedy generation from a fixed prompt (instruction + the two worked examples in the item's style), one
pass per host and style. Scoring per item: parse rate, exact tree match, per-node test accuracy aligned by path, and the
primary, **leaf accuracy**: the decoded tree executed on the item's 24 records against the gold tree's leaves. Baselines:
`regex` (the templated style's parser: the format's upper bound), `first-leaf` (a reader that only copies the first
outcome) and gold (1.0). `report` pools hosts: means with 2,000-resample bootstrap intervals over trees, by style and
depth, paired style and host contrasts, and the pre-registered decision rule.

    python -m vsa_embed.experiments.e14_treetext build [--seed 1] [--per-depth 24] [--records 24] [--output DIR]
    python -m vsa_embed.experiments.e14_treetext read --host H-q17 --items DIR --output OUT [--styles templated,paraphrased]
        [--limit N] [--depths 2,4] [--device cpu] [--batch-size 8] [--smoke]
    python -m vsa_embed.experiments.e14_treetext report --runs OUT ... --items DIR --output DIR [--resamples 2000]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from ..logic import reference as R
from ..logic import world as W
from .e5_common import finish_output, start_output, write_json

SET = "trees-text-v1"
STYLES = ("templated", "paraphrased")
DEPTHS = (2, 3, 4)
ITEMS_DIR = Path("experiments/e14-logic/items") / SET
HOST_NAMES = {"H-135": "SmolLM2-135M", "H-360": "SmolLM2-360M", "H-q06": "Qwen3-0.6B", "H-q17": "Qwen3-1.7B"}
DECISION = {"leaf_accuracy": 0.8, "ci_low": 0.7, "style": "paraphrased"}      # the addendum's rule: worth queuing more iff met

INSTRUCTION = ("Each policy below is a decision procedure for handling a request. Rewrite it as a decision tree in the standard "
               "form. Every test is written as \"If its <attribute> is <value>:\" or \"If its <attribute> is at most <number>:\"; "
               "the branch taken when the test holds comes next, indented; then \"Otherwise:\" and the other branch, indented; "
               "every outcome is written as \"then <action>.\". Start from step 1 and follow the steps in the order they are used.")


# -- tree helpers -------------------------------------------------------------------------------------------------------

def preorder(tree: dict[str, Any]) -> list[dict[str, Any]]:
    """Internal nodes in pre-order."""
    out: list[dict[str, Any]] = []

    def walk(n: dict[str, Any]) -> None:
        if "leaf" in n:
            return
        out.append(n); walk(n["yes"]); walk(n["no"])
    walk(tree)
    return out


def leaves(tree: dict[str, Any]) -> list[str]:
    return [tree["leaf"]] if "leaf" in tree else leaves(tree["yes"]) + leaves(tree["no"])


def is_cherry(n: dict[str, Any]) -> bool:
    return "test" in n and "leaf" in n["yes"] and "leaf" in n["no"]


def target_text(tree: dict[str, Any]) -> str:
    return "\n".join(W.tree_text(tree))


# -- prose renderers ---------------------------------------------------------------------------------------------------

def _cond(test: dict[str, Any]) -> str:
    return W.test_text(test)                                             # "its X is v" / "its X score is at most t"


def _step_numbers(tree: dict[str, Any], inlined: set[int]) -> dict[int, int]:
    """node id → step number: pre-order over the internal nodes that are not inlined."""
    numbers, k = {}, 0
    for n in preorder(tree):
        if id(n) not in inlined:
            k += 1
            numbers[id(n)] = k
    return numbers


def render_templated(tree: dict[str, Any]) -> str:
    numbers = _step_numbers(tree, set())

    def action(n: dict[str, Any]) -> str:
        return n["leaf"] if "leaf" in n else f"go to step {numbers[id(n)]}"
    steps = [f"Step {numbers[id(n)]}: if {_cond(n['test'])}, {action(n['yes'])}; otherwise, {action(n['no'])}."
             for n in preorder(tree)]
    return f"The policy has {len(steps)} steps; start at step 1.\n" + "\n".join(steps)


COND_IS = ["its {a} is {v}", "the {a} is {v}", "its {a} reads {v}", "it has {v} as its {a}", "the {a} comes out as {v}"]
NEG_IS = ["its {a} is not {v}", "its {a} is anything other than {v}", "the {a} differs from {v}", "its {a} is something other than {v}"]
COND_LE = ["its {a} is at most {v}", "its {a} is {v} or less", "its {a} does not exceed {v}", "its {a} is no higher than {v}",
           "the {a} is not above {v}"]
NEG_LE = ["its {a} is above {v}", "its {a} exceeds {v}", "its {a} is more than {v}", "its {a} is higher than {v}"]
LEAF = ["{leaf}", "it goes to the {desk} desk", "route it to the {desk} desk", "the {desk} desk takes it", "pass it on to the {desk} desk",
        "it belongs with the {desk} desk"]
GOTO = ["go to step {j}", "continue with step {j}", "move on to step {j}", "apply step {j}", "step {j} decides"]
INTRO = ["Here is how a request is handled.", "The handling rules are these; begin with step 1.",
         "Process every request as follows, starting from step 1.", "A request is dealt with by the steps below, from step 1."]
# (sentence, uses the negated condition)
STEP = [("Step {k}: if {cond}, {yes}; otherwise, {no}.", False),
        ("At step {k}, ask whether {cond}. If so, {yes}. If not, {no}.", False),
        ("Step {k}: {no} unless {cond}, in which case {yes}.", False),
        ("Step {k}: when {neg}, {no}; when {cond}, {yes}.", True),
        ("Step {k}: should {neg}, {no}; in every other case, {yes}.", True),
        ("For step {k}: {yes} as long as {cond}; failing that, {no}.", False)]
INLINE = ["it then comes down to whether {cond}: if that holds, {yes}; if it fails, {no}",
          "the decision then turns on whether {cond} ({yes} when it does, {no} when it does not)",
          "one more check decides: {cond}? If yes, {yes}; if no, {no}"]
_DESK = re.compile(r"the (\S+) desk")


def render_paraphrased(tree: dict[str, Any], rng: random.Random) -> str:
    """Steps after the first in random order; varied connectives; negated tests with swapped branches; a cherry (a test whose
    two branches are outcomes) inlined into its parent's sentence with probability 1/2 (at most one per parent)."""
    nodes = preorder(tree)
    inlined: set[int] = set()
    for n in nodes:
        kids = [c for c in (n["yes"], n["no"]) if is_cherry(c)]
        if kids and rng.random() < 0.5:
            inlined.add(id(rng.choice(kids)))
    numbers = _step_numbers(tree, inlined)

    def cond(test: dict[str, Any], neg: bool = False) -> str:
        pool = (NEG_IS if neg else COND_IS) if test["op"] == "is" else (NEG_LE if neg else COND_LE)
        return rng.choice(pool).format(a=test["attr"], v=test["value"])

    def leaf(n: dict[str, Any]) -> str:
        m = _DESK.search(n["leaf"])
        return rng.choice(LEAF).format(leaf=n["leaf"], desk=m.group(1)) if m else n["leaf"]

    def action(n: dict[str, Any]) -> str:
        return leaf(n) if "leaf" in n else rng.choice(GOTO).format(j=numbers[id(n)])

    sentences = []
    for n in nodes:
        if id(n) in inlined:
            continue
        k = numbers[id(n)]
        inline_child = next((c for c in (n["yes"], n["no"]) if id(c) in inlined), None)
        if inline_child is not None:                                     # the inlined clause goes last, after a full stop
            c = inline_child
            clause = rng.choice(INLINE).format(cond=cond(c["test"]), yes=leaf(c["yes"]), no=leaf(c["no"]))
            if c is n["no"]:
                first = rng.choice([f"Step {k}: if {cond(n['test'])}, {action(n['yes'])}.",
                                    f"At step {k}, {action(n['yes'])} when {cond(n['test'])}."])
            else:
                first = rng.choice([f"Step {k}: if {cond(n['test'], neg=True)}, {action(n['no'])}.",
                                    f"At step {k}, {action(n['no'])} when {cond(n['test'], neg=True)}."])
            sentences.append((k, f"{first} Otherwise {clause}."))
            continue
        template, _ = rng.choice(STEP)
        sentences.append((k, template.format(k=k, cond=cond(n["test"]), neg=cond(n["test"], neg=True), yes=action(n["yes"]),
                                             no=action(n["no"]))))
    first, rest = sentences[0], sentences[1:]
    rng.shuffle(rest)
    return rng.choice(INTRO) + "\n" + "\n".join(s for _, s in [first] + rest)


# -- the template-regex reader (baseline for the templated style) --------------------------------------------------------

_STEP = re.compile(r"^Step (\d+): if its (.+?) is (?:at most (\d+)|(.+?)), (.+?); otherwise, (.+?)\.$")
_GOTO = re.compile(r"^go to step (\d+)$")


def regex_reader(text: str) -> dict[str, Any] | None:
    steps: dict[int, tuple[dict[str, Any], str, str]] = {}
    for line in text.splitlines():
        m = _STEP.match(line.strip())
        if not m:
            continue
        k, attr, le, eq, yes, no = m.groups()
        test = {"attr": attr, "op": "le", "value": int(le)} if le is not None else {"attr": attr, "op": "is", "value": eq}
        steps[int(k)] = (test, yes, no)

    def build(k: int, seen: set[int]) -> dict[str, Any] | None:
        if k not in steps or k in seen:
            return None
        test, yes, no = steps[k]
        branches = []
        for a in (yes, no):
            g = _GOTO.match(a)
            b = build(int(g.group(1)), seen | {k}) if g else {"leaf": a}
            if b is None:
                return None
            branches.append(b)
        return {"test": test, "yes": branches[0], "no": branches[1]}
    return build(1, set()) if steps else None


# -- the tolerant parser of the structured form -------------------------------------------------------------------------

_IF_LE = re.compile(r"^(?:if|when)\s+(?:its\s+|the\s+)?(.+?)\s+(?:is\s+at\s+most|is\s+no\s+more\s+than|is\s+not\s+above|at\s+most|<=|≤|is\s+<=)\s+(-?\d+(?:\.\d+)?)\s*:?\s*$", re.I)
_IF_GT = re.compile(r"^(?:if|when)\s+(?:its\s+|the\s+)?(.+?)\s+(?:is\s+above|is\s+more\s+than|is\s+higher\s+than|is\s+greater\s+than|exceeds|is\s+over|>)\s+(-?\d+(?:\.\d+)?)\s*:?\s*$", re.I)
_IF_NOT = re.compile(r"^(?:if|when)\s+(?:its\s+|the\s+)?(.+?)\s+(?:is\s+not|isn't|is\s+anything\s+other\s+than|is\s+something\s+other\s+than|is\s+anything\s+but|differs\s+from|!=)\s+(.+?)\s*:?\s*$", re.I)
_IF_IS = re.compile(r"^(?:if|when)\s+(?:its\s+|the\s+)?(.+?)\s+(?:is|=|==|equals)\s+(.+?)\s*:?\s*$", re.I)
_OTHERWISE = re.compile(r"^(?:otherwise|else|if\s+not)\s*[:,.]?\s*$", re.I)
_THEN = re.compile(r"^(?:then\s+|->\s*|=>\s*)?(.+?)\s*\.?\s*$", re.I)


class ParseError(ValueError):
    pass


def canon_leaf(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip().rstrip(".").strip().lower())
    m = _DESK.search(text)
    return f"send it to the {m.group(1)} desk" if m else text


def canon_test(test: dict[str, Any]) -> tuple[str, str, Any]:
    attr = re.sub(r"\s+", " ", str(test["attr"]).strip().lower())
    value = test["value"]
    if test["op"] == "le":
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = str(value)
    else:
        value = re.sub(r"\s+", " ", str(value).strip().lower())
    return attr, test["op"], value


def canon_tree(tree: dict[str, Any]) -> Any:
    if "leaf" in tree:
        return ("leaf", canon_leaf(tree["leaf"]))
    return (canon_test(tree["test"]), canon_tree(tree["yes"]), canon_tree(tree["no"]))


def parse_tree(text: str) -> dict[str, Any] | None:
    """The structured form back to a tree dict, or None when the text does not follow the if / otherwise / then grammar
    (leading indentation, case, trailing punctuation and a few spellings are tolerated; a negated test — "is not v", "is above
    t", "exceeds t" — is accepted and normalised to the executor's form by swapping its branches; text after the first
    complete tree is ignored)."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pos = 0

    def node() -> dict[str, Any]:
        nonlocal pos
        if pos >= len(lines):
            raise ParseError("unexpected end")
        line = lines[pos]
        pos += 1
        if _OTHERWISE.match(line):
            raise ParseError("otherwise without if")
        negated = False
        for pattern, op, neg in ((_IF_LE, "le", False), (_IF_GT, "le", True), (_IF_NOT, "is", True), (_IF_IS, "is", False)):
            m = pattern.match(line)
            if m:
                attr, value = m.groups()
                test: dict[str, Any] = {"attr": attr, "op": op, "value": (float(value) if "." in value else int(value)) if op == "le" else value.strip()}
                negated = neg
                break
        else:
            m = _THEN.match(line)
            if not m or not m.group(1).strip():
                raise ParseError(f"bad line: {line!r}")
            return {"leaf": m.group(1).strip()}
        first = node()
        if pos >= len(lines) or not _OTHERWISE.match(lines[pos]):
            raise ParseError("missing otherwise")
        pos += 1
        second = node()
        return {"test": test, "yes": second, "no": first} if negated else {"test": test, "yes": first, "no": second}
    try:
        return node()
    except ParseError:
        return None


# -- scoring ------------------------------------------------------------------------------------------------------------

def resolve_attr(name: str, attributes: Sequence[dict[str, Any]]) -> str:
    """The item's attribute a decoded name refers to (exact, then containment either way), else the name itself."""
    low = name.strip().lower()
    names = [a["name"] for a in attributes]
    for n in names:
        if n.lower() == low:
            return n
    hits = [n for n in names if low in n.lower() or n.lower() in low]
    return hits[0] if len(hits) == 1 else name


def run_decoded(tree: dict[str, Any], record: dict[str, Any], attributes: Sequence[dict[str, Any]]) -> str | None:
    """The decoded tree executed on a record; None where a test cannot be evaluated (unknown attribute, non-numeric
    threshold)."""
    node = tree
    for _ in range(10_000):
        if "leaf" in node:
            return canon_leaf(node["leaf"])
        test = node["test"]
        attr = resolve_attr(test["attr"], attributes)
        if attr not in record:
            return None
        value = record[attr]
        if test["op"] == "is":
            ok = str(value).strip().lower() == str(test["value"]).strip().lower()
        else:
            try:
                ok = float(value) <= float(test["value"])
            except (TypeError, ValueError):
                return None
        node = node["yes"] if ok else node["no"]
    return None


def score_tree(decoded: dict[str, Any] | None, item: dict[str, Any]) -> dict[str, Any]:
    gold = item["tree"]
    internal, leaf_nodes = [], []

    def walk(g: dict[str, Any], d: dict[str, Any] | None) -> None:
        if "leaf" in g:
            leaf_nodes.append(int(d is not None and "leaf" in d and canon_leaf(d["leaf"]) == canon_leaf(g["leaf"])))
            return
        ok = d is not None and "test" in d and canon_test(_resolved(d["test"], item)) == canon_test(g["test"])
        internal.append(int(ok))
        nxt = d if (d is not None and "test" in d) else None
        walk(g["yes"], nxt["yes"] if nxt else None); walk(g["no"], nxt["no"] if nxt else None)
    walk(gold, decoded)
    leaf_hits = [int(decoded is not None and run_decoded(decoded, r["record"], item["attributes"]) == canon_leaf(r["gold"]))
                 for r in item["records"]]
    return {"parsed": int(decoded is not None),
            "exact": int(decoded is not None and canon_tree(_resolve_tree(decoded, item)) == canon_tree(gold)),
            "node_accuracy": float(np.mean(internal)), "leaf_node_accuracy": float(np.mean(leaf_nodes)),
            "leaf_accuracy": float(np.mean(leaf_hits)), "decoded_internal": len(preorder(decoded)) if decoded else 0,
            "gold_internal": len(internal)}


def _resolved(test: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    return {**test, "attr": resolve_attr(test["attr"], item["attributes"])}


def _resolve_tree(tree: dict[str, Any], item: dict[str, Any]) -> dict[str, Any]:
    if "leaf" in tree:
        return tree
    return {"test": _resolved(tree["test"], item), "yes": _resolve_tree(tree["yes"], item), "no": _resolve_tree(tree["no"], item)}


METRICS = ("parsed", "exact", "node_accuracy", "leaf_node_accuracy", "leaf_accuracy")


# -- items --------------------------------------------------------------------------------------------------------------

def _tree_item(seed: int, depth: int, q: int, rng: random.Random, namer: W.Namer, *, records: int, prefix: str) -> dict[str, Any]:
    numeric = q % 2 == 1
    attrs = W._attributes(namer, 8, 0.5 if numeric else 0.0, rng)
    outcomes = [f"send it to the {namer.word()} desk" for _ in range(6)]
    tree = W.random_tree(depth, attrs, outcomes, rng)
    recs = []
    for _ in range(records):
        record = {a.name: (rng.choice(a.values) if a.values else rng.randint(0, 100)) for a in attrs}
        leaf, _ = R.run_tree(tree, record)
        recs.append({"record": record, "gold": leaf})
    item_id = f"{prefix}-{seed}-{depth}-{q}"
    text = {"templated": render_templated(tree),
            "paraphrased": render_paraphrased(tree, random.Random(f"trees-text-para:{seed}:{item_id}"))}
    return {"id": item_id, "set": SET, "world": "trees-text", "depth": depth, "numeric": numeric, "attributes": [a.__dict__ for a in attrs],
            "outcomes": outcomes, "tree": tree, "records": recs, "text": text, "target": target_text(tree)}


def build_items(seed: int, *, depths: Sequence[int] = DEPTHS, per_depth: int = 24, records: int = 24,
                namer: W.Namer | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(items, worked examples). The examples come from their own seed key and namer, so no item shares a tree or a name
    with them."""
    rng = random.Random(f"trees-text:{seed}")
    namer = namer or W.Namer(seed + 50)
    items = [_tree_item(seed, depth, q, rng, namer, records=records, prefix="ttree") for depth in depths for q in range(per_depth)]
    drng = random.Random(f"trees-text-demo:{seed}")
    dnamer = W.Namer(seed + 51, forbidden=namer.forbidden)
    demos = [_tree_item(seed, 2, 0, drng, dnamer, records=4, prefix="demo"), _tree_item(seed, 3, 1, drng, dnamer, records=4, prefix="demo")]
    return items, demos


def baseline_scores(items: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """`regex` on the templated text, `first-leaf`, `gold`; each a mean of every metric over items."""
    readers: dict[str, Callable[[dict[str, Any]], dict[str, Any] | None]] = {
        "regex:templated": lambda it: regex_reader(it["text"]["templated"]),
        "first-leaf": lambda it: {"leaf": leaves(it["tree"])[0]},
        "gold": lambda it: it["tree"]}
    out = {}
    for name, reader in readers.items():
        rows = [score_tree(reader(it), it) for it in items]
        out[name] = {m: float(np.mean([r[m] for r in rows])) for m in METRICS}
        out[name]["n"] = len(rows)
    return out


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_items(items: Sequence[dict[str, Any]], demos: Sequence[dict[str, Any]], out: Path, *, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.jsonl").write_text("".join(json.dumps(it, sort_keys=True) + "\n" for it in items))
    (out / "demos.json").write_text(json.dumps(list(demos), sort_keys=True, indent=1) + "\n")
    counts = defaultdict(int)
    for it in items:
        counts[f"depth={it['depth']}"] += 1
        counts[f"numeric={it['numeric']}"] += 1
    manifest = {"set": SET, "seed": seed, **config, "items": len(items), "records_per_item": len(items[0]["records"]) if items else 0,
                "counts": dict(counts), "styles": list(STYLES), "demos": [d["id"] for d in demos],
                "files": {name: _sha256(out / name) for name in ("items.jsonl", "demos.json")},
                "baselines": baseline_scores(items),
                "text_chars": {s: {f"depth={d}": float(np.mean([len(it["text"][s]) for it in items if it["depth"] == d]))
                                   for d in sorted({it["depth"] for it in items})} for s in STYLES}}
    write_json(out / "manifest.json", manifest)
    return manifest


def load_items(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    items = [json.loads(ln) for ln in (path / "items.jsonl").read_text().splitlines() if ln.strip()]
    demos = json.loads((path / "demos.json").read_text())
    return items, demos


def run_build(args: argparse.Namespace) -> dict[str, Any]:
    items, demos = build_items(args.seed, per_depth=args.per_depth, records=args.records)
    manifest = write_items(items, demos, Path(args.output), seed=args.seed, config={"per_depth": args.per_depth, "records": args.records,
                                                                                    "depths": list(DEPTHS)})
    print(f"{SET}: {len(items)} trees -> {args.output}")
    print(json.dumps(manifest["baselines"], indent=1))
    return manifest


# -- the reader ---------------------------------------------------------------------------------------------------------

def prompt_prefix(demos: Sequence[dict[str, Any]], style: str) -> str:
    parts = [INSTRUCTION, ""]
    for d in demos:
        parts += [f"Policy:\n{d['text'][style]}", "", f"Decision tree:\n{d['target']}", "", "###", ""]
    return "\n".join(parts)


def prompt_of(demos: Sequence[dict[str, Any]], item: dict[str, Any], style: str) -> str:
    return prompt_prefix(demos, style) + f"Policy:\n{item['text'][style]}\n\nDecision tree:\n"


def read_items(host: Any, items: Sequence[dict[str, Any]], demos: Sequence[dict[str, Any]], styles: Sequence[str], *, margin: float = 1.5,
               log=print) -> list[dict[str, Any]]:
    """One greedy generation per item and style, grouped by style and depth (the token budget of a group is `margin` times its
    longest gold target plus 16, so a correct tree is never cut)."""
    out = []
    for style in styles:
        for depth in sorted({it["depth"] for it in items}):
            group = [it for it in items if it["depth"] == depth]
            budget = int(margin * max(host.count(it["target"]) for it in group)) + 16
            prompts = [prompt_of(demos, it, style) for it in group]
            t0 = time.monotonic()
            texts = host.generate(prompts, max_new_tokens=budget, stop=("\n\n", "###", "\nPolicy:"))
            log(f"  {style} depth {depth}: {len(group)} items, budget {budget} tokens, prompt ~{host.count(prompts[0])} tokens, "
                f"{time.monotonic() - t0:.0f}s")
            for it, text in zip(group, texts):
                decoded = parse_tree(text)
                n_out = host.count(text)
                out.append({"id": it["id"], "style": style, "depth": it["depth"], "numeric": it["numeric"], "output": text,
                            "output_tokens": n_out, "budget": budget, "truncated": int(n_out >= budget - 1), "decoded": decoded,
                            **score_tree(decoded, it)})
    return out


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in METRICS:
            acc[r["style"]][f"{m}|all"].append(r[m])
            acc[r["style"]][f"{m}|depth={r['depth']}"].append(r[m])
    return {style: {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in subs.items()} for style, subs in acc.items()}


def run_read(args: argparse.Namespace) -> dict[str, Any]:
    from ..logic.hosts import HostLM
    styles = [s.strip() for s in args.styles.split(",")] if args.styles else list(STYLES)
    config = {"experiment": "e14-treetext", "host": args.host, "items": str(args.items), "styles": styles, "limit": args.limit,
              "depths": args.depths, "margin": args.margin, "smoke": bool(args.smoke)}
    output = Path(args.output)
    git_at_start = start_output(output, config)
    items, demos = load_items(Path(args.items))
    if args.depths:
        keep = {int(d) for d in args.depths.split(",")}
        items = [it for it in items if it["depth"] in keep]
    if args.limit:
        items = items[: args.limit]
    host = HostLM(args.host, device=args.device, batch_size=args.batch_size)
    t0 = time.monotonic()
    rows = read_items(host, items, demos, styles, margin=args.margin)
    with open(output / "generations.jsonl", "w") as handle:
        for r in rows:
            handle.write(json.dumps(r) + "\n")
    summary = {"source": host.describe(), "items": len(items), "styles": styles, "smoke": bool(args.smoke),
               "seconds": round(time.monotonic() - t0, 1), "truncated": int(sum(r["truncated"] for r in rows)),
               "prompt_tokens": {s: host.count(prompt_of(demos, items[0], s)) for s in styles}, "accuracy": summarize(rows)}
    write_json(output / "summary.json", summary)
    lines = (["**SMOKE TEST — not a result.**", ""] if args.smoke else []) + [f"# E14 tree-text reading — `{host.name}`", "",
             f"{len(items)} trees; {summary['seconds']}s; {summary['truncated']} truncated outputs.", "",
             "| style | metric | all | by depth |", "|---|---|---:|---|"]
    for style, subs in summary["accuracy"].items():
        for m in METRICS:
            depths = ", ".join(f"{k.split('=')[1]}: {v['mean']:.2f}" for k, v in sorted(subs.items()) if k.startswith(f"{m}|depth="))
            lines.append(f"| {style} | {m} | {subs[f'{m}|all']['mean']:.3f} ({subs[f'{m}|all']['n']}) | {depths} |")
    (output / "report.md").write_text("\n".join(lines) + "\n")
    finish_output(output, config, git_at_start=git_at_start, device=host.device, source=host.describe())
    return summary


# -- report -------------------------------------------------------------------------------------------------------------

def mean_ci(values: Sequence[float], *, resamples: int, seed: int) -> dict[str, Any]:
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    draws = v[np.random.default_rng(seed).integers(0, v.size, size=(resamples, v.size))].mean(1)
    return {"mean": float(v.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)), "n": int(v.size)}


def paired(a: dict[str, float], b: dict[str, float], *, resamples: int, seed: int) -> dict[str, Any] | None:
    from .e5_zeroshot import paired_difference
    ids = sorted(set(a) & set(b))
    if not ids:
        return None
    return paired_difference(np.array([a[i] for i in ids]), np.array([b[i] for i in ids]), resamples=resamples, seed=seed)


def fmt(c: dict[str, Any] | None, signed: bool = False) -> str:
    if not c or c["n"] == 0:
        return "—"
    s = f"{c['mean']:+.3f}" if signed else f"{c['mean']:.3f}"
    return s + f" [{c['ci_low']:.3f}, {c['ci_high']:.3f}] (n {c['n']})" + (f" p {c['p_value']:.4f}" if "p_value" in c else "")


def host_label(name: str) -> str:
    for short in HOST_NAMES.values():
        if short in name:
            return short
    return name.rsplit("/", 1)[-1]


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    items, _ = load_items(Path(args.items))
    by_id = {it["id"]: it for it in items}
    present = [Path(p) for p in args.runs if (Path(p) / "summary.json").exists()]
    missing = [str(p) for p in args.runs if not (Path(p) / "summary.json").exists()]
    doc: dict[str, Any] = {"items": str(args.items), "missing": missing, "decision_rule": DECISION, "hosts": {}, "baselines": {}}
    lines = ["# E14 addendum 9 — decision trees read from prose (tree-text)", "",
             f"Items `{args.items}` ({len(items)} trees, {len(items[0]['records'])} records each). Means over trees with "
             f"{args.resamples}-resample percentile bootstrap intervals. Primary: leaf accuracy (the decoded tree executed on the "
             f"records against the gold tree).", ""] + ([f"Missing: {', '.join(missing)}", ""] if missing else [])
    runs: dict[str, list[dict[str, Any]]] = {}
    for run in present:                                                      # rescored from the stored outputs with the current parser
        summary = json.loads((run / "summary.json").read_text())
        rows = [json.loads(ln) for ln in (run / "generations.jsonl").read_text().splitlines() if ln.strip()]
        rows = [{**r, **score_tree(parse_tree(r["output"]), by_id[r["id"]])} for r in rows if r["id"] in by_id]
        runs.setdefault(host_label(summary["source"]["host"]), []).extend(rows)        # a host's folders are pooled
        if summary.get("smoke"):
            lines += [f"**{host_label(summary['source']['host'])} is a SMOKE run — not a result.**", ""]
    for name, rows in baseline_rows(items).items():
        doc["baselines"][name] = {m: mean_ci([r[m] for r in rows], resamples=args.resamples, seed=args.seed) for m in METRICS}
    lines += ["## Baselines (no model)", "", "| reader | parse | exact | node acc | leaf acc |", "|---|---:|---:|---:|---:|"]
    for name, b in doc["baselines"].items():
        lines.append(f"| {name} | {b['parsed']['mean']:.3f} | {b['exact']['mean']:.3f} | {fmt(b['node_accuracy'])} | {fmt(b['leaf_accuracy'])} |")
    lines += ["", "## Hosts", "", "| host | style | parse | exact | node acc | leaf acc | truncated |", "|---|---|---:|---:|---:|---:|---:|"]
    for host, rows in runs.items():
        rep: dict[str, Any] = {"styles": {}, "by_depth": {}, "contrasts": {}}
        for style in STYLES:
            sub = [r for r in rows if r["style"] == style]
            if not sub:
                continue
            rep["styles"][style] = {m: mean_ci([r[m] for r in sub], resamples=args.resamples, seed=args.seed) for m in METRICS}
            rep["styles"][style]["truncated"] = int(sum(r["truncated"] for r in sub))
            rep["by_depth"][style] = {d: {m: mean_ci([r[m] for r in sub if r["depth"] == d], resamples=args.resamples, seed=args.seed)
                                          for m in ("leaf_accuracy", "exact", "node_accuracy")} for d in sorted({r["depth"] for r in sub})}
            rep["by_depth"][style]["numeric"] = {str(v): mean_ci([r["leaf_accuracy"] for r in sub if r["numeric"] == v],
                                                                 resamples=args.resamples, seed=args.seed) for v in (False, True)}
            s = rep["styles"][style]
            lines.append(f"| {host} | {style} | {s['parsed']['mean']:.3f} | {s['exact']['mean']:.3f} | {fmt(s['node_accuracy'])} | "
                         f"**{fmt(s['leaf_accuracy'])}** | {s['truncated']} |")
        leaf = {style: {r["id"]: r["leaf_accuracy"] for r in rows if r["style"] == style} for style in STYLES}
        rep["contrasts"]["templated_minus_paraphrased"] = paired(leaf["templated"], leaf["paraphrased"], resamples=args.resamples, seed=args.seed)
        doc["hosts"][host] = rep
    lines += ["", "### By depth (leaf accuracy [exact])", "", "| host | style | " + " | ".join(f"depth {d}" for d in DEPTHS) + " | categorical | numeric |",
              "|---|---|" + "---:|" * (len(DEPTHS) + 2)]
    for host, rep in doc["hosts"].items():
        for style, bd in rep["by_depth"].items():
            cells = [f"{fmt(bd[d]['leaf_accuracy'])} [{bd[d]['exact']['mean']:.2f}]" if d in bd else "—" for d in DEPTHS]
            cells += [fmt(bd["numeric"]["False"]), fmt(bd["numeric"]["True"])]
            lines.append(f"| {host} | {style} | " + " | ".join(cells) + " |")
    lines += ["", "### Contrasts (leaf accuracy, paired over trees)", ""]
    for host, rep in doc["hosts"].items():
        lines.append(f"- {host}: templated − paraphrased {fmt(rep['contrasts']['templated_minus_paraphrased'], signed=True)}")
    hosts = list(runs)
    for i, a in enumerate(hosts):
        for b in hosts[i + 1:]:
            for style in STYLES:
                c = paired({r["id"]: r["leaf_accuracy"] for r in runs[a] if r["style"] == style},
                           {r["id"]: r["leaf_accuracy"] for r in runs[b] if r["style"] == style}, resamples=args.resamples, seed=args.seed)
                doc["hosts"][a]["contrasts"][f"minus {b} ({style})"] = c
                lines.append(f"- {a} − {b}, {style}: {fmt(c, signed=True)}")
    # the decision rule
    best = None
    for host, rep in doc["hosts"].items():
        s = rep["styles"].get(DECISION["style"])
        if s and (best is None or s["leaf_accuracy"]["mean"] > best[1]["mean"]):
            best = (host, s["leaf_accuracy"])
    smoke = any(json.loads((run / "summary.json").read_text()).get("smoke") for run in present)
    verdict = bool(best and best[1]["mean"] >= DECISION["leaf_accuracy"] and best[1]["ci_low"] >= DECISION["ci_low"] and not smoke)
    doc["decision"] = {"best_host": best[0] if best else None, "leaf_accuracy": best[1] if best else None, "worth_queuing_more": verdict,
                       "smoke": smoke}
    lines += ["", "## Decision rule (addendum 9)", "",
              f"Worth queuing more iff the best host's leaf accuracy on the {DECISION['style']} style is ≥ {DECISION['leaf_accuracy']} "
              f"with the interval's lower bound ≥ {DECISION['ci_low']}.",
              f"- best host on {DECISION['style']}: {best[0] if best else '—'}, leaf accuracy {fmt(best[1]) if best else '—'}",
              f"- **verdict: {'worth queuing more' if verdict else 'not worth queuing more'}**" + (" (SMOKE runs present: no verdict)" if smoke else "")]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "report.json", doc)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    return doc


def baseline_rows(items: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {"regex:templated": [score_tree(regex_reader(it["text"]["templated"]), it) for it in items],
            "first-leaf": [score_tree({"leaf": leaves(it["tree"])[0]}, it) for it in items],
            "gold": [score_tree(it["tree"], it) for it in items]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--seed", type=int, default=1); b.add_argument("--per-depth", type=int, default=24); b.add_argument("--records", type=int, default=24)
    b.add_argument("--output", type=Path, default=ITEMS_DIR)
    r = sub.add_parser("read")
    r.add_argument("--host", required=True, help="H-360, H-q06, H-q17, a Hugging Face name or a local path")
    r.add_argument("--items", type=Path, default=ITEMS_DIR); r.add_argument("--output", type=Path, required=True)
    r.add_argument("--styles", default=None); r.add_argument("--limit", type=int, default=None); r.add_argument("--margin", type=float, default=1.5)
    r.add_argument("--depths", default=None, help="smoke only: keep these depths, e.g. 2,4")
    r.add_argument("--device", default=None); r.add_argument("--batch-size", type=int, default=8); r.add_argument("--smoke", action="store_true")
    p = sub.add_parser("report")
    p.add_argument("--runs", nargs="+", required=True); p.add_argument("--items", type=Path, default=ITEMS_DIR)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--resamples", type=int, default=2000); p.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    {"build": run_build, "read": run_read, "report": run_report}[args.command](args)


if __name__ == "__main__":
    main()
