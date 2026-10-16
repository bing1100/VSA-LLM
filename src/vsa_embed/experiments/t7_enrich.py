"""Richer T7-ROOD ontologies for the ontology-richness dose-response, stages 2–3 (2026-10-15 addendum 2,
`experiments/e9-retrofit/preregistration-addendum-2026-10-15.md`).

    python -m vsa_embed.experiments.t7_enrich mesh --output DIR                                   # stage 2a
    python -m vsa_embed.experiments.t7_enrich read --reader teacher --output DIR [--max-usd 60]   # stage 2b (Claude)
    python -m vsa_embed.experiments.t7_enrich read --reader host --run RUN --output DIR           # stage 3 (the model)
    python -m vsa_embed.experiments.t7_enrich queue --variant NAME --arms C5tch C5tchsh --priority P [--min-enriched F]

Each command writes `DIR/ontology.pt`: the base T7-ROOD ontology (`track_spec("t7rood").ontology`) with extra edges
appended to entries' frames. It also links every other item of the base corpus folder (train, eval, eval-general,
eval-pubmed, presample, hosts) into DIR. A run planned with `e9_plan --data-root DIR` therefore trains and evaluates on
the same text, alias table, entries and held-out set; only the frames, and the dictionary they draw on, differ. Existing
relation and atomic indices are unchanged; new ones are appended.

**Stage 2a, `mesh` — facts inherited from MeSH itself** (deterministic, free). A 2026 supplementary record's frame
names its mapped headings, its own drug actions, its record class and the top two tree levels of its headings. For each
mapped heading D, `mesh` adds:
- `heading_parent`: D's tree parents;
- `heading_action`: D's pharmacological actions;
- `heading_see_also`: D's related headings;
- `branch_third`: D's tree positions to two components (e.g. `branch:D12.776`), one level finer than `branch_second`.

New descriptor fillers and branches become new atomics. Frames are capped at `--max-degree` (24) edges, base edges first.

**Stages 2b and 3, `read` — facts read from each term's MeSH note.** A reader of E11 (`read_to_learn`) reads
`<name>: <note>`, the record's `<Note>` with its bibliographic parts removed (`e11_read_to_learn.scr_definition`). It
proposes (relation, filler) edges over the base relation set, with fillers resolved through the concept finder, for
**every** entry with a note (training and held-out alike). The read edges absent from the base frame are appended.
- **`teacher`:** Claude (`authoring_baselines.TeacherAuthor`, cached, cost recorded). MeSH is public domain, so the
  licence rule of E11 allows it. It stops once `--max-usd` is spent and records how many entries were read.
- **`host`:** the trained run's host reads with E11's host prompt and demonstrations (`--run`, e.g. T7-ROOD C5 seed 1).
  This is the model enriching its own ontology.

Every output records its provenance under `ontology["enrichment"]`: kind, counts, edges added and spend.

**`queue` — the arms on a variant, once its build has finished** (run as the queue job after the build). It reads
`<corpus parent>/<variant>/enrichment.json`, and plans and queues the training and evaluation jobs of `--arms`
(`e9_plan.ONTOLOGY_ARMS`, stage `t7rood`) only if at least `--min-enriched` of the entries gained an edge. The decision
is written to `queue-decision.json`, so a reader that adds nothing (the 360M host's smoke, 0 of 4 notes) costs no
training.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import torch

BASE_TRACK = "t7rood"
MESH_RELATIONS = ("heading_parent", "heading_action", "heading_see_also", "branch_third")
MAX_DEGREE = 24


def base_frames(ontology: dict[str, Any]) -> list[list[tuple[int, int]]]:
    offsets = [int(x) for x in ontology["offsets"]]
    relations, fillers = [int(x) for x in ontology["relations"]], [int(x) for x in ontology["fillers"]]
    return [list(zip(relations[a:b], fillers[a:b])) for a, b in zip(offsets[:-1], offsets[1:])]


def merged(base: Sequence[tuple[int, int]], extra: Sequence[tuple[int, int]], max_degree: int = MAX_DEGREE) -> list[tuple[int, int]]:
    """Base edges first, then the extra edges not already present, up to `max_degree`."""
    out, seen = list(base)[:max_degree], set(base)
    for edge in extra:
        if len(out) >= max_degree:
            break
        if edge not in seen:
            seen.add(edge); out.append(edge)
    return out


def with_frames(ontology: dict[str, Any], frames: Sequence[Sequence[tuple[int, int]]], *, relation_names: Sequence[str],
                atomic_names: Sequence[str], enrichment: dict[str, Any]) -> dict[str, Any]:
    """A copy of `ontology` with these frames and (extended) relation / atomic tables; every other key is kept."""
    if len(frames) != int(ontology["entry_count"]):
        raise ValueError("one frame per entry")
    if list(relation_names[:int(ontology["relation_count"])]) != list(ontology["relation_names"]):
        raise ValueError("relation indices must be kept (append only)")
    if list(atomic_names[:int(ontology["atomic_count"])]) != list(ontology["atomic_names"]):
        raise ValueError("atomic indices must be kept (append only)")
    offsets, relations, fillers = [0], [], []
    for frame in frames:
        for r, a in frame:
            relations.append(int(r)); fillers.append(int(a))
        offsets.append(len(relations))
    out = dict(ontology)
    out.update(offsets=torch.tensor(offsets, dtype=torch.long), relations=torch.tensor(relations, dtype=torch.long),
               fillers=torch.tensor(fillers, dtype=torch.long), relation_names=list(relation_names), relation_count=len(relation_names),
               atomic_names=list(atomic_names), atomic_count=len(atomic_names), enrichment=dict(enrichment))
    return out


def mesh_enrichment(ontology: dict[str, Any], descriptors: Sequence[dict[str, Any]], scrs: Sequence[dict[str, Any]], *,
                    max_degree: int = MAX_DEGREE) -> dict[str, Any]:
    """Stage 2a (module docstring): inherited heading facts and a finer tree level."""
    from ..ontologies.mesh import _parents
    by_ui = {r["ui"]: r for r in descriptors}
    scr_by_ui = {r["ui"]: r for r in scrs}
    parents = _parents(list(descriptors))
    relation_names = list(ontology["relation_names"]) + [r for r in MESH_RELATIONS if r not in ontology["relation_names"]]
    rid = {n: i for i, n in enumerate(relation_names)}
    atomic_names = list(ontology["atomic_names"])
    index = {n: i for i, n in enumerate(atomic_names)}

    def atom(name: str) -> int:
        if name not in index:
            index[name] = len(atomic_names); atomic_names.append(name)
        return index[name]

    frames, counts = [], Counter()
    for e, frame in enumerate(base_frames(ontology)):
        ui = ontology["concept_names"][ontology["entry_concepts"][e][0]]
        if ui in scr_by_ui:
            headings = [m for m in scr_by_ui[ui]["mapped"] if m in by_ui]
            counts["scr"] += 1
        elif ui in by_ui:
            headings = []                                  # a new descriptor: its own frame already has parents and actions
            counts["descriptor"] += 1
        else:
            headings = []
            counts["unknown"] += 1
        extra: list[tuple[int, int]] = []
        trees = [t for m in headings for t in by_ui[m]["trees"]] + (list(by_ui[ui]["trees"]) if ui in by_ui else [])
        for m in headings:
            extra += [(rid["heading_parent"], atom(f"mesh:{p}")) for p in parents.get(m, [])]
            extra += [(rid["heading_action"], atom(f"mesh:{a}")) for a in by_ui[m]["actions"] if a in by_ui]
            extra += [(rid["heading_see_also"], atom(f"mesh:{r}")) for r in by_ui[m]["related"] if r in by_ui]
        for tree in trees:
            parts = tree.split(".")
            if len(parts) >= 2:
                extra.append((rid["branch_third"], atom(f"branch:{'.'.join(parts[:2])}")))
        new = merged(frame, extra, max_degree)
        counts["edges_added"] += len(new) - len(frame)
        counts["entries_enriched"] += int(len(new) > len(frame))
        frames.append(new)
    enrichment = {"kind": "mesh", "relations_added": [r for r in MESH_RELATIONS], "max_degree": max_degree,
                  "atomics_added": len(atomic_names) - int(ontology["atomic_count"]), **dict(counts),
                  "edges_per_entry_base": sum(map(len, base_frames(ontology))) / max(1, len(frames)),
                  "edges_per_entry": sum(map(len, frames)) / max(1, len(frames))}
    return with_frames(ontology, frames, relation_names=relation_names, atomic_names=atomic_names, enrichment=enrichment)


def read_tasks_all(ctx: Any, records: dict[str, dict[str, Any]]) -> tuple[list[Any], list[int]]:
    """One E11 read task per entry with an informative supplementary-record note (`<name>: <note>`)."""
    from . import e11_read_to_learn as e11
    from .. import read_to_learn as rtl
    tasks, entries = [], []
    for e in range(int(ctx.ontology["entry_count"])):
        cid = e11._entry_concept_name(ctx, e)
        record = records.get(cid) or {}
        note = e11.scr_definition(record.get("note") or "")
        name = record.get("name")
        if note and name:
            tasks.append(rtl.ReadTask(cid, "scr", f"{name}: {note}", name, ctx.frame(e), own_atom=ctx.own_atom(cid)))
            entries.append(e)
    return tasks, entries


def read_enrichment(reader: str, *, run: Path | None = None, cache: Path | None = None, max_usd: float = 60.0, batch: int = 8,
                    chunk: int = 200, model: str = "claude-opus-5-5", device: str = "cuda", max_degree: int = MAX_DEGREE,
                    limit: int | None = None, log: Any = print) -> dict[str, Any]:
    """Stages 2b / 3 (module docstring): the base ontology plus the edges `reader` reads from every entry's note."""
    from . import e9_tracks as tracks
    from . import e11_read_to_learn as e11
    from .. import read_to_learn as rtl
    from ..ontologies.mesh_novel import parse_supplementary
    ctx = e11.track_context(BASE_TRACK)
    spec = tracks.track_spec(BASE_TRACK)
    config = tracks._load_config(spec)["ontology"]
    records = {r["ui"]: r for r in parse_supplementary(Path(config["supplementary_path"]).expanduser())}
    tasks, entries = read_tasks_all(ctx, records)
    import random
    order = list(range(len(tasks)))
    random.Random(20261015).shuffle(order)         # a cap or limit stops on a random subset, not the first UIs
    tasks, entries = [tasks[i] for i in order], [entries[i] for i in order]
    if limit:
        tasks, entries = tasks[:int(limit)], entries[:int(limit)]
    results: list[Any] = []
    record: dict[str, Any] = {"kind": reader, "notes": len(tasks), "max_degree": max_degree}
    started = time.monotonic()
    if reader == "teacher":
        from ..authoring_baselines import TeacherAuthor
        descriptions = {r: r.replace("_", " ") for r in ctx.typing.relation_names}
        teacher = TeacherAuthor(Path(cache or Path("~/data/vsa-llm/e9/enrich/teacher-cache").expanduser()), ctx.typing.relation_names,
                                descriptions, filler_hint="a concept named in the entry text, copied exactly as it is written there.",
                                model=model)
        for i in range(0, len(tasks), chunk):
            if teacher.spent_usd >= max_usd:
                record["stopped_at_cap"] = True
                break
            results += rtl.read_teacher(tasks[i:i + chunk], ctx.fillers, ctx.typing, teacher, batch=batch)
            log(json.dumps({"read": len(results), "of": len(tasks), "usd": round(float(teacher.spent_usd), 3)}))
        record.update(model=model, spent_usd=float(teacher.spent_usd), calls=getattr(teacher, "calls", None),
                      cached=getattr(teacher, "cached", None), max_usd=max_usd)
    elif reader == "host":
        if run is None:
            raise ValueError("the host reader needs --run (a trained T7-ROOD model)")
        from .e5_common import open_run
        trained = open_run(Path(run), device=device, alias_table=tracks.ensure_alias_table(spec))   # never the WordNet fallback
        generator = e11.host_generator(trained)
        demos = e11.demonstrations_for(SimpleNamespace(concepts=[], kind="heldout"), ctx)
        for i in range(0, len(tasks), chunk):
            results += rtl.read_host(tasks[i:i + chunk], ctx.fillers, ctx.typing, generator, demos)
            log(json.dumps({"read": len(results), "of": len(tasks)}))
        record.update(run=str(run))
    else:
        raise ValueError("reader must be teacher or host")
    read = {entries[i]: list(r.frame or []) for i, r in enumerate(results)}
    frames, counts = [], Counter()
    for e, frame in enumerate(base_frames(ctx.ontology)):
        extra = [(int(rel), int(a)) for rel, a in read.get(e, [])]
        new = merged(frame, extra, max_degree)
        counts["edges_added"] += len(new) - len(frame)
        counts["entries_enriched"] += int(len(new) > len(frame))
        counts["entries_read"] += int(e in read)
        frames.append(new)
    record.update(dict(counts), seconds=round(time.monotonic() - started, 1),
                  edges_per_entry_base=sum(map(len, base_frames(ctx.ontology))) / max(1, len(frames)),
                  edges_per_entry=sum(map(len, frames)) / max(1, len(frames)))
    return with_frames(ctx.ontology, frames, relation_names=list(ctx.ontology["relation_names"]),
                       atomic_names=list(ctx.ontology["atomic_names"]), enrichment=record)


def write_variant(ontology: dict[str, Any], output: Path, base_root: Path, *, overwrite: bool = False) -> dict[str, Any]:
    """`output/ontology.pt` plus links to every other item of the base corpus folder."""
    output = Path(output)
    if (output / "ontology.pt").exists() and not overwrite:
        raise FileExistsError(f"{output}/ontology.pt exists; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    linked = []
    for item in sorted(Path(base_root).iterdir()):
        if item.name == "ontology.pt":
            continue
        target = output / item.name
        if not target.exists():
            os.symlink(item.resolve(), target)
        linked.append(item.name)
    torch.save(ontology, output / "ontology.pt")
    summary = {"output": str(output), "base": str(base_root), "linked": linked, **ontology.get("enrichment", {})}
    (output / "enrichment.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def queue_arms(variant: str, arms: Sequence[str], *, hosts: Sequence[str], seeds: Sequence[int], priority: float,
               min_enriched: float = 0.0, queue_dir: Path | None = None) -> dict[str, Any]:
    """Plan and queue `arms` on `variant` if its build enriched at least `min_enriched` of the entries (module docstring)."""
    from . import e9_plan
    from . import e9_tracks as tracks
    spec = tracks.track_spec(BASE_TRACK)
    folder = Path(spec.ontology).parent.parent / variant
    unknown = [a for a in arms if a not in e9_plan.ONTOLOGY_ARMS or e9_plan.ONTOLOGY_ARMS[a][0] != variant]
    if unknown:
        raise ValueError(f"{unknown} are not ontology arms on {variant}")
    built = json.loads((folder / "enrichment.json").read_text())
    entries = int(torch.load(folder / "ontology.pt", weights_only=False)["entry_count"])
    share = float(built.get("entries_enriched", 0)) / max(1, entries)
    decision = {"variant": variant, "arms": list(arms), "entries_enriched": built.get("entries_enriched"), "share": share,
                "min_enriched": min_enriched, "queued": []}
    if share >= min_enriched:
        paths = e9_plan.write_stage(BASE_TRACK, hosts=list(hosts), models=list(arms), seeds=list(seeds), track=BASE_TRACK)
        decision["queued"] = e9_plan.queue_jobs(paths, BASE_TRACK, priority, track=BASE_TRACK, queue_dir=queue_dir, level_step=0.000001)
    else:
        decision["skipped"] = f"only {share:.1%} of the entries gained an edge (< {min_enriched:.0%}): no training queued"
    (folder / "queue-decision.json").write_text(json.dumps(decision, indent=2) + "\n")
    return decision


def main(argv: list[str] | None = None) -> None:
    from . import e9_tracks as tracks
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    mesh = sub.add_parser("mesh")
    mesh.add_argument("--output", type=Path, required=True); mesh.add_argument("--max-degree", type=int, default=MAX_DEGREE)
    mesh.add_argument("--overwrite", action="store_true")
    read = sub.add_parser("read")
    read.add_argument("--reader", choices=("teacher", "host"), required=True); read.add_argument("--output", type=Path, required=True)
    read.add_argument("--run", type=Path, default=None); read.add_argument("--cache", type=Path, default=None)
    read.add_argument("--max-usd", type=float, default=60.0); read.add_argument("--batch", type=int, default=8)
    read.add_argument("--model", default="claude-opus-5-5"); read.add_argument("--device", default="cuda")
    read.add_argument("--max-degree", type=int, default=MAX_DEGREE); read.add_argument("--limit", type=int, default=None)
    read.add_argument("--overwrite", action="store_true")
    queue = sub.add_parser("queue")
    queue.add_argument("--variant", required=True); queue.add_argument("--arms", nargs="+", required=True)
    queue.add_argument("--hosts", nargs="+", default=["SmolLM2-135M", "SmolLM2-360M"])
    queue.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3]); queue.add_argument("--priority", type=float, required=True)
    queue.add_argument("--min-enriched", type=float, default=0.0)
    args = parser.parse_args(argv)
    if args.command == "queue":
        print(json.dumps(queue_arms(args.variant, args.arms, hosts=args.hosts, seeds=args.seeds, priority=args.priority,
                                    min_enriched=args.min_enriched), indent=2, default=str))
        return
    spec = tracks.track_spec(BASE_TRACK)
    base_root = Path(spec.ontology).parent
    if args.command == "mesh":
        from ..ontologies.mesh_novel import load_mesh_records
        config = tracks._load_config(spec)["ontology"]
        descriptors, scrs = load_mesh_records(Path(config["path"]).expanduser(), Path(config["supplementary_path"]).expanduser())
        ontology = mesh_enrichment(torch.load(spec.ontology, weights_only=False), descriptors, scrs, max_degree=args.max_degree)
    else:
        ontology = read_enrichment(args.reader, run=args.run, cache=args.cache, max_usd=args.max_usd, batch=args.batch, model=args.model,
                                   device=args.device, max_degree=args.max_degree, limit=args.limit)
    print(json.dumps(write_variant(ontology, args.output, base_root, overwrite=args.overwrite), indent=2, default=str))


if __name__ == "__main__":
    main()
