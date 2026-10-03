"""Same-site row-source baselines of E9 (WP-PQ1, novelty check §2.5 A-B1/A-B2/A-B4 and §6.5 item 1): per-entry
source tables for the arms C6m / C6d / C6g of `e9_plan` (`channel.mode: source`, `row_sources`).

Each arm adds `g · MLP(source[entry])` at the last subtoken of a linked span — the C5 site and gate — with a trained
MLP projector whose parameters match the C5 channel's dictionary plus projector (the budget `e4_plan.matched_sizes`
gives C2). The source vector of every entry is fixed before training and exists for held-out terms too (so these are
zero-shot competitors for held-out terms, unlike the free table C2):

- `subtoken_mean` (C6m; FVT / Hewitt initialization of a new token): the mean of the *pretrained* host's input-
  embedding rows over the subtokens of the term (each alias of the entry tokenized as it occurs mid-sentence, with a
  leading space; display casing where the concept name normalizes to the alias), averaged over the entry's aliases;
- `definition` (C6d; a definition encoder, Bahdanau et al. 2017 / Token-Distillation style): the frozen pretrained
  host's last-layer hidden state, mean-pooled over the tokens of the entry's *verbalized frame* — every edge of the
  ontology frame written with the track's relation templates and the subject "It" ("It is a process. It is owned by
  the Plorkplum Pod. …"); the term's name is not in the text, so the vector carries exactly the frame's information,
  as C5's composed row does;
- `kge` (C6g; KnowLA / map-tuning style KG embedding): TransE (L1, self-adversarial negative sampling, Sun et al.
  2019) on the track ontology's triples (entry, relation, filler), trained on CPU or GPU; an atomic that names a
  concept (T5 `term:<name>`, T4 `chebi:<id>`, T1 `mesh:<id>`, WordNet `synset:<name>`) is the same entity as that
  concept's single-concept entry, so the graph links terms to each other. Held-out entries' triples are included
  (transductive, like C5's frames at evaluation).

Every table is whitened per dimension over the non-held-out entries and unit-normalized per row
(`row_sources.standardize_rows`), stored in FP16 with the frames digest of the ontology it was built from.

**Text per track** (the verbalization table `verbalized.jsonl.gz` next to the tables records the exact text): T5 —
only the frames (the glossary has structured fields, no definitions; T5 documents are generated from the same
frames); T4 — the frames, plus natural ChEBI definitions for real 3-star entities (`docs/train.jsonl.gz`; not used,
so all three tracks encode the same frame information); T1 — the frames (MeSH scope notes exist in the MeSH XML; not
used); WordNet — the frames (glosses exist; not used). Relations without a template are written generically ("It
contains element As.", "It is computed from …").

    python -m vsa_embed.experiments.e9_rowsource build --track t5 --kind subtoken_mean --host SmolLM2-360M [--device cuda]
    python -m vsa_embed.experiments.e9_rowsource build --track t5 --kind definition --host SmolLM2-360M [--device cuda]
    python -m vsa_embed.experiments.e9_rowsource build --track t5 --kind kge [--device cuda]
    python -m vsa_embed.experiments.e9_rowsource texts --track t5        # the verbalization table only

Tables go to `~/data/vsa-llm/e9/row-sources/<track>-<family>/` (`subtoken_mean-<host>.pt`, `definition-<host>.pt`,
`kge-transe-d256.pt`); an existing table is kept unless `--overwrite`.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..row_sources import frames_digest, load_source_table, save_source_table, standardize_rows
from ..span_channel import AliasTable, normalize_alias

ROOT = Path("~/data/vsa-llm/e9/row-sources").expanduser()
KINDS = ("subtoken_mean", "definition", "kge")
ARM_KINDS = {"C6m": "subtoken_mean", "C6d": "definition", "C6g": "kge"}
KGE_DIMENSION = 256
SUBJECT = "It"


def table_path(track: str, family: str, kind: str, host: str | None = None, *, root: Path | None = None) -> Path:
    """Where `build` writes (and `e9_plan` reads) a row-source table; host-specific for the host-derived kinds."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    folder = Path(root or ROOT) / f"{track}-{family}"
    if kind == "kge":
        return folder / f"kge-transe-d{KGE_DIMENSION}.pt"
    if not host:
        raise ValueError(f"{kind} tables are built per host")
    return folder / f"{kind}-{host}.pt"


def source_dimension(kind: str, host_width: int) -> int:
    return KGE_DIMENSION if kind == "kge" else int(host_width)


def matched_hidden(budget: int, source_width: int, model_width: int) -> int:
    """Hidden width `h` of the projector MLP `source → h → model` whose `h·(source + model)` parameters match `budget`
    (the C5 dictionary-plus-projector budget of `e4_plan.matched_sizes`)."""
    return max(8, round(budget / (source_width + model_width)))


# ---------------------------------------------------------------- surfaces and verbalizations


def entry_surfaces(table: AliasTable, ontology: dict[str, Any]) -> list[list[str]]:
    """Per entry its aliases, in display casing where a concept name normalizes to the alias."""
    names = list(ontology.get("concept_names") or [])
    display: dict[str, str] = {}
    for name in names:
        display.setdefault(normalize_alias(str(name)), str(name))
    out: list[list[str]] = [[] for _ in range(int(ontology["entry_count"]))]
    for alias, entry in sorted(table.alias_to_entry.items()):
        if entry < len(out):
            out[entry].append(display.get(alias, alias))
    return out


def _relation_phrase(relation: str) -> str:
    words = relation.replace("_", " ").split()
    if not words:
        return "is related to"
    return " ".join(words) if words[0].endswith("s") else "is " + " ".join(words)


def _atom_text(lexicon: Any, atom: str) -> str:
    text = lexicon.text(atom) if lexicon is not None else None
    if text:
        return str(text)
    value = atom.split(":", 1)[1] if ":" in atom else atom
    return value.replace("_", " ")


def verbalize_frame(frame: Sequence[tuple[int, int]], ontology: dict[str, Any], lexicon: Any, *, subject: str = SUBJECT) -> str:
    """One sentence per edge: the track template's statement with `{x}` = the subject (else a generic sentence)."""
    relations, atoms = ontology["relation_names"], ontology["atomic_names"]
    templates = getattr(lexicon, "templates", {}) or {}
    sentences = []
    for r, a in frame:
        relation, filler = relations[int(r)], _atom_text(lexicon, atoms[int(a)])
        spec = templates.get(relation)
        if spec is not None and "{y}" in spec.statement:
            filled = lexicon._filler(relation, filler) if hasattr(lexicon, "_filler") else filler
            sentences.append(spec.statement.format(x=subject, y=filled))
        else:
            sentences.append(f"{subject} {_relation_phrase(relation)} {filler}.")
    return " ".join(s[0].upper() + s[1:] if s else s for s in sentences)


def entry_frames(ontology: dict[str, Any]) -> list[list[tuple[int, int]]]:
    offsets = np.asarray(ontology["offsets"]); relations = np.asarray(ontology["relations"]); fillers = np.asarray(ontology["fillers"])
    return [list(zip(relations[offsets[e]:offsets[e + 1]].tolist(), fillers[offsets[e]:offsets[e + 1]].tolist()))
            for e in range(int(ontology["entry_count"]))]


def verbalizations(ontology: dict[str, Any], lexicon: Any) -> list[str]:
    return [verbalize_frame(frame, ontology, lexicon) for frame in entry_frames(ontology)]


def write_verbalizations(path: Path, texts: Sequence[str], surfaces: Sequence[Sequence[str]], heldout: set[int]) -> str:
    """The verbalization table (`entry`, `name`, `heldout`, `text`), gzipped JSONL with mtime 0; returns its sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = "".join(json.dumps({"entry": e, "name": (surfaces[e] or [None])[0], "heldout": e in heldout, "text": t}) + "\n"
                    for e, t in enumerate(texts))
    payload = lines.encode()
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write(payload)
    return hashlib.sha256(payload).hexdigest()


# ---------------------------------------------------------------- the three sources (raw rows, before standardization)


@torch.no_grad()
def subtoken_mean_rows(embedding: Tensor, tokenizer: Any, surfaces: Sequence[Sequence[str]]) -> tuple[Tensor, dict[str, Any]]:
    """FVT rows: per entry, the mean over its aliases of the mean input-embedding row of the alias's subtokens."""
    weight = embedding.detach().float().cpu()
    rows = torch.zeros(len(surfaces), weight.shape[1])
    empty = 0
    for e, names in enumerate(surfaces):
        vectors = []
        for name in names:
            ids = tokenizer.encode(" " + name, add_special_tokens=False)
            if ids:
                vectors.append(weight[torch.tensor(ids)].mean(0))
        if vectors:
            rows[e] = torch.stack(vectors).mean(0)
        else:
            empty += 1
    return rows, {"entries_without_alias": empty}


@torch.no_grad()
def definition_rows(model: nn.Module, tokenizer: Any, texts: Sequence[str], *, device: torch.device | str = "cpu",
                    batch_size: int = 32, max_length: int = 256, layer: int = -1) -> Tensor:
    """Mean over tokens of the host's hidden state at `layer` (default: the last) for each text; the host is frozen."""
    model = model.to(device).eval()
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))         # similar lengths per batch
    out = torch.zeros(len(texts), model.get_input_embeddings().weight.shape[1])
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (tokenizer.eos_token_id or 0)
    for start in range(0, len(order), batch_size):
        index = order[start:start + batch_size]
        encoded = [tokenizer.encode(texts[i], add_special_tokens=False)[:max_length] or [pad] for i in index]
        width = max(map(len, encoded))
        ids = torch.full((len(encoded), width), pad, dtype=torch.long)
        mask = torch.zeros(len(encoded), width, dtype=torch.long)
        for row, tokens in enumerate(encoded):
            ids[row, :len(tokens)] = torch.tensor(tokens); mask[row, :len(tokens)] = 1
        ids, mask = ids.to(device), mask.to(device)
        with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=torch.device(device).type == "cuda"):
            hidden = model(input_ids=ids, attention_mask=mask, output_hidden_states=True).hidden_states[layer].float()
        pooled = (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True)
        out[torch.tensor(index)] = pooled.cpu()
    return out


def kge_graph(ontology: dict[str, Any]) -> dict[str, Any]:
    """TransE triples (head entity, relation, tail entity) of the ontology frames; entries are entities `0 … E−1`, and an
    atomic naming a concept with a single-concept entry is that entry (else a new entity)."""
    entries = int(ontology["entry_count"])
    names = list(ontology.get("concept_names") or [])
    single: dict[str, int] = {}
    for e, concepts in enumerate(ontology.get("entry_concepts") or []):
        if len(concepts) == 1 and int(concepts[0]) < len(names):
            single.setdefault(str(names[int(concepts[0])]), e)
    atomic_entity, next_id, unified = [], entries, 0
    for atom in ontology["atomic_names"]:
        value = atom.split(":", 1)[1] if ":" in atom else atom
        if value in single:
            atomic_entity.append(single[value]); unified += 1
        else:
            atomic_entity.append(next_id); next_id += 1
    offsets = torch.as_tensor(np.asarray(ontology["offsets"]), dtype=torch.long)
    heads = torch.repeat_interleave(torch.arange(entries), offsets[1:] - offsets[:-1])
    tails = torch.as_tensor(atomic_entity, dtype=torch.long)[torch.as_tensor(np.asarray(ontology["fillers"]), dtype=torch.long)]
    relations = torch.as_tensor(np.asarray(ontology["relations"]), dtype=torch.long)
    return {"heads": heads, "relations": relations, "tails": tails, "entities": next_id, "unified_atomics": unified,
            "relation_count": int(ontology["relation_count"])}


def train_transe(graph: dict[str, Any], *, dimension: int = KGE_DIMENSION, gamma: float = 12.0, negatives: int = 64,
                 batch: int = 1024, epochs: float = 100.0, min_steps: int = 3000, max_steps: int = 20000, lr: float = 1e-3,
                 temperature: float = 1.0, seed: int = 0, device: torch.device | str = "cpu",
                 log_every: int = 1000) -> tuple[Tensor, dict[str, Any]]:
    """TransE (score `γ − ‖h + r − t‖₁`) with self-adversarial negative sampling (head and tail corruption alternate);
    entity and relation vectors trained with Adam. Returns (entity vectors, record with the loss and a raw MRR / hits@10
    of tail prediction on up to 2,000 training triples)."""
    generator = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    heads, relations, tails = graph["heads"], graph["relations"], graph["tails"]
    count, entities = heads.numel(), int(graph["entities"])
    bound = (gamma + 2.0) / dimension
    entity = nn.Embedding(entities, dimension).to(device)
    relation = nn.Embedding(int(graph["relation_count"]), dimension).to(device)
    nn.init.uniform_(entity.weight, -bound, bound); nn.init.uniform_(relation.weight, -bound, bound)
    # Dense Adam: a sparse table's gradient coalescing (≈ batch × negatives rows per step) costs more than a dense
    # update of the whole table at these sizes (≤ 10⁵ entities).
    optimizers = [torch.optim.Adam(list(entity.parameters()) + list(relation.parameters()), lr=lr)]
    steps = int(min(max_steps, max(min_steps, math.ceil(epochs * count / batch))))
    h_all, r_all, t_all = heads.to(device), relations.to(device), tails.to(device)
    losses: list[float] = []
    started = time.monotonic()
    for step in range(steps):
        index = torch.randint(0, count, (min(batch, count),), generator=generator).to(device)
        h, r, t = entity(h_all[index]), relation(r_all[index]), entity(t_all[index])
        corrupt = torch.randint(0, entities, (index.numel(), negatives), generator=generator).to(device)
        negative = entity(corrupt)
        positive_score = gamma - (h + r - t).abs().sum(-1)
        if step % 2 == 0:          # corrupt tails
            negative_score = gamma - (h[:, None] + r[:, None] - negative).abs().sum(-1)
        else:                      # corrupt heads
            negative_score = gamma - (negative + r[:, None] - t[:, None]).abs().sum(-1)
        weights = torch.softmax(negative_score * temperature, -1).detach()
        loss = (-F.logsigmoid(positive_score) - (weights * F.logsigmoid(-negative_score)).sum(-1)).mean()
        for optimizer in optimizers:
            optimizer.zero_grad()
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        losses.append(float(loss.detach()))
        if log_every and (step + 1) % log_every == 0:
            print(json.dumps({"step": step + 1, "steps": steps, "loss": float(np.mean(losses[-log_every:])),
                              "seconds": round(time.monotonic() - started, 1)}), flush=True)
    with torch.no_grad():
        sample = torch.randperm(count, generator=generator)[:min(2000, count)].to(device)
        target = entity.weight[t_all[sample]]
        query = entity.weight[h_all[sample]] + relation.weight[r_all[sample]]
        ranks = []
        for part, goal in zip(query.split(256), target.split(256)):
            distance = torch.cdist(part, entity.weight, p=1)
            true = (part - goal).abs().sum(-1, keepdim=True)
            ranks.append(1 + (distance < true).sum(-1))
        ranks = torch.cat(ranks).float()
    record = {"steps": steps, "epochs": steps * batch / count, "triples": count, "entities": entities, "dimension": dimension,
              "gamma": gamma, "negatives": negatives, "batch": batch, "lr": lr, "temperature": temperature, "seed": seed,
              "loss_first_1000": float(np.mean(losses[:1000])), "loss_last_1000": float(np.mean(losses[-1000:])),
              "train_tail_mrr_raw": float((1 / ranks).mean()), "train_tail_hits10_raw": float((ranks <= 10).float().mean()),
              "unified_atomics": int(graph["unified_atomics"]), "seconds": round(time.monotonic() - started, 1)}
    return entity.weight.detach().float().cpu(), record


# ---------------------------------------------------------------- building a table


def build_table(kind: str, *, ontology: dict[str, Any], output: Path, table: AliasTable | None = None, lexicon: Any = None,
                model: nn.Module | None = None, tokenizer: Any = None, device: torch.device | str = "cpu",
                meta: dict[str, Any] | None = None, verbalization_path: Path | None = None, **kge: Any) -> dict[str, Any]:
    """Compute, standardize and save one row-source table; returns its record."""
    heldout = {int(e) for e in ontology["heldout_entries"]}
    reference = [e for e in range(int(ontology["entry_count"])) if e not in heldout]
    record: dict[str, Any] = dict(meta or {})
    if kind == "subtoken_mean":
        surfaces = entry_surfaces(table, ontology)
        raw, info = subtoken_mean_rows(model.get_input_embeddings().weight, tokenizer, surfaces)
        record.update(info, aliases=sum(map(len, surfaces)))
    elif kind == "definition":
        texts = verbalizations(ontology, lexicon)
        if verbalization_path is not None:
            record["verbalizations_sha256"] = write_verbalizations(verbalization_path, texts,
                                                                   entry_surfaces(table, ontology) if table else [[]] * len(texts), heldout)
            record["verbalizations"] = str(verbalization_path)
        raw = definition_rows(model, tokenizer, texts, device=device)
        record.update(layer="last", pooling="mean over tokens", subject=SUBJECT,
                      mean_tokens=float(np.mean([len(tokenizer.encode(t, add_special_tokens=False)) for t in texts[:2000]])))
    elif kind == "kge":
        vectors, info = train_transe(kge_graph(ontology), device=device, **kge)
        raw = vectors[:int(ontology["entry_count"])]
        record.update(kge=info, model="TransE (L1, self-adversarial negative sampling)")
    else:
        raise ValueError(f"kind must be one of {KINDS}")
    rows, stats = standardize_rows(raw, reference)
    record.update(raw_row_norm_mean=float(raw.norm(dim=-1).mean()), standardized="per-dimension z-score over non-held-out "
                  "entries, then unit L2 rows")
    return save_source_table(output, rows, kind=kind, ontology=ontology, meta=record)


def build_for_track(track: str, kind: str, *, host: str | None = None, family: str | None = None, output: Path | None = None,
                    device: str = "cpu", overwrite: bool = False, **kge: Any) -> dict[str, Any]:
    """Build a track's table for a host (the host's pretrained weights from the Hugging Face cache)."""
    from .cpt_plan import HOSTS
    from .e9_tracks import FAMILY_TOKENIZERS, ensure_alias_table, lexicon_for, track_spec
    from ..evaluation.channel_probes import load_alias_table
    if kind != "kge" and not host:
        raise ValueError(f"{kind} needs --host")
    family = family or (HOSTS[host]["corpus"] if host else "smollm2")
    spec = track_spec(track, family)
    path = Path(output) if output else table_path(track, family, kind, host)
    ontology = torch.load(spec.ontology, weights_only=False)
    if path.exists() and not overwrite:
        _, record = load_source_table(path, ontology)
        return {"path": str(path), "existing": True, **record}
    alias_path = ensure_alias_table(spec)
    if alias_path is not None:
        table = load_alias_table(alias_path)
    else:                                      # WordNet: rebuilt from the ontology (channel_probes)
        from ..evaluation.channel_probes import resolve_alias_table
        table, _ = resolve_alias_table(ontology, spec.ontology)
    meta: dict[str, Any] = {"track": track, "family": family, "ontology": str(spec.ontology), "host": host,
                            "alias_table": str(alias_path) if alias_path else None}
    model = tokenizer = None
    if kind != "kge":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        pretrained = HOSTS[host]["pretrained"]
        from ..training.lm import dtype_kwargs
        model = AutoModelForCausalLM.from_pretrained(pretrained, local_files_only=True, **dtype_kwargs(torch.float32))
        tokenizer = AutoTokenizer.from_pretrained(FAMILY_TOKENIZERS[family], local_files_only=True)
        meta.update(pretrained=pretrained, tokenizer=FAMILY_TOKENIZERS[family])
    lexicon = lexicon_for(spec, ontology) if kind == "definition" else None
    record = build_table(kind, ontology=ontology, output=path, table=table, lexicon=lexicon, model=model, tokenizer=tokenizer,
                         device=device, meta=meta, verbalization_path=path.parent / "verbalized.jsonl.gz" if kind == "definition" else None,
                         **kge)
    return {"path": str(path), "existing": False, **record}


def write_track_texts(track: str, family: str = "smollm2", output: Path | None = None) -> dict[str, Any]:
    from .e9_tracks import ensure_alias_table, lexicon_for, track_spec
    from ..evaluation.channel_probes import load_alias_table
    spec = track_spec(track, family)
    ontology = torch.load(spec.ontology, weights_only=False)
    alias_path = ensure_alias_table(spec)
    table = load_alias_table(alias_path) if alias_path else None
    texts = verbalizations(ontology, lexicon_for(spec, ontology))
    path = Path(output) if output else ROOT / f"{track}-{family}" / "verbalized.jsonl.gz"
    digest = write_verbalizations(path, texts, entry_surfaces(table, ontology) if table else [[]] * len(texts),
                                  {int(e) for e in ontology["heldout_entries"]})
    return {"path": str(path), "entries": len(texts), "sha256": digest, "frames_sha256": frames_digest(ontology), "example": texts[0]}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build", help="build one row-source table")
    build.add_argument("--track", required=True); build.add_argument("--kind", required=True, choices=KINDS)
    build.add_argument("--host", default=None, help="host (cpt_plan.HOSTS) for subtoken_mean / definition")
    build.add_argument("--family", default=None, help="tokenizer family (default: the host's; smollm2 for kge)")
    build.add_argument("--output", type=Path, default=None); build.add_argument("--device", default="cpu")
    build.add_argument("--overwrite", action="store_true")
    build.add_argument("--kge-max-steps", type=int, default=20000); build.add_argument("--kge-epochs", type=float, default=100.0)
    texts = sub.add_parser("texts", help="write the verbalization table of a track")
    texts.add_argument("--track", required=True); texts.add_argument("--family", default="smollm2")
    texts.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.command == "texts":
        print(json.dumps(write_track_texts(args.track, args.family, args.output), indent=2))
        return
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    extra = {"max_steps": args.kge_max_steps, "epochs": args.kge_epochs} if args.kind == "kge" else {}
    result = build_for_track(args.track, args.kind, host=args.host, family=args.family, output=args.output, device=device,
                             overwrite=args.overwrite, **extra)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
