"""Aligned objectives (E15 stage 1c; pre-registration `experiments/e15-hrr-target/preregistration.md`, "Stage 1c").

Stage 1 showed that a model can be trained to *decode* a composed term vector (C5+WR reads 0.48 of a held-out word's
facts at layer 11) while the implicit route — using those facts to answer a question about the word — barely moves
(+0.03). The stage-1 read loss lives in atom space behind an external head; the test is a judgement in token space by
the LM head. The two objectives here are trained in the space of the test:

1. **Route distillation** (`channel.distill`, arms C5RD / C2RD): a *teacher* forward pass of the same model, with the
   current parameters and no gradient (no EMA: the teacher's advantage is its context, not its weights, and a second
   parameter set would have to travel with every checkpoint), sees each linked term's **gold frame rendered as the E14
   explicit-recall lines** (`self_query.RecallWriter`, the rendering of `e14_reading.RecallText.symbolic`) **prepended to
   the window** that contains the term (the simplest faithful variant: one recall block per distinct entry of the
   window, in order of first occurrence, capped at `max_entries` blocks and `max_prefix_tokens` tokens; the vector is
   injected as in C5 at the shifted span positions). The *student* pass is the ordinary C5 pass on the untouched window
   (vector only). Loss: `L_LM + λ_rd · KL(teacher ‖ student)` on the next-token distributions over the K = 8 positions
   after each prefixed term (the `after` stratum of the trainer: positions e..e+7 predict tokens e+1..e+8), temperature
   1. **No leak:** the teacher's input is a separate tensor built here; the student's forward receives the trainer's
   `ids` unchanged (the test decodes both). The teacher runs under `torch.random.fork_rng`, so the student's RNG
   stream equals a run without the block.
2. **Fact verbalisation** (`channel.verbalize`, arms C5FV / C2FV): an auxiliary cloze loss in token space. For each
   linked occurrence, prompts `"<term tokens>: <field label>:"` — the term's own tokens from the window, then a
   template from `FV_TEMPLATES` (a fixed set per relation, every wording with ≥ 2 content words and **disjoint** from
   the E14 question-item templates and the RD-4e router wordings under `e14_reading.near_duplicate`; test
   `test_fv_templates_are_disjoint_from_the_e14_qa_item_templates`) — with the vector injected at the term's last
   subtoken and the filler's *name* (its lexicon text, e.g. "the Rolk Desk", "plans headcount") as the target tokens.
   One prompt per fact of the frame the run injects (`max_facts` = 6 per occurrence, a seeded subset of a larger
   frame; at most `max_prompts` prompts per micro-batch, a seeded subset), scored as separate short right-padded
   sequences in a second forward of the same micro-batch. `L = L_LM + λ_fv · L_cloze` (cross-entropy over the target
   tokens only).

Both blocks are opt-in: a config without them builds nothing here, and the trainer's loop is unchanged (regression
test). Neither adds parameters, so `final.pt` and every evaluation CLI load such runs as they load a C5 run.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from . import self_query as sq

K_AFTER = 8          # the "after the term" stratum of `training.lm` (AFTER_WINDOW)

# Fixed cloze templates per T5 relation: field-style wordings "<term>: <label>:" with at least two content words each,
# none of them an exact or near-duplicate (`e14_reading.near_duplicate`) of an E14 question-item template (property,
# statement, entailment; every T5 item set) or of an RD-4e router wording (`e14_reading.ROUTE_PARAPHRASES`). Relations
# without an entry (other tracks, toy worlds) use `fallback_templates`.
FV_TEMPLATES: dict[str, list[str]] = {
    "is_a": ["{x}: entry type:", "{x}: catalogue category:"],
    "area": ["{x}: home department:", "{x}: org area:"],
    "purpose": ["{x}: stated purpose:", "{x}: purpose statement:"],
    "status": ["{x}: lifecycle stage:", "{x}: current state:"],
    "cadence": ["{x}: repeat frequency:", "{x}: recurrence schedule:"],
    "tier": ["{x}: tier rating:", "{x}: tier band:"],
    "subtype_of": ["{x}: parent type:", "{x}: specialised variant of:"],
    "owned_by": ["{x}: owning team:", "{x}: accountable custodian:"],
    "depends_on": ["{x}: upstream dependency:", "{x}: required dependency:"],
    "part_of": ["{x}: parent item:", "{x}: enclosing container:"],
    "measured_by": ["{x}: health indicator:", "{x}: tracking kpi:"],
    "governed_by": ["{x}: governing policy:", "{x}: governance standard:"],
    "replaces": ["{x}: replaced predecessor:", "{x}: superseded item:"],
    "uses": ["{x}: tooling used:", "{x}: working toolchain:"],
    "produces": ["{x}: main deliverable:", "{x}: generated item:"],
    "reports_to": ["{x}: parent org:", "{x}: reporting line:"],
    "computed_from": ["{x}: input source:", "{x}: computed from source:"],
    "reported_in": ["{x}: reporting publication:", "{x}: published in report:"],
    "approved_by": ["{x}: approval authority:", "{x}: approving body:"],
    "applies_to": ["{x}: applicable scope:", "{x}: covered scope:"],
    "describes": ["{x}: documented topic:", "{x}: reference topic:"],
    "sponsored_by": ["{x}: executive sponsor:", "{x}: sponsoring body:"],
    "produced_by": ["{x}: producing team:", "{x}: authoring group:"],
}


def fallback_templates(relation: str) -> list[str]:
    """Templates of a relation without a hand-written entry: the relation phrase as a field label."""
    phrase = sq.relation_phrase(relation)
    return [f"{{x}}: {phrase} field:", f"{{x}}: recorded {phrase}:"]


def fv_templates(relation: str) -> list[str]:
    return FV_TEMPLATES.get(relation) or fallback_templates(relation)


class FrameTable:
    """Frames per entry from CSR tensors (`offsets`, `relations`, `fillers`): an ontology's or a composer schedule's."""

    def __init__(self, offsets: Any, relations: Any, fillers: Any) -> None:
        self.offsets = np.asarray(torch.as_tensor(offsets).cpu(), dtype=np.int64)
        self.relations = np.asarray(torch.as_tensor(relations).cpu(), dtype=np.int64)
        self.fillers = np.asarray(torch.as_tensor(fillers).cpu(), dtype=np.int64)

    def frame(self, entry: int) -> list[tuple[int, int]]:
        lo, hi = int(self.offsets[entry]), int(self.offsets[entry + 1])
        return list(zip(self.relations[lo:hi].tolist(), self.fillers[lo:hi].tolist()))


@dataclasses.dataclass
class TeacherBatch:
    """The teacher's inputs and the paired positions of one micro-batch (see `AlignedObjectives.teacher_batch`)."""
    ids: Tensor                         # (batch, max prefix + T), right-padded
    spans: dict[str, Tensor]            # the student's spans with positions shifted by each row's prefix length
    rows: Tensor                        # distilled positions: batch row …
    student_cols: Tensor                # … position in the student's window …
    teacher_cols: Tensor                # … and in the teacher's input (prefix length added)
    prefix_lengths: list[int]
    prefix_texts: list[str]
    covered_entries: list[list[int]]
    logprobs: Tensor | None = None      # teacher log-probabilities at the positions, (positions, vocab), float32


@dataclasses.dataclass
class ClozeBatch:
    ids: Tensor                         # (prompts, length), right-padded
    labels: Tensor                      # −100 everywhere but the filler tokens
    spans: dict[str, Tensor]
    prompts: list[dict[str, Any]]       # per prompt: entry, relation, filler, template, text (diagnostics / tests)


class AlignedObjectives:
    """The two objectives of the module docstring, built by `aligned_from_config`. `gold` renders the teacher's recall
    (the ontology's frames); `injected` is the frame the run injects (the composer's schedule for compose channels, the
    ontology's frames for a free table) and gives the verbalised facts."""

    def __init__(self, tokenizer: Any, ontology: dict[str, Any], *, gold: FrameTable, injected: FrameTable, writer: sq.RecallWriter,
                 distill: dict[str, Any] | None, verbalize: dict[str, Any] | None, seed: int, max_positions: int | None = None) -> None:
        self.tokenizer, self.ontology, self.gold, self.injected, self.writer = tokenizer, ontology, gold, injected, writer
        self.seed = int(seed)
        self.max_positions = int(max_positions) if max_positions else None     # the host's absolute position table, if it has one
        self.relation_names = list(ontology["relation_names"])
        d, v = dict(distill or {}), dict(verbalize or {})
        self.distill_weight = float(d.get("weight", 0.0)) if distill is not None else 0.0
        self.k = int(d.get("k", K_AFTER)); self.max_entries = int(d.get("max_entries", 8)); self.max_prefix_tokens = int(d.get("max_prefix_tokens", 384))
        self.verbalize_weight = float(v.get("weight", 0.0)) if verbalize is not None else 0.0
        self.max_facts = int(v.get("max_facts", 6)); self.max_prompts = int(v.get("max_prompts", 64))
        if distill is not None and (self.distill_weight <= 0 or self.k <= 0 or self.max_entries <= 0 or self.max_prefix_tokens <= 0):
            raise ValueError("channel.distill needs weight > 0, k > 0, max_entries > 0 and max_prefix_tokens > 0")
        if verbalize is not None and (self.verbalize_weight <= 0 or self.max_facts <= 0 or self.max_prompts <= 0):
            raise ValueError("channel.verbalize needs weight > 0, max_facts > 0 and max_prompts > 0")
        pad = getattr(tokenizer, "pad_token_id", None)
        self.pad_id = int(pad if pad is not None else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0))
        self._suffix_cache: dict[str, list[int]] = {}
        self._target_cache: dict[str, list[int]] = {}

    @property
    def distill(self) -> bool:
        return self.distill_weight > 0

    @property
    def verbalize(self) -> bool:
        return self.verbalize_weight > 0

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"lexicon": type(self.writer.lexicon).__name__ if self.writer.lexicon is not None else None}
        if self.distill:
            out["distill"] = {"weight": self.distill_weight, "k": self.k, "max_entries": self.max_entries, "max_prefix_tokens": self.max_prefix_tokens,
                              "teacher": "same model, current parameters, no gradient, no EMA", "frames": "gold (ontology)"}
        if self.verbalize:
            out["verbalize"] = {"weight": self.verbalize_weight, "max_facts": self.max_facts, "max_prompts": self.max_prompts,
                                "templates": sum(len(fv_templates(r)) for r in self.relation_names), "frames": "injected"}
        return out

    # -- text ----------------------------------------------------------------------------------------------------------

    def surface(self, row: Tensor, start: int, end: int) -> str:
        """The term as written in the window (tokens start..end inclusive)."""
        return self.tokenizer.decode(row[start:end + 1].tolist(), skip_special_tokens=True).strip()

    def recall_text(self, surface: str, frame: Sequence[tuple[int, int]]) -> str:
        """The E14 explicit-recall rendering of a frame (`e14_reading.RecallText.symbolic`)."""
        return self.writer.render(surface, sq.symbolic_lines(list(frame)))

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    # -- route distillation ------------------------------------------------------------------------------------------

    def teacher_batch(self, ids: Tensor, spans: dict[str, Tensor] | None) -> TeacherBatch | None:
        """Build the teacher's inputs for a micro-batch (None when no span is linked). Per row: one recall block per
        distinct entry in order of first occurrence (stopping at `max_entries` blocks or when the next block would
        exceed `max_prefix_tokens`), joined by blank lines and followed by a blank line, prepended to the window; the
        distilled positions are the K after every occurrence of a prefixed entry (deduplicated), clipped to the window
        like the trainer's `after` stratum."""
        if spans is None or spans["entry"].numel() == 0:
            return None
        batch, length = ids.shape
        cap = self.max_prefix_tokens
        if self.max_positions is not None:          # a host with an absolute position table: the prefix must fit before the window
            cap = min(cap, self.max_positions - length)
            if cap <= 0:
                return None
        cpu = {k: v.detach().cpu() for k, v in spans.items()}
        order = torch.argsort(cpu["inject"], stable=True)
        prefixes: list[list[int]] = [[] for _ in range(batch)]
        texts: list[str] = ["" for _ in range(batch)]
        covered: list[list[int]] = [[] for _ in range(batch)]
        for i in order.tolist():
            b, entry = int(cpu["batch"][i]), int(cpu["entry"][i])
            if entry in covered[b] or len(covered[b]) >= self.max_entries:
                continue
            block = self.recall_text(self.surface(ids[b], int(cpu["start"][i]), int(cpu["end"][i])), self.gold.frame(entry))
            text = (texts[b] + "\n\n" + block) if texts[b] else block
            tokens = self.encode(text + "\n\n")
            if len(tokens) > cap:
                if covered[b]:
                    continue                    # the next block would not fit: keep the prefix as it stands
                tokens = tokens[:cap]           # a single block longer than the cap is truncated
            prefixes[b], texts[b] = tokens, text
            covered[b].append(entry)
        lengths = [len(p) for p in prefixes]
        if not any(lengths):
            return None
        rows, s_cols, t_cols = [], [], []
        seen: set[tuple[int, int]] = set()
        for i in range(cpu["entry"].numel()):
            b, entry, end = int(cpu["batch"][i]), int(cpu["entry"][i]), int(cpu["end"][i])
            if entry not in covered[b]:
                continue
            for pos in range(end, min(length - 1, end + self.k)):
                if (b, pos) in seen:
                    continue
                seen.add((b, pos)); rows.append(b); s_cols.append(pos); t_cols.append(pos + lengths[b])
        if not rows:
            return None
        width = max(lengths) + length
        teacher = torch.full((batch, width), self.pad_id, dtype=ids.dtype, device=ids.device)
        for b in range(batch):
            if lengths[b]:
                teacher[b, :lengths[b]] = torch.tensor(prefixes[b], dtype=ids.dtype, device=ids.device)
            teacher[b, lengths[b]:lengths[b] + length] = ids[b]
        shift = torch.tensor(lengths, dtype=torch.long, device=spans["batch"].device)[spans["batch"]]
        t_spans = {k: (v + shift if k in ("start", "end", "inject") else v) for k, v in spans.items()}
        make = lambda v: torch.tensor(v, dtype=torch.long, device=ids.device)  # noqa: E731
        return TeacherBatch(ids=teacher, spans=t_spans, rows=make(rows), student_cols=make(s_cols), teacher_cols=make(t_cols),
                            prefix_lengths=lengths, prefix_texts=texts, covered_entries=covered)

    @staticmethod
    def _logits(model: Any, hidden: Tensor) -> Tensor:
        head = model.model.get_output_embeddings()
        return F.linear(hidden.to(head.weight.dtype), head.weight, getattr(head, "bias", None)).float()

    def teacher(self, model: Any, ids: Tensor, spans: dict[str, Tensor] | None) -> TeacherBatch | None:
        """The teacher pass: no gradient, the RNG stream left as it was (`fork_rng`), log-probabilities at the paired
        positions kept on the batch."""
        batch = self.teacher_batch(ids, spans)
        if batch is None:
            return None
        devices = [ids.device.index] if ids.is_cuda else []
        with torch.no_grad(), torch.random.fork_rng(devices=devices):
            hidden = model.hidden_states(batch.ids, spans=batch.spans)
            batch.logprobs = F.log_softmax(self._logits(model, hidden[batch.rows, batch.teacher_cols]), dim=-1)
        return batch

    def distill_loss(self, model: Any, student_hidden: Tensor, batch: TeacherBatch | None) -> tuple[Tensor, int]:
        """`KL(teacher ‖ student)` averaged over the distilled positions (temperature 1), and their count."""
        if batch is None or batch.logprobs is None:
            return student_hidden.new_zeros(()).float(), 0
        student = F.log_softmax(self._logits(model, student_hidden[batch.rows, batch.student_cols]), dim=-1)
        kl = (batch.logprobs.exp() * (batch.logprobs - student)).sum(-1).mean()
        return kl, int(batch.rows.numel())

    # -- fact verbalisation ------------------------------------------------------------------------------------------

    def _suffix(self, template: str) -> list[int]:
        if template not in self._suffix_cache:
            self._suffix_cache[template] = self.encode(template.split("{x}", 1)[1])
        return self._suffix_cache[template]

    def _target(self, atom: int) -> list[int]:
        text = " " + self.writer.atom_text(atom)
        if text not in self._target_cache:
            self._target_cache[text] = self.encode(text)
        return self._target_cache[text]

    def cloze_batch(self, ids: Tensor, spans: dict[str, Tensor] | None, *, step: int, micro_step: int) -> ClozeBatch | None:
        """The cloze prompts of a micro-batch (None when no span is linked): per linked occurrence up to `max_facts` facts
        of the injected frame (a seeded subset of a larger frame), one prompt each — the term's tokens from the window,
        the template's suffix, the filler's name as the target — at most `max_prompts` prompts (a seeded subset),
        deterministic per (run seed, step, micro-step)."""
        if spans is None or spans["entry"].numel() == 0:
            return None
        cpu = {k: v.detach().cpu() for k, v in spans.items()}
        rng = np.random.default_rng([self.seed, int(step), int(micro_step), 0x1C])
        prompts: list[dict[str, Any]] = []
        for i in range(cpu["entry"].numel()):
            b, entry = int(cpu["batch"][i]), int(cpu["entry"][i])
            start, end = int(cpu["start"][i]), int(cpu["end"][i])
            frame = self.injected.frame(entry)
            if not frame:
                continue
            if len(frame) > self.max_facts:
                frame = [frame[j] for j in sorted(rng.choice(len(frame), self.max_facts, replace=False).tolist())]
            for r, f in frame:
                templates = fv_templates(self.relation_names[r])
                template = templates[int(rng.integers(len(templates)))]
                prompts.append({"row": b, "start": start, "end": end, "entry": entry, "relation": r, "filler": f, "template": template,
                                "confidence": float(cpu["confidence"][i]) if "confidence" in cpu else 1.0})
        if not prompts:
            return None
        if len(prompts) > self.max_prompts:
            keep = sorted(rng.choice(len(prompts), self.max_prompts, replace=False).tolist())
            prompts = [prompts[j] for j in keep]
        sequences, label_rows = [], []
        for p in prompts:
            term = ids[p["row"], p["start"]:p["end"] + 1].tolist()
            suffix, target = self._suffix(p["template"]), self._target(p["filler"])
            sequences.append(term + suffix + target)
            label_rows.append([-100] * (len(term) + len(suffix)) + target)
            p["term_tokens"], p["target_tokens"] = len(term), len(target)
            p["text"] = self.tokenizer.decode(term + suffix + target, skip_special_tokens=True)
        width = max(len(s) for s in sequences)
        out = torch.full((len(sequences), width), self.pad_id, dtype=ids.dtype, device=ids.device)
        labels = torch.full((len(sequences), width), -100, dtype=torch.long, device=ids.device)
        for j, (s, l) in enumerate(zip(sequences, label_rows)):
            out[j, :len(s)] = torch.tensor(s, dtype=ids.dtype, device=ids.device)
            labels[j, :len(l)] = torch.tensor(l, dtype=torch.long, device=ids.device)
        n = torch.tensor([p["term_tokens"] for p in prompts], dtype=torch.long, device=ids.device)
        c_spans = {"batch": torch.arange(len(prompts), dtype=torch.long, device=ids.device), "start": torch.zeros_like(n), "end": n - 1,
                   "inject": n - 1, "entry": torch.tensor([p["entry"] for p in prompts], dtype=torch.long, device=ids.device), "length": n,
                   "confidence": torch.tensor([p["confidence"] for p in prompts], dtype=torch.float32, device=ids.device)}
        return ClozeBatch(ids=out, labels=labels, spans=c_spans, prompts=prompts)

    def cloze_loss(self, model: Any, batch: ClozeBatch | None, reference: Tensor) -> tuple[Tensor, int]:
        """Cross-entropy over the filler tokens of the prompts (zero when there is none), and the prompt count."""
        if batch is None:
            return reference.new_zeros(()).float(), 0
        return model(batch.ids, spans=batch.spans, labels=batch.labels)["loss"].float(), len(batch.prompts)

    # -- the trainer's entry point -------------------------------------------------------------------------------------

    def losses(self, model: Any, ids: Tensor, spans: dict[str, Tensor] | None, student_hidden: Tensor, teacher: TeacherBatch | None, *,
               step: int, micro_step: int) -> dict[str, Any]:
        """`{"kl", "cloze", "kl_positions", "cloze_prompts"}` of one micro-batch; the weights are applied by the trainer."""
        kl, positions = self.distill_loss(model, student_hidden, teacher) if self.distill else (student_hidden.new_zeros(()).float(), 0)
        cloze, prompts = (self.cloze_loss(model, self.cloze_batch(ids, spans, step=step, micro_step=micro_step), student_hidden)
                          if self.verbalize else (student_hidden.new_zeros(()).float(), 0))
        return {"kl": kl, "cloze": cloze, "kl_positions": positions, "cloze_prompts": prompts}


def recall_writer(config: dict[str, Any], ontology: dict[str, Any]) -> sq.RecallWriter:
    """The E14 recall rendering for the run's track (`e9_track`): the track lexicon's statement templates and filler
    texts when the track is known, else the ontology's names through `self_query.statement_phrase`."""
    lexicon = None
    track = config.get("e9_track")
    if track:
        try:
            from .experiments import e9_tracks as tracks
            lexicon = tracks.lexicon_for(tracks.track_spec(track), ontology)
        except Exception as error:           # noqa: BLE001 — a toy or unknown track renders through the plain phrases
            print(f"aligned_targets: no track lexicon for {track!r} ({error}); rendering with the ontology's names", flush=True)
    return sq.RecallWriter(ontology["relation_names"], ontology["atomic_names"], lexicon, style="statements", confidence=True)


def host_positions(host: Any) -> int | None:
    """The host's absolute position table (GPT-2's `n_positions`); None for rotary hosts (SmolLM2, Qwen) and unknown ones."""
    hf = getattr(host, "config", None)
    n = getattr(hf, "n_positions", None) if hf is not None else None
    return int(n) if n else None


def aligned_from_config(config: dict[str, Any], channel: Any, ontology: dict[str, Any], *, tokenizer_name: str,
                        host: Any = None) -> AlignedObjectives:
    """Build the objectives of `channel.distill` / `channel.verbalize` (opt-in; see the module docstring). `host`: the
    host model, only to read an absolute position limit (`host_positions`)."""
    from transformers import AutoTokenizer
    settings = config["channel"]
    distill, verbalize = settings.get("distill"), settings.get("verbalize")
    if not distill and not verbalize:
        raise ValueError("aligned_from_config needs channel.distill or channel.verbalize")
    if channel is None or channel.mode not in {"compose", "free"}:
        raise ValueError("channel.distill / channel.verbalize support the compose and free channels")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    gold = FrameTable(ontology["offsets"], ontology["relations"], ontology["fillers"])
    composer = getattr(channel, "composer", None)
    injected = FrameTable(composer.schedule.offsets, composer.schedule.relations, composer.schedule.fillers) if composer is not None else gold
    return AlignedObjectives(tokenizer, ontology, gold=gold, injected=injected, writer=recall_writer(config, ontology),
                             distill=dict(distill) if distill else None, verbalize=dict(verbalize) if verbalize else None,
                             seed=int(config.get("seed", 0)), max_positions=host_positions(host))
