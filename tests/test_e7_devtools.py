"""WP-E7 dev-tools track (D7.0b / D7.1) on the C6 v1 library: identifier candidates, masked schemas as gold."""

import json
import re
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from vsa_embed.experiments import e7_authoring

C6 = Path("experiments/c6-devtools-benchmark/v1")


def _ok() -> bool:
    try:
        transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
        return (C6 / "frames.json").exists()
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _ok(), reason="gpt2 tokenizer or C6 v1 benchmark missing")


def test_devtools_track_scores_schema_edges(tmp_path: Path, monkeypatch) -> None:
    frames = json.loads((C6 / "frames.json").read_text())
    schema = {e7_authoring.identifier_key(name): frames["frames"][i] for i, name in enumerate(frames["concepts"])}
    atoms, relations = frames["atoms"], frames["relations"]

    def scripted(model, tokenizer, prompts, device, *, samples, temperature, top_p, max_new_tokens, batch, seed):
        out = []
        for prompt in prompts:
            surface = re.findall(r"Concept: (.+)\n", prompt)[-1].strip()
            edges = [f"{relations[r]}: {atoms[a].split(':', 1)[1]}" for r, a in schema.get(surface, [])][:2]
            out.append(["\n".join(edges + ["kind: widget"])] * samples)
        return out, 10, 10

    monkeypatch.setattr(e7_authoring, "HOSTS", {"tiny": "gpt2"})
    monkeypatch.setattr(e7_authoring, "CONSUMER", "tiny")
    monkeypatch.setattr(e7_authoring, "generate_samples", scripted)
    run = tmp_path / "run"
    config = e7_authoring._merge(e7_authoring.DEFAULTS, {
        "devtools_root": str(C6), "data_root": str(tmp_path / "data"), "limit_documents": 400,
        "discovery": {"min_count": 2, "keep": 300, "max_occurrences": 4}, "authoring": {"candidates": 300, "contexts": 2}})
    summary = e7_authoring.prepare("devtools", config, run)
    assert summary["masked"]["concepts"] > 0
    track = e7_authoring.load_track(run)
    hidden = json.loads((tmp_path / "data" / "gold" / "gold.json").read_text())
    view = json.loads((tmp_path / "data" / "visible" / "visible.json").read_text())
    assert not set(view["base"]) & set(hidden["masked_strings"])
    assert all("_" in k or k.isalpha() for k in hidden["masked_strings"])           # identifiers keep underscores
    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=50257, n_positions=1024, n_embd=32, n_layer=1, n_head=2)).eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    e7_authoring.discover(run, "tiny", device=torch.device("cpu"), model=model, tokenizer=tokenizer)
    aset = e7_authoring.link(run)
    assert set(c["surface"] for c in aset["candidates"]) & set(hidden["masked_strings"])
    e7_authoring.author(run, "tiny", device=torch.device("cpu"), model=model, tokenizer=tokenizer)
    e7_authoring.author(run, "hearst", device=torch.device("cpu"))
    quality = e7_authoring.quality(run)
    tiny = quality["authors"]["tiny"]
    assert tiny["precision"]["value"] == pytest.approx(2 / 3, abs=0.05)              # two schema edges + one wrong kind
    assert tiny["relation_accuracy"]["value"] == pytest.approx(1.0)
    hearst = quality["authors"]["hearst"]
    assert hearst["edges"] > 0 and hearst["precision"]["value"] > 0.5               # "`x` is a method of …" → kind: method
