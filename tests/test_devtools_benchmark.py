from pathlib import Path

from vsa_embed.benchmarks.devtools import build, documents, frames, generate_library, leakage_audit


def test_library_is_deterministic_and_names_are_unique_and_forbidden_words_avoided() -> None:
    a = generate_library(seed=3, forbidden={"bank", "flora"})
    b = generate_library(seed=3, forbidden={"bank", "flora"})
    assert a == b
    names = [s["name"] for s in a["symbols"]]
    assert len(names) == len(set(names))
    assert not any(part in {"bank", "flora"} for n in names for part in n.lower().split("_"))


def test_heldout_symbols_never_reach_training_documents(tmp_path: Path) -> None:
    library = generate_library(seed=4)
    docs = documents(library, seed=4)
    assert leakage_audit(library, docs)["leaked_into_train"] == []
    summary = build(tmp_path, seed=4)
    assert summary["leaked_into_train"] == 0 and summary["heldout_symbols"] > 0
    f = frames(library)
    assert len(f["frames"]) == len(library["symbols"]) and all(f["frames"])
    assert all(0 <= a < len(f["atoms"]) for frame in f["frames"] for _, a in frame)
