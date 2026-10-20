"""E13 pre-registration amendment 3 (2026-10-20, from E14 RD-4b): L5b — selective recall (`recall-rel:own`, only the
asked relation's lines on relation items) − none — as a seventh co-primary, its secondaries (against the whole-frame
recall; another stage-2 reader's frames, `recall-rel:own@typeprior`), the verdict, and the configs' stage-3 conditions."""

from pathlib import Path

import pytest
import torch
import yaml

from vsa_embed.experiments import e12_self_query as e12
from vsa_embed.experiments import e13_cycle as e13

torch.set_num_threads(1)


def _reasons(selective: float, whole: float, typeprior: float | None = None, seeds: int = 2) -> list[dict]:
    """Stage-3 summaries of `seeds` seeds: 30 anchors with a relation item and a reverse item each."""
    out = []
    for s in range(seeds):
        names = ["none", "recall:own", "recall-rel:own"] + (["recall-rel:own@typeprior"] if typeprior is not None else [])
        units = {n: {} for n in names}
        anchor_of = {}
        for a in range(30):
            for kind in ("relation", "reverse"):
                item = f"a{a}-{kind}"
                anchor_of[item] = f"a{a}"
                jitter = 0.1 * ((a + s) % 3 - 1)
                units["none"][item] = {"accuracy": 0.4}
                units["recall:own"][item] = {"accuracy": whole + jitter}
                units["recall-rel:own"][item] = {"accuracy": selective + jitter}
                if typeprior is not None:
                    units["recall-rel:own@typeprior"][item] = {"accuracy": typeprior + jitter}
        out.append({"units": units, "anchor_of": anchor_of, "conditions": {}})
    return out


def test_l5b_and_its_secondaries() -> None:
    block = e13._l5b(_reasons(0.8, 0.6, typeprior=0.7), resamples=500)
    assert block["mean"] == pytest.approx(0.4, abs=1e-6) and block["ci_low"] > 0
    assert block["vs_whole_frame"]["mean"] == pytest.approx(0.2, abs=1e-6)
    tp = block["readers"]["typeprior"]
    assert tp["vs_none"]["mean"] == pytest.approx(0.3, abs=1e-6) and tp["vs_read"]["mean"] == pytest.approx(-0.1, abs=1e-6)
    assert e13._l5b(_reasons(0.8, 0.6), resamples=200)["readers"] == {}


def test_l5b_is_in_the_family_and_gets_a_verdict() -> None:
    assert e13.HOLM_FAMILY[-1] == "L5b" and len(e13.HOLM_FAMILY) == 7
    good = {"L5b": {"mean": 0.3, "ci_low": 0.2, "ci_high": 0.4, "p_value": 0.001}}
    assert e13.verdicts(good, e13.family_holm({"L5b": 0.001}), {"holm_alpha": 0.05})["L5b"] is True
    null = {"L5b": {"mean": 0.01, "ci_low": -0.02, "ci_high": 0.04, "p_value": 0.4}}
    assert e13.verdicts(null, e13.family_holm({"L5b": 0.4}), {"holm_alpha": 0.05})["L5b"] is False


def test_recall_rel_is_a_store_condition_and_both_configs_score_it() -> None:
    cond = e12.Condition.parse("recall-rel:own")
    assert cond.kind == "recall-rel" and cond.store == "own" and "recall-rel" in e12.STORE_KINDS
    root = Path(__file__).resolve().parents[1] / "experiments" / "e13-learning-cycle"
    for name in ("t5.yaml", "t7-rood.yaml"):
        reason = yaml.safe_load((root / name).read_text())["reason"]
        names = [c.name for c in e12.parse_conditions(reason["conditions"])]
        assert names[:3] == ["none", "recall:own", "recall-rel:own"] and "definition" in names
        assert reason["secondary_frames"] == ["typeprior"]
