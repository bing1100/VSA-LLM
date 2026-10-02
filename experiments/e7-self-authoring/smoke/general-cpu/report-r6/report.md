# R6 self-authoring (general track)

Status labels: every number below comes from the run folders listed in `summary.json`; LLM-judged plausibility is an estimate (experiments §0.12).

## D7.0 setup

Masked concepts: 26,368 (sha256 `eb2bf92cde740a0b…`); visible aliases 117,680.

## D7.1 authoring quality (masked gold concepts in the authoring set)

| author | kind | concepts | edges | precision | recall | F1 | relation accuracy |
|---|---|---:|---:|---|---|---:|---|
| Qwen2.5-0.5B | host | 0 | 0 | — | — | — | — |
| SmolLM2-135M | host | 0 | 0 | — | — | — | — |
| SmolLM2-360M | host | 0 | 0 | — | — | — | — |
| direct-SmolLM2-135M | direct | 0 | 0 | — | — | — | — |
| hearst | hearst | 0 | 0 | — | — | — | — |
| random | random | 0 | 0 | — | — | — | — |
| teacher | teacher | 0 | 0 | — | — | — | — |

Discovery recall of the 18 discoverable masked concepts:

| host | @100 | @250 | @500 | @1000 | @2500 | @5000 |
|---|---|---|---|---|---|---|
| Qwen2.5-0.5B | 0.056 [0.010, 0.258] (n=18) | 0.056 [0.010, 0.258] (n=18) | 0.056 [0.010, 0.258] (n=18) | 0.056 [0.010, 0.258] (n=18) | 0.056 [0.010, 0.258] (n=18) | 0.056 [0.010, 0.258] (n=18) |
| SmolLM2-135M | 0.056 [0.010, 0.258] (n=18) | 0.056 [0.010, 0.258] (n=18) | 0.056 [0.010, 0.258] (n=18) | 0.056 [0.010, 0.258] (n=18) | 0.056 [0.010, 0.258] (n=18) | 0.056 [0.010, 0.258] (n=18) |
| SmolLM2-360M | 0.167 [0.058, 0.392] (n=18) | 0.333 [0.163, 0.563] (n=18) | 0.389 [0.203, 0.614] (n=18) | 0.389 [0.203, 0.614] (n=18) | 0.389 [0.203, 0.614] (n=18) | 0.389 [0.203, 0.614] (n=18) |

## D7.2 round 0 → 1

| seed | author | proposed edges | accepted | concepts accepted | median U |
|---|---|---:|---:|---:|---:|
| 1 | SmolLM2-360M | 24 | 0 | 0 | — |
| 1 | teacher | 14 | 0 | 0 | — |

Authored-edge precision vs the hidden gold (masked concepts, pooled over seeds):

| author | verified | unverified | verified − unverified [95% CI] |
|---|---|---|---|
| SmolLM2-360M | — (0) | — (0) | — |
| teacher | — (0) | — (0) | — |

Loss gaps at the final evaluation (treatment − control, nats/token; negative = treatment better):

| stratum | comparison | Δ [95% CI] | relative | p (Holm) | seeds | method |
|---|---|---|---|---|---|---|
| ref_masked | self − cm | -0.0005 [-0.0030, 0.0021] | -0.02% | 1.000 | 1 | window bootstrap |
| ref_masked | self − random | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_masked | self − entigraph | -0.0017 [-0.0061, 0.0019] | -0.06% | 1.000 | 1 | window bootstrap |
| ref_masked | self − notes | -0.0005 [-0.0030, 0.0021] | -0.02% | 1.000 | 1 | window bootstrap |
| ref_masked | self − spa | -0.0013 [-0.0048, 0.0018] | -0.04% | 1.000 | 1 | window bootstrap |
| ref_masked | self − verbal | -0.0005 [-0.0030, 0.0021] | -0.02% | 1.000 | 1 | window bootstrap |
| ref_masked | self − selfnv | -0.0000 [-0.0000, 0.0000] | -0.00% | 1.000 | 1 | window bootstrap |
| ref_masked | self − selfrand | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_masked | teacher − cm | -0.0005 [-0.0030, 0.0021] | -0.02% | 1.000 | 1 | window bootstrap |
| ref_masked | gold − cm | -0.0010 [-0.0036, 0.0021] | -0.03% | 1.000 | 1 | window bootstrap |
| ref_masked | entigraph − cm | +0.0012 [-0.0019, 0.0052] | +0.04% | 1.000 | 1 | window bootstrap |
| ref_masked | notes − cm | +0.0000 [-0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_masked | spa − cm | +0.0008 [-0.0013, 0.0033] | +0.03% | 1.000 | 1 | window bootstrap |
| ref_masked | verbal − cm | -0.0000 [-0.0000, 0.0000] | -0.00% | 1.000 | 1 | window bootstrap |
| ref_masked | selfnv − cm | -0.0005 [-0.0030, 0.0021] | -0.02% | 1.000 | 1 | window bootstrap |
| ref_masked | random − cm | -0.0005 [-0.0030, 0.0021] | -0.02% | 1.000 | 1 | window bootstrap |
| ref_masked | teacher − self | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_base_after | self − cm | -0.0015 [-0.0049, 0.0012] | -0.05% | 1.000 | 1 | window bootstrap |
| ref_base_after | self − random | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_base_after | self − entigraph | -0.0021 [-0.0095, 0.0024] | -0.07% | 1.000 | 1 | window bootstrap |
| ref_base_after | self − notes | -0.0015 [-0.0049, 0.0012] | -0.05% | 1.000 | 1 | window bootstrap |
| ref_base_after | self − spa | +0.0003 [-0.0061, 0.0059] | +0.01% | 1.000 | 1 | window bootstrap |
| ref_base_after | self − verbal | -0.0015 [-0.0049, 0.0012] | -0.05% | 1.000 | 1 | window bootstrap |
| ref_base_after | self − selfnv | +0.0000 [-0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_base_after | self − selfrand | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_base_after | teacher − cm | -0.0015 [-0.0049, 0.0012] | -0.05% | 1.000 | 1 | window bootstrap |
| ref_base_after | gold − cm | -0.0016 [-0.0049, 0.0012] | -0.05% | 1.000 | 1 | window bootstrap |
| ref_base_after | entigraph − cm | +0.0006 [-0.0030, 0.0059] | +0.02% | 1.000 | 1 | window bootstrap |
| ref_base_after | notes − cm | +0.0000 [-0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_base_after | spa − cm | -0.0018 [-0.0069, 0.0031] | -0.06% | 1.000 | 1 | window bootstrap |
| ref_base_after | verbal − cm | +0.0000 [-0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_base_after | selfnv − cm | -0.0015 [-0.0049, 0.0012] | -0.05% | 1.000 | 1 | window bootstrap |
| ref_base_after | random − cm | -0.0015 [-0.0049, 0.0012] | -0.05% | 1.000 | 1 | window bootstrap |
| ref_base_after | teacher − self | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_unlinked | self − cm | +0.0004 [-0.0002, 0.0010] | +0.01% | 1.000 | 1 | window bootstrap |
| ref_unlinked | self − random | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_unlinked | self − entigraph | -0.0000 [-0.0011, 0.0010] | -0.00% | 1.000 | 1 | window bootstrap |
| ref_unlinked | self − notes | +0.0004 [-0.0002, 0.0010] | +0.01% | 1.000 | 1 | window bootstrap |
| ref_unlinked | self − spa | +0.0001 [-0.0009, 0.0013] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_unlinked | self − verbal | +0.0004 [-0.0002, 0.0010] | +0.01% | 1.000 | 1 | window bootstrap |
| ref_unlinked | self − selfnv | +0.0000 [0.0000, 0.0000] | +0.00% | 0.340 | 1 | window bootstrap |
| ref_unlinked | self − selfrand | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_unlinked | teacher − cm | +0.0004 [-0.0002, 0.0010] | +0.01% | 1.000 | 1 | window bootstrap |
| ref_unlinked | gold − cm | +0.0004 [-0.0002, 0.0010] | +0.01% | 1.000 | 1 | window bootstrap |
| ref_unlinked | entigraph − cm | +0.0004 [-0.0003, 0.0012] | +0.01% | 1.000 | 1 | window bootstrap |
| ref_unlinked | notes − cm | +0.0000 [-0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_unlinked | spa − cm | +0.0002 [-0.0005, 0.0009] | +0.01% | 1.000 | 1 | window bootstrap |
| ref_unlinked | verbal − cm | +0.0000 [-0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| ref_unlinked | selfnv − cm | +0.0003 [-0.0002, 0.0010] | +0.01% | 1.000 | 1 | window bootstrap |
| ref_unlinked | random − cm | +0.0004 [-0.0002, 0.0010] | +0.01% | 1.000 | 1 | window bootstrap |
| ref_unlinked | teacher − self | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| all | self − cm | -0.0003 [-0.0013, 0.0007] | -0.01% | 1.000 | 1 | window bootstrap |
| all | self − random | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| all | self − entigraph | -0.0008 [-0.0031, 0.0007] | -0.03% | 1.000 | 1 | window bootstrap |
| all | self − notes | -0.0003 [-0.0013, 0.0007] | -0.01% | 1.000 | 1 | window bootstrap |
| all | self − spa | +0.0000 [-0.0020, 0.0019] | +0.00% | 1.000 | 1 | window bootstrap |
| all | self − verbal | -0.0003 [-0.0013, 0.0007] | -0.01% | 1.000 | 1 | window bootstrap |
| all | self − selfnv | +0.0000 [-0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| all | self − selfrand | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| all | teacher − cm | -0.0003 [-0.0013, 0.0007] | -0.01% | 1.000 | 1 | window bootstrap |
| all | gold − cm | -0.0004 [-0.0014, 0.0007] | -0.01% | 1.000 | 1 | window bootstrap |
| all | entigraph − cm | +0.0005 [-0.0007, 0.0022] | +0.02% | 1.000 | 1 | window bootstrap |
| all | notes − cm | +0.0000 [-0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| all | spa − cm | -0.0003 [-0.0020, 0.0012] | -0.01% | 1.000 | 1 | window bootstrap |
| all | verbal − cm | +0.0000 [-0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |
| all | selfnv − cm | -0.0003 [-0.0013, 0.0007] | -0.01% | 1.000 | 1 | window bootstrap |
| all | random − cm | -0.0003 [-0.0013, 0.0007] | -0.01% | 1.000 | 1 | window bootstrap |
| all | teacher − self | +0.0000 [0.0000, 0.0000] | +0.00% | 1.000 | 1 | window bootstrap |

### G5 items

```
{
 "self_beats_cm": {
  "ref_authored_new_self": null,
  "ref_masked": false
 },
 "self_beats_random": {
  "ref_authored_new_self": null,
  "ref_masked": false
 },
 "locality": true,
 "verification_improves_precision": null,
 "h_g_controls": {
  "self_beats_entigraph": {
   "ref_authored_new_self": null,
   "ref_masked": false
  },
  "self_beats_notes": {
   "ref_authored_new_self": null,
   "ref_masked": false
  },
  "self_beats_spa": {
   "ref_authored_new_self": null,
   "ref_masked": false
  },
  "self_beats_verbal": {
   "ref_authored_new_self": null,
   "ref_masked": false
  }
 },
 "seeds": [
  1
 ]
}
```

## Compute accounting (outside training)

| source | stage | parameters | forward | prompt | generated | PFLOP | s |
|---|---|---:|---:|---:|---:|---:|---:|
| discovery/Qwen2.5-0.5B.json | discovery | 4.94e+08 | 29,223 | 0 | 0 | 0.029 | 102 |
| discovery/SmolLM2-135M.json | discovery | 1.35e+08 | 29,042 | 0 | 0 | 0.008 | 31 |
| discovery/SmolLM2-360M.json | discovery | 3.62e+08 | 48,365 | 0 | 0 | 0.035 | 125 |
| proposals/Qwen2.5-0.5B.json | authoring | 4.94e+08 | 0 | 5,828 | 1,152 | 0.007 | 35 |
| proposals/SmolLM2-135M.json | authoring | 1.35e+08 | 0 | 6,370 | 1,152 | 0.002 | 14 |
| proposals/SmolLM2-360M.json | authoring | 3.62e+08 | 0 | 6,370 | 1,152 | 0.005 | 32 |
| proposals/direct-SmolLM2-135M.json | direct | 1.35e+08 | 0 | 907 | 288 | 0.000 | — |
| round1/entigraph.json | entigraph | 1.35e+08 | 0 | 2,110 | 384 | 0.001 | 5 |
| round1/notes.json | notes | 1.35e+08 | 0 | 2,989 | 384 | 0.001 | 6 |
| round1/s1/verify-SmolLM2-360M.json | verification | 1.35e+08 | 0 | 0 | 0 | 0.000 | 0 |
| round1/s1/verify-notes.json | verification | 1.35e+08 | 0 | 0 | 0 | 0.000 | — |
| round1/s1/verify-teacher.json | verification | 1.35e+08 | 0 | 0 | 0 | 0.000 | 0 |
| round1/spa.json | spa | 1.35e+08 | 0 | 1,860 | 288 | 0.001 | 5 |

Compute-matched control, seed 1: 2,048 + 75,163 = 77,211 training tokens (stages authoring, discovery, verification).

Unstructured notes through the same utility filter (`notes` control):

| seed | notes | kept | tokens | median U |
|---|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | — |

Training tokens of the compute-matched and synthetic-text controls (side PFLOP = discovery, authoring, generation and verification they used):

| seed | condition | training tokens | side PFLOP | synthetic share |
|---|---|---:|---:|---:|
| 1 | cm | 77,211 | 0.040 | — |
| 1 | entigraph | 77,211 | 0.036 | 0.10 |
| 1 | notes | 77,211 | 0.036 | — |
| 1 | spa | 77,211 | 0.001 | 0.10 |
| 1 | verbal | 77,211 | 0.040 | — |

## D7.3 cross-authoring (50M from scratch)

| stratum | comparison | Δ [95% CI] | relative | p (Holm) |
|---|---|---|---|---|
| ref_masked | authored − none | +0.0024 [-0.0103, 0.0152] | +0.03% | 1.000 |
| ref_masked | curated − none | +0.0002 [-0.0088, 0.0084] | +0.00% | 1.000 |
| ref_masked | authored − curated | +0.0022 [-0.0064, 0.0110] | +0.02% | 1.000 |
| ref_unlinked | authored − none | +0.0069 [0.0038, 0.0098] | +0.07% | 0.003 |
| ref_unlinked | curated − none | +0.0078 [0.0047, 0.0115] | +0.08% | 0.003 |
| ref_unlinked | authored − curated | -0.0009 [-0.0035, 0.0015] | -0.01% | 0.426 |
| all | authored − none | +0.0064 [0.0033, 0.0093] | +0.07% | 0.003 |
| all | curated − none | +0.0070 [0.0042, 0.0101] | +0.07% | 0.003 |
| all | authored − curated | -0.0006 [-0.0024, 0.0010] | -0.01% | 0.454 |
| after | authored − none | +0.0419 [—, —] | +0.44% | — |
| after | curated − none | +0.0048 [0.0003, 0.0094] | +0.05% | 0.033 |
| after | authored − curated | +0.0371 [—, —] | +0.39% | — |
| after_rare | authored − none | +0.0395 [—, —] | +0.41% | — |
| after_rare | curated − none | +0.0041 [-0.0006, 0.0094] | +0.04% | 0.098 |
| after_rare | authored − curated | +0.0354 [—, —] | +0.37% | — |

## Related work this round must be read against

Self-improvement from self-authored knowledge is not new in general; the E7 claim is the narrowed one of the 2025–26 re-check (typed frames, held-out-utility verification, non-textual channel, against the controls below).

- Evontree — Tu et al., *Evontree: Ontology Rule-Guided Self-Evolution of Large Language Models*, arXiv:2510.26683 (self-authored ontology knowledge, rule-verified, fed back by self-distillation).
- Montessori-Instruct — Li, Yu & Xiong, ICLR 2025, arXiv:2410.14208; OptimSyn — Fan et al., arXiv:2604.00536 (synthetic data scored by its first-order effect on held-out loss; cf. TracIn, LESS, In-Run Data Shapley arXiv:2406.11011).
- GraphGen — Chen et al., arXiv:2505.20416; Synthesize-on-Graph (SoG) — Ma et al., arXiv:2505.00979 (KG- and loss-targeted synthetic data for training).
- Active Reading — Lin et al., *Learning Facts at Scale with Active Reading*, arXiv:2508.09494 (ICLR 2026); SEAL — Zweiger et al., *Self-Adapting Language Models*, NeurIPS 2025, arXiv:2506.10943 (a model's own study data / self-edits beat frontier-teacher data) → the `notes` control.
- SPA — Tang et al., *SPA: A Simple but Tough-to-Beat Baseline for Knowledge Injection*, arXiv:2603.22213 → the `spa` control.
- The `verbal` control gives the same verified edges as text (no channel), isolating the channel.
- Collapse: Kazdan et al., ICML 2025 (*Collapse or Thrive*); Yi et al., arXiv:2510.16657 (verifier-guided retraining converges to the verifier's knowledge centre) — report gains per round and audit on data disjoint from validation.
