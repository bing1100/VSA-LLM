# WP-probe validation: channel-probe suite on untouched hosts

Plain `ModelAdapter`, bf16 weights, final layer, batch 32 (as B8). Full LAMBADA (5,153), WiC dev, Raganato ALL (7,253), BLESS noun pairs, HyperLex (2,616).

## B8 probes

| Host | Metric | This suite | b8_committed | equal | b8_rerun_this_machine | equal |
|---|---|---:|---:|---|---:|---|
| HuggingFaceTB/SmolLM2-135M | lambada_accuracy | 0.4277 | 0.4277 | yes | 0.4277 | yes |
| HuggingFaceTB/SmolLM2-135M | lambada_word_loss | 2.4830 | 2.4830 | no | 2.4830 | yes |
| HuggingFaceTB/SmolLM2-135M | n | 5153 | 5153 | yes | 5153 | yes |
| HuggingFaceTB/SmolLM2-135M | wic_prompt_accuracy | 0.5956 | 0.5956 | yes | 0.5956 | yes |
| HuggingFaceTB/SmolLM2-135M | wic_probe_accuracy | 0.5768 | 0.5768 | yes | 0.5768 | yes |
| HuggingFaceTB/SmolLM2-135M | wic_majority | 0.5000 | 0.5000 | yes | 0.5000 | yes |
| HuggingFaceTB/SmolLM2-135M | n_dev | 638 | 638 | yes | 638 | yes |
| HuggingFaceTB/SmolLM2-135M | card660.spearman | 0.2111 | 0.2145 | no | 0.2111 | yes |
| HuggingFaceTB/SmolLM2-135M | rare_words.spearman | 0.2620 | 0.2619 | no | 0.2620 | yes |
| gpt2 | lambada_accuracy | 0.3097 | 0.3097 | yes | 0.3097 | yes |
| gpt2 | lambada_word_loss | 3.1761 | 3.1761 | no | 3.1761 | yes |
| gpt2 | n | 5153 | 5153 | yes | 5153 | yes |
| gpt2 | wic_prompt_accuracy | 0.5517 | 0.5517 | yes | 0.5517 | yes |
| gpt2 | wic_probe_accuracy | 0.5627 | 0.5627 | yes | 0.5627 | yes |
| gpt2 | wic_majority | 0.5000 | 0.5000 | yes | 0.5000 | yes |
| gpt2 | n_dev | 638 | 638 | yes | 638 | yes |
| gpt2 | card660.spearman | 0.0832 | 0.0838 | no | 0.0832 | yes |
| gpt2 | rare_words.spearman | 0.1075 | 0.1075 | no | 0.1075 | yes |

`b8_committed` was produced on the replaced machine (faulty RAM); R0 found that its CARD-660/RW Spearman and LAMBADA loss reproduce only to within 0.0035 even with B8's own code. `b8_rerun_this_machine` is B8's code re-run here (R0); equality with it is bit-exact.

## New probes

| Host | WSD probe F1 | centroid F1 | MFS F1 | WN1 F1 | probe F1 (ambiguous) | MFS F1 (ambiguous) | BLESS probe acc | macro-F1 | relatum-only acc | pair probe acc | pair macro-F1 | majority | prompt MAP | prompt AUC | HyperLex cosine ρ | prompt ρ | probe ρ (test) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| HuggingFaceTB/SmolLM2-135M | 0.7055 | 0.6913 | 0.6546 | 0.6519 | 0.6203 | 0.5446 | 0.8905 | 0.8869 | 0.8942 | 0.8812 | 0.8661 | 0.4640 | 0.3520 | 0.7408 | 0.0118 | 0.3887 | 0.3737 |
| gpt2 | 0.7010 | 0.6840 | 0.6546 | 0.6519 | 0.6136 | 0.5446 | 0.9023 | 0.8993 | 0.9012 | 0.8726 | 0.8539 | 0.4640 | 0.2641 | 0.6932 | 0.0139 | 0.3168 | 0.2754 |

## Time

| Host | total s | peak GiB | per probe (s) |
|---|---:|---:|---|
| HuggingFaceTB/SmolLM2-135M | 103.5 | 1.5 | lambada 17.62, wic 8.67, card660 0.88, rare_words 1.99, wsd 43.86, bless 22.96, hyperlex 5.01 |
| gpt2 | 72.8 | 1.93 | lambada 15.36, wic 5.57, card660 0.34, rare_words 0.74, wsd 35.71, bless 11.35, hyperlex 3.39 |

## Protocol

- States: final layer, last subtoken of the word; isolated words are read after "The word" (as B8).
- WSD: SemCor → Raganato ALL; per lemma#POS multinomial L2 logistic (l2 = 0.01, 200 L-BFGS steps; 400 gives the same predictions on GPT-2) on standardized states, at most 100 sampled SemCor occurrences per lemma#POS (seed 0); one seen sense → that sense; unseen lemma#POS → WordNet first sense (as MFS). Centroid = nearest sense centroid (cosine). MFS = most frequent SemCor sense, ties by WordNet order.
- BLESS: noun pairs of {hyper, coord, mero, random-n}; 4-way probe on [h_x; h_y], concept-disjoint split (30% of the 200 concepts, seed 0); relatum-only control (h_y); pair probe on [|h_x − h_y|; h_x ⊙ h_y]; prompting = PMI of "The x is a kind of y" vs "The thing is a kind of y" (per-concept AP of hypernyms; pooled AUC hyper vs rest).
- HyperLex: cosine and PMI (verbs: "To x is a way to y" vs "To do …") on all 2,616 pairs; ridge on [h_x; h_y] trained on the lexical split's train part, α from {1, …, 10⁴} on its dev part, Spearman on its test part.
- Spearman for the new probes and all subset/paired statistics uses average ranks for ties; CARD-660/RW `spearman` keeps B8's ordinal-rank definition.

