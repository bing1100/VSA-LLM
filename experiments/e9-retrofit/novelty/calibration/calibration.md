# E9 novelty screen — proxy calibration (decision 55)

Novelty = (C0′ − P0) / P0 and gain = (C5 − C0′) / C0′ after terms, seed 1, mean over the strata after_heldout, after_unseen, after_rare_seen, after_len3plus. Proxies come from the P0 run alone (or from no run: subtokens, general-text frequency).

| track | host | novelty | gain | inside loss | after / unlinked | inside nats / span | subtokens / span | general / domain rate | general-absent share |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| t5 | SmolLM2-360M | -76.1% | -7.5% | 3.377 | 0.887 | 15.31 | 5.53 | 0.000 | 1.00 |
| t4 | SmolLM2-360M | -48.3% | -1.2% | 1.637 | 1.099 | 5.34 | 5.09 | 0.036 | 0.73 |
| t1 | SmolLM2-360M | -6.6% | -0.0% | 0.763 | 0.991 | 1.26 | 2.77 | 0.383 | 0.38 |
| wordnet | SmolLM2-360M | -0.7% | +0.0% | 0.909 | 0.899 | 1.07 | 2.25 | 0.909 | 0.26 |
| t5 | SmolLM2-135M | -74.9% | -5.1% | 3.380 | 0.885 | 15.32 | 5.53 | 0.000 | 1.00 |
| t4 | SmolLM2-135M | -43.7% | -1.9% | 1.850 | 1.066 | 6.03 | 5.09 | 0.036 | 0.73 |
| t1 | SmolLM2-135M | -6.6% | -0.1% | 0.936 | 0.993 | 1.55 | 2.77 | 0.383 | 0.38 |
| wordnet | SmolLM2-135M | -0.3% | -0.0% | 1.043 | 0.902 | 1.23 | 2.25 | 0.909 | 0.26 |

Rank agreement with novelty magnitude and with gain magnitude over the tracks (τ = Kendall τ-b, ρ = Spearman, p = exact one-sided permutation p of τ; with 4 tracks a perfect ordering has p = 1/24 ≈ 0.042). Gains below 0.1% in magnitude are tied at 0 (T1 and WordNet: n.s. in R9), so the best possible gain ordering has three levels (τ-b 0.91, p = 0.083).

| host | proxy | τ novelty | ρ novelty | p | τ gain | ρ gain | p |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-360M | inside_loss | +0.67 | +0.80 | 0.167 | +0.91 | +0.95 | 0.083 |
| SmolLM2-360M | after_unlinked | +0.00 | -0.20 | 0.625 | -0.18 | -0.32 | 0.750 |
| SmolLM2-360M | inside_nats_per_span | +1.00 | +1.00 | 0.042 | +0.91 | +0.95 | 0.083 |
| SmolLM2-360M | subtokens_per_span | +1.00 | +1.00 | 0.042 | +0.91 | +0.95 | 0.083 |
| SmolLM2-360M | general_to_domain | +1.00 | +1.00 | 0.042 | +0.91 | +0.95 | 0.083 |
| SmolLM2-360M | general_absent_share | +1.00 | +1.00 | 0.042 | +0.91 | +0.95 | 0.083 |
| SmolLM2-135M | inside_loss | +0.67 | +0.80 | 0.167 | +0.91 | +0.95 | 0.083 |
| SmolLM2-135M | after_unlinked | +0.00 | -0.20 | 0.625 | -0.18 | -0.32 | 0.750 |
| SmolLM2-135M | inside_nats_per_span | +1.00 | +1.00 | 0.042 | +0.91 | +0.95 | 0.083 |
| SmolLM2-135M | subtokens_per_span | +1.00 | +1.00 | 0.042 | +0.91 | +0.95 | 0.083 |
| SmolLM2-135M | general_to_domain | +1.00 | +1.00 | 0.042 | +0.91 | +0.95 | 0.083 |
| SmolLM2-135M | general_absent_share | +1.00 | +1.00 | 0.042 | +0.91 | +0.95 | 0.083 |
