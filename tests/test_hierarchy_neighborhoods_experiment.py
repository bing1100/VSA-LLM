from vsa_embed.experiments.hierarchy_neighborhoods import summarize


def test_hierarchy_summary_gates_hrr_against_matched_and_shuffled_controls() -> None:
    methods = ["basis_offset_residual_hrr", "basis_offset_diagonal_control",
               "shuffled_basis_offset_residual_hrr"]
    rows = []
    values = {
        "basis_offset_residual_hrr": (0.40, 0.50),
        "basis_offset_diagonal_control": (0.37, 0.47),
        "shuffled_basis_offset_residual_hrr": (0.20, 0.30),
    }
    for seed in (1, 2):
        for method, (mrr, recall) in values.items():
            rows.append({"seed": seed, "method": method, "query_mrr": mrr,
                         "set_recall_at_10": recall, "set_hit_at_10": 0.8,
                         "set_mass_nll": 1.0, "target_centroid_cosine": 0.2,
                         "mean_targets_per_query": 2.0})
    config = {"methods": methods, "seeds": [1, 2], "acceptance": {
        "candidate": methods[0], "non_hrr_control": methods[1],
        "shuffled_control": methods[2], "min_mrr_gain_over_non_hrr": 0.01,
        "min_recall_gain_over_non_hrr": 0.01, "min_paired_wins": 2,
        "promotion_eligible": False,
    }}
    summary = summarize(rows, config)
    assert summary["development_criteria_passed"]
    assert not summary["gate_passed"]