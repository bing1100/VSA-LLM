# Run HP tuning on the warmup part of the VSA pre-training
# Most of the gains come in the initial learning, optimize this part

# 2023-02-23: Make eval delay 3000 steps so we train for longer before attempting to prune runs

# ignores words snomed_all
tsp python3 BERTHA_Tuning_Optuna_Pretrain_Warmup.py --snomed_group=ignore --dp_type=words --dp_composition=snomed_all