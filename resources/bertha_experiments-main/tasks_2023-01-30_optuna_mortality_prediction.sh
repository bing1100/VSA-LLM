# Run HP Tuning on no vsa and one vsa config

# Snomed_all Words_rv Ignore
tsp python3 BERTHA_Tuning_Optuna_Mortality_Prediction.py --dp_type=words_rv --dp_composition=snomed_all --snomed_group=ignore \
    --model_checkpoint="/root/output/ood_pretraining/2023_01_21-132558_ignore_words_rv_snomed_all/output/checkpoint-41000"

# No VSA
tsp python3 BERTHA_Tuning_Optuna_Mortality_Prediction.py --use_random_embeddings \
    --model_checkpoint="/root/output/ood_pretraining/2023_01_20-220015_ignore_atomic_snomed_all/output/checkpoint-41000"