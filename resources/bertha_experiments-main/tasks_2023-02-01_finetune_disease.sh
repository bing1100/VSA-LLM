# Run Finetuning on no vsa and 4 vsa configs
# Balance positive class by factor of 11 (1077/12184)

# No VSA
tsp python3 BERTHA_OOD_Finetuning_Disease_Prediction.py --use_random_embeddings \
    --num_trials=5 --epochs=20 \
    --model_checkpoint="/root/output/ood_pretraining/2023_01_20-220015_ignore_atomic_snomed_all/output/checkpoint-41000"

# ignore words snomed_all
tsp python3 BERTHA_OOD_Finetuning_Disease_Prediction.py --snomed_group=ignore --dp_type=words --dp_composition=snomed_all \
    --num_trials=5 --epochs=20 \
    --model_checkpoint="/root/output/ood_pretraining/2023_01_21-035149_ignore_words_snomed_all/output/checkpoint-41000"

# ignore words_rv snomed_all
tsp python3 BERTHA_OOD_Finetuning_Disease_Prediction.py --snomed_group=ignore --dp_type=words_rv --dp_composition=snomed_all \
    --num_trials=5 --epochs=20 \
    --model_checkpoint="/root/output/ood_pretraining/2023_01_21-132558_ignore_words_rv_snomed_all/output/checkpoint-41000"

# group_vectors words snomed_all
tsp python3 BERTHA_OOD_Finetuning_Disease_Prediction.py --snomed_group=group_vectors --dp_type=words --dp_composition=snomed_all \
    --num_trials=5 --epochs=20 \
    --model_checkpoint="/root/output/ood_pretraining/2023_01_21-230521_group_vectors_words_snomed_all/output/checkpoint-41000"

# group_vectors words_rv snomed_all
tsp python3 BERTHA_OOD_Finetuning_Disease_Prediction.py --snomed_group=group_vectors --dp_type=words_rv --dp_composition=snomed_all \
    --num_trials=5 --epochs=20 \
    --model_checkpoint="/root/output/ood_pretraining/2023_01_22-084101_group_vectors_words_rv_snomed_all/output/checkpoint-41000"
