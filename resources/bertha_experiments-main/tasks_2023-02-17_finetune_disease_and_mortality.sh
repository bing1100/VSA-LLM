# Run Finetuning on no vsa and 4 vsa configs
# Balance positive class by factor of 11 (1077/12184)


# mortality

# ignore words_rv snomed_all
tsp python3 BERTHA_OOD_Finetuning_Mortality_Prediction.py --snomed_group=ignore --dp_type=words_rv --dp_composition=snomed_all \
    --positive_class_weight=11 --num_trials=5 --epochs=15 \
    --model_checkpoint="/root/output/ood_pretraining/2023_02_21-230222_ignore_words_rv_snomed_all/output/checkpoint-49200"

# group_vectors words_rv snomed_all
tsp python3 BERTHA_OOD_Finetuning_Mortality_Prediction.py --snomed_group=group_vectors --dp_type=words_rv --dp_composition=snomed_all \
    --positive_class_weight=11 --num_trials=5 --epochs=15 \
    --model_checkpoint="/root/output/ood_pretraining/2023_02_22-205826_group_vectors_words_rv_snomed_all/output/checkpoint-49200"