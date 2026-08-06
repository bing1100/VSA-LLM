# export CUDA_VISIBLE_DEVICES="1"

# For some reason, the hyperparameters chosen with optuna seem to not be peforming well on this new dataset.
# Re-try some optimal combinations based on optuna results with VSA

# Can only run trainer.train() without torchrun, or else it tries to put it into NCCL

# Run pretraining for the different VSA configurations

# Snomed_all Words Ignore
tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words --dp_composition=snomed_all --snomed_group=ignore --lr 0.0000921 --epochs 30

# Snomed_all Words_rv Ignore
tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words_rv --dp_composition=snomed_all --snomed_group=ignore --lr 0.000115331 --epochs 30


# # Snomed_all Words Group_vectors
# tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words --dp_composition=snomed_all --snomed_group=group_vectors

# # Snomed_all Words_rv Group_vectors
# tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words_rv --dp_composition=snomed_all --snomed_group=group_vectors
