# export CUDA_VISIBLE_DEVICES="1"

# Run pretraining for the random embeddings

# Can only run trainer.train() without torchrun, or else it tries to put it into NCCL
tsp python3 BERTHA_OOD_Pretraining.py --use_random_embeddings


# Run pretraining for the different VSA configurations

# Snomed_all Words Ignore
tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words --dp_composition=snomed_all --snomed_group=ignore

# Snomed_all Words_rv Ignore
tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words_rv --dp_composition=snomed_all --snomed_group=ignore


# Snomed_all Words Group_vectors
tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words --dp_composition=snomed_all --snomed_group=group_vectors

# Snomed_all Words_rv Group_vectors
tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words_rv --dp_composition=snomed_all --snomed_group=group_vectors
