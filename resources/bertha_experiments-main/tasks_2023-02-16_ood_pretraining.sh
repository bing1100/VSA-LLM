export CUDA_VISIBLE_DEVICES="0"

# Run pretraining for the random embeddings

# Run pretraining for the different VSA configurations

# Snomed_all Words Ignore
# tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words --dp_composition=snomed_all --snomed_group=ignore --num_trials=1

# Snomed_all Words_rv Ignore
# tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words_rv --dp_composition=snomed_all --snomed_group=ignore --num_trials=1 

# Snomed_all Words Group_vectors
# tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words --dp_composition=snomed_all --snomed_group=group_vectors --num_trials=1

# Snomed_all Words_rv Group_vectors
tsp python3 BERTHA_OOD_Pretraining.py --dp_type=words_rv --dp_composition=snomed_all --snomed_group=group_vectors --num_trials=1

