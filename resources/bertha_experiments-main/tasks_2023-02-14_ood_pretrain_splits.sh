# Train different splits of the training data. Same test data.

# 25% split
tsp python3 BERTHA_OOD_Pretraining_Dataset_Splits.py --use_random_embeddings --num_trials=1 \
    --train_ds_path="/root/data/mimic-iv_data/ood/hf_dataset_pretrain_splits/ds_25"

# 50% split
tsp python3 BERTHA_OOD_Pretraining_Dataset_Splits.py --use_random_embeddings --num_trials=1 \
    --train_ds_path="/root/data/mimic-iv_data/ood/hf_dataset_pretrain_splits/ds_50"

# 75% split
tsp python3 BERTHA_OOD_Pretraining_Dataset_Splits.py --use_random_embeddings --num_trials=1 \
    --train_ds_path="/root/data/mimic-iv_data/ood/hf_dataset_pretrain_splits/ds_75"

# 100% split
tsp python3 BERTHA_OOD_Pretraining_Dataset_Splits.py --use_random_embeddings --num_trials=1 \
    --train_ds_path="/root/data/mimic-iv_data/ood/hf_dataset_pretrain_splits/ds_100"