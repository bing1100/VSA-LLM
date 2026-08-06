# export CUDA_VISIBLE_DEVICES="1"

# Run pretraining for the random embeddings for 50 epochs, saving every 5 epochs (8200 steps)

# Can only run trainer.train() without torchrun, or else it tries to put it into NCCL
tsp python3 BERTHA_OOD_Pretraining_save_5_epochs.py --use_random_embeddings --epochs=50
