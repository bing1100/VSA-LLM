import os
import argparse
import time
import logging
import sys
import torch
from torch import nn

from datasets import Dataset
from transformers import AutoTokenizer
from medair_models.bertha.modelling_bertha import BerthaForSequenceClassification

if __name__ == "__main__":

    # arg parser
        
    parser = argparse.ArgumentParser(description='choose VSA embedding variables')
    parser.add_argument('--snomed_group', type=str, choices=['ignore', 'group_zero', 'group_vectors'], default='ignore')
    parser.add_argument('--dp_type', type=str, choices=['words', 'words_rv', 'atomic'], default='atomic')
    parser.add_argument('--dp_composition', type=str, choices=['icd_name', 'snomed_name', 'snomed_all'], default='snomed_all')
    parser.add_argument('--lr', type=float, default=2.5e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    # parser.add_argument('--model_checkpoint', type=str, default=os.environ['SM_MODEL_CHKPT'])
    parser.add_argument('--positive_class_weight', type=float, default=1)
    parser.add_argument('--num_trials', type=int, choices=range(1, 15), default=1)

    # Using random embeddings overrides VSA options above
    use_vsa_parser = parser.add_mutually_exclusive_group(required=False)
    use_vsa_parser.add_argument('--use_vsa', dest='use_vsa', action='store_true')
    use_vsa_parser.add_argument('--use_random_embeddings', dest='use_vsa', action='store_false')
    parser.set_defaults(use_vsa=False)

    # Arguments that may be needed for AWS
    parser.add_argument('--input_model_dir', type=str, default=os.environ['SM_CHANNEL_MODEL'])
    parser.add_argument('--output_model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    # parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument('--tokenizer_path', type=str, default=os.environ['SM_CHANNEL_TOKENIZER'])

    args, unknown = parser.parse_known_args() 

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # tokenizer and datasets
    logger.info(f"{args.tokenizer_path}")
    logger.info(f"{args.train}")
    logger.info(f"{args.test}")

    logger.info(f"{os.listdir(args.tokenizer_path)}")
    logger.info(f"{os.listdir(args.train)}")
    logger.info(f"{os.listdir(args.test)}")
    logger.info(f"{os.listdir(args.input_model_dir)}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset_train = Dataset.load_from_disk(args.train) 
    dataset_test = Dataset.load_from_disk(args.test)

    logger.info(f" loaded dataset_train length is: {len(dataset_train)}")
    logger.info(f" loaded dataset_test length is: {len(dataset_test)}")

    logger.info(f"Loading a model...")
    

    logger.info(f"Saving a model...")

    model = nn.Sequential(
        nn.Linear(100, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )
    #path = f"/opt/ml/checkpoints/mlp_{time.strftime('%Y-%m-%d_%H%M%S')}.pt"
    path = args.output_model_dir + f"/mlp_{time.strftime('%Y-%m-%d_%H%M%S')}.pt"
    torch.save(model, path)
    logger.info(f"Saved model to {path}")
    
    # Confirm batch size
    logger.info(f"Batch size set to {args.batch_size}")

    #model = torch.load(args.model_dir + "/")
    logger.info("Attempting to load model...")
    logger.info(os.listdir(args.input_model_dir))
    model = BerthaForSequenceClassification.from_pretrained(args.input_model_dir, num_labels=1)
    logger.info(f"Successfully loaded model from {args.input_model_dir}")