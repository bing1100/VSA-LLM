import os
import sys
from datasets import Dataset
from medair_models.bertha.modelling_bertha import BerthaForSequenceClassification

from transformers import AutoTokenizer
from evaluate import load

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import argparse
import logging
import tarfile
import json

from aws_collators import DataCollatorForMortalityPrediction

rng = np.random.default_rng()

if __name__ == "__main__":
    # arg parser
    parser = argparse.ArgumentParser(description='choose VSA embedding variables')

    # Arguments that may be needed for AWS
    parser.add_argument('--input_model_dir', type=str, default=os.environ['SM_CHANNEL_MODEL'])
    parser.add_argument('--output_model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    # parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument('--tokenizer_path', type=str, default=os.environ['SM_CHANNEL_TOKENIZER'])
    parser.add_argument('--vsa_parser_path', type=str, default=None)
    parser.add_argument('--num_labels', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mortality_days_threshold', type=int, default=30)

    args, unknown = parser.parse_known_args() 

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset_test = Dataset.load_from_disk(args.test)
    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=DataCollatorForMortalityPrediction(
            tokenizer,
            eol_threshold=args.mortality_days_threshold,
            inference_metadata=True
        )
    )
    logger.info(f"Loaded dataset_test length is: {len(dataset_test)}")

    # Load model
    with tarfile.open(args.input_model_dir + "/model.tar.gz") as tar:
        tar.extractall(args.input_model_dir)
    logger.info(f"{os.listdir(args.input_model_dir)}")
    model = BerthaForSequenceClassification.from_pretrained(args.input_model_dir)
    model.to(device)
    logger.info(f"Loaded model from {args.input_model_dir}")
    
    # Run inference
    logger.info("Running inference...")
    results = {}
    model.eval()

    for i, batch in tqdm(enumerate(dataloader_test)):
        with torch.no_grad():
            input_data = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            outputs = model(**input_data)
            logits = outputs.logits
            probs = torch.sigmoid(logits).flatten()
            preds = torch.round(probs)
            is_correct = (preds == input_data["labels"]).long()  # Convert bool to int
        for k in ['patient_id', 'visit_id', 'visit_date']:
            results.setdefault(k, []).extend(batch[k])
        for k, v in zip(
            ["label", "prediction", "correct_prediction", "probability_of_mortality"],
            [input_data["labels"], preds, is_correct, probs]
        ):
            results.setdefault(k, []).extend(v.cpu().tolist())

    logger.info("Finished inference, saving results...")
    # Save results
    df = pd.DataFrame(results)
    csv_path = f"{args.output_data_dir}/mortality_inference.csv"
    df.to_csv(csv_path)
    logger.info(f"Saved results to {csv_path}")

    metric_types = ["accuracy", "precision", "recall", "f1"]
    metrics = {}
    for metric in metric_types:
        metric = load(metric)
        metrics.update(metric.compute(predictions=df["prediction"], references=df["label"]))
    
    metrics_path = f"{args.output_data_dir}/metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f)
    logger.info(f"Saved metrics to {metrics_path}")
