import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import time

from datasets import Dataset
from medair_models.bertha.modelling_bertha import BerthaForSequenceClassification
from medair_models.bertha.configuration_bertha import BerthaConfig
from torch.optim import Adam
from transformers import get_constant_schedule, Trainer, TrainingArguments, AutoTokenizer
from evaluate import load
from collators import DataCollatorForMortalityPrediction

import torch
import numpy as np


parser = argparse.ArgumentParser(description='choose VSA embedding variables')
parser.add_argument('--snomed_group', type=str, choices=['ignore', 'group_zero', 'group_vectors'], default='ignore')
parser.add_argument('--dp_type', type=str, choices=['words', 'words_rv', 'atomic'], default='atomic')
parser.add_argument('--dp_composition', type=str, choices=['icd_name', 'snomed_name', 'snomed_all'], default='snomed_all')
parser.add_argument('--lr', type=float, default=2.5e-5)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--model_checkpoint', type=str, default=None)
parser.add_argument('--positive_class_weight', type=float, default=1)
parser.add_argument('--num_trials', type=int, choices=range(1, 15), default=3)

# Using random embeddings overrides VSA options above
use_vsa_parser = parser.add_mutually_exclusive_group(required=False)
use_vsa_parser.add_argument('--use_vsa', dest='use_vsa', action='store_true')
use_vsa_parser.add_argument('--use_random_embeddings', dest='use_vsa', action='store_false')
parser.set_defaults(use_vsa=True)

args, unknown = parser.parse_known_args() 


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BATCH_SIZE = 80

LEARNING_RATE = args.lr  # By default 2.5e-5

EPOCHS = args.epochs  # By default 5

DROPOUT_PROB = 0.3

DIM = 768

INITIALIZER_STD = 0.02

WEIGHT_DECAY = 4e-6

accuracy_metric = load("accuracy")
recall_metric = load("recall")
precision_metric = load("precision")

def compute_metrics(eval_pred):
    """
    Calculate metrics for model predictions during evaluation.

    Parameters
    ----------
    eval_pred: 
        Predictions from the model including the logits or preds, and the corresponding labels

    Returns
    -------
    metrics: dict {str: float}
        Metrics dictionary
    """
    preds, labels = eval_pred
    preds = preds[0]
    metrics = {}
    sample_weight = [args.positive_class_weight if label == 1 else 1 for label in labels]
    for metric in [accuracy_metric, precision_metric, recall_metric]:
        metrics.update(metric.compute(predictions=preds, references=labels, sample_weight=sample_weight))
    return metrics


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function for calculating the Cross Entropy loss given set of inputs and the training accuracy

         Parameters
         ----------
         model: BERT-type model

         inputs: array-like
            Inputs to the model for computing loss and accuracy

        return_outputs: bool
            Whether to return outputs in a tuple with loss. Defaults to False

        Returns
        --------
        loss: float
            Calculated loss for the given inputs.

        Yields
        ------
        acc: float
            Training accuracy on the inputs given
        """
        labels = inputs.get("labels").float()
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits").flatten()
        # compute loss
        device = logits.device
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([args.positive_class_weight]).to(device))
        loss = loss_fct(logits, labels)

        preds = torch.round(torch.sigmoid(logits))
        sample_weight = [args.positive_class_weight if label == 1 else 1 for label in labels.cpu().numpy()]
        acc = accuracy_metric.compute(predictions=preds.detach().cpu().numpy(), references=labels.cpu().numpy(), sample_weight=sample_weight)
        self.log(acc)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        """
        Custom function for setting the optimizer and the scheduler for the learning rate

        Parameters
        ----------

        num_training_steps: int
            Total number of steps expected for training
    
        """
        self.optimizer = Adam(params=self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        self.lr_scheduler = get_constant_schedule(
            optimizer=self.optimizer)


def preprocess_logits_for_metrics(logits, labels):
    """ 
    Reduce the dimensions of the predictions before they are passed to the compute_metrics function to prevent the out of memory errors

    Parameters
    ----------
    logits: array-like
        predictions
    labels: array-like
        corresponding token labels
    Returns
    -------
    pred_ids: array-like
        predictions with reduced dimensions
    labels: array-like
        corresponding token labels
    """
    pred_ids = torch.round(torch.sigmoid(logits))
    labels = labels.float()
    return pred_ids, labels


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('/root/data/tokenizer-mimic-iv-icd-final')
    dataset_train = Dataset.load_from_disk('/root/data/mimic-iv_data/rood/hf_dataset_finetune_mortality_prediction_train/train')
    dataset_test = Dataset.load_from_disk('/root/data/mimic-iv_data/rood/hf_dataset_finetune_mortality_prediction_train/test')

  
    tokenization_params = {
        'max_length': 128,
        'truncation': True,
        'padding': 'max_length',
        'is_split_into_words': True,
        'return_special_tokens_mask': True
    }

    vsa_args = {
        "dp_composition": args.dp_composition,
        "dp_type": args.dp_type,
        "snomed_groups": args.snomed_group
    }

    config = BerthaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dropout_prob=DROPOUT_PROB,
        attention_probs_dropout_prob=DROPOUT_PROB,
        max_position_embeddings=tokenization_params['max_length'],
        type_vocab_size=100,
        initializer_range=INITIALIZER_STD,
        layer_norm_position="post",
        position_embedding_type="absolute",
        apply_ffn_to_embeddings=None,
        freeze_word_embeddings=False,
        normalize_word_embeddings=False,
        path_to_vsa_parser_data="/root/data/mimic-iv_data/icd_to_vsa_data.pkl",
        tokenizer_vocab=tokenizer.vocab,
        use_vsa=args.use_vsa,
        num_labels=1,
        **vsa_args )

    def model_init():
        if args.model_checkpoint is None:
            print("Instantiating a new model...")
            model = BerthaForSequenceClassification(config)
        else:
            print(f"Loading pre-trained model from {args.model_checkpoint}")
            model = BerthaForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=1)
        model.to(device)
        return model

    seeds = [42, 123, 500, 87, 11, 5, 55, 57, 100, 301, 32, 77, 97, 117, 3]

    for trial in range(args.num_trials):
        output_log_folder = '~/output/mortality_finetune_experiments_rood/updated/'
        timestr = time.strftime("%Y_%m_%d-%H%M%S_")

        if args.use_vsa:
            output_dir = output_log_folder + timestr + f'{args.snomed_group}_{args.dp_type}_{args.dp_composition}/output'
            logging_dir = output_log_folder + timestr + f'{args.snomed_group}_{args.dp_type}_{args.dp_composition}/logs'
        else:
            output_dir = output_log_folder + timestr + "no_vsa/output"
            logging_dir = output_log_folder + timestr + "no_vsa/logs"
        

        trainingArgs = TrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir,
            seed=seeds[trial],
            do_eval=True,
            do_train=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            metric_for_best_model="eval_accuracy",
            save_total_limit=2,
            load_best_model_at_end=True,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            per_device_train_batch_size=BATCH_SIZE, 
            per_device_eval_batch_size=8,
            eval_accumulation_steps=20,
            num_train_epochs=EPOCHS,
            logging_strategy="steps",
            logging_steps=32, 
            gradient_accumulation_steps=1,
            fp16=False,
            bf16=False,
            tf32=True,
            ddp_find_unused_parameters=False
        )


        trainer = CustomTrainer(
            model_init=model_init,
            args=trainingArgs,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        data_collator=DataCollatorForMortalityPrediction(tokenizer, eol_threshold=180),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        trainer.train()
