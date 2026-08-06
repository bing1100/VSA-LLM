import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import time

from transformers import AutoTokenizer
from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from medair_models.bertha.modelling_bertha import BerthaForSequenceClassification
from medair_models.bertha.configuration_bertha import BerthaConfig
from torch.optim import Adam
from transformers import get_constant_schedule, Trainer, TrainingArguments
from evaluate import load

import torch
import numpy as np
rng = np.random.default_rng()


parser = argparse.ArgumentParser(description='choose VSA embedding variables')
parser.add_argument('--snomed_group', type=str, choices=['ignore', 'group_zero', 'group_vectors'], default='ignore')
parser.add_argument('--dp_type', type=str, choices=['words', 'words_rv', 'atomic'], default='atomic')
parser.add_argument('--dp_composition', type=str, choices=['icd_name', 'snomed_name', 'snomed_all'], default='snomed_all')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--model_checkpoint', type=str, default=None)
parser.add_argument('--num_trials', type=int, choices=range(1, 15), default=3)

# Using random embeddings overrides VSA options above
use_vsa_parser = parser.add_mutually_exclusive_group(required=False)
use_vsa_parser.add_argument('--use_vsa', dest='use_vsa', action='store_true')
use_vsa_parser.add_argument('--use_random_embeddings', dest='use_vsa', action='store_false')
parser.set_defaults(use_vsa=True)

args, unknown = parser.parse_known_args()


@dataclass
class DataCollatorForMortalityPrediction(DataCollatorMixin):
    """
    Data collator used for mortality. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        eol_threshold (`int`, *optional*):
            Threshold on the number of days left in a patient's life to consider for mortality prediction.
            E.g. `eol_threshold=30`: patients that died within a month of the visit are considered label 1,
            the remainder are label 0. If None, patients with any non-nan days left in life are labelled 1.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
    """

    tokenizer: PreTrainedTokenizerBase
    eol_threshold: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self):
        self.return_tensors = 'pt'

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        if isinstance(examples[0], Mapping):
            batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)
        else:
            batch = {
                "input_ids": _torch_collate_batch(examples, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of)
            }

        batch = self.generate_labels(batch)
        return batch

    def generate_labels(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare binary labels for mortality prediction. Dynamically select a visit and remove all inputs after that visit.
        Based on the number of remaining days in the patient's life and the threshold, set the label.
        """
        new_batch = {}
        for input_ids, token_type_ids, attention_mask, eol_days in zip(*batch.values()):
            num_visits = np.max(token_type_ids.numpy())
            if num_visits == 1:
                cutoff_visit = 1
            else:
                cutoff_visit = rng.integers(1, num_visits, endpoint=True)
            last_visit_ind = np.max(np.nonzero(token_type_ids.numpy() == cutoff_visit))
            cutoff_ind = last_visit_ind + 1
            label = int(eol_days[last_visit_ind] < self.eol_threshold)

            # Remove items after cutoff
            input_ids[cutoff_ind:] = self.tokenizer.pad_token_id
            input_ids[cutoff_ind] = self.tokenizer.sep_token_id  # Put a [SEP] token at the cutoff point

            token_type_ids[cutoff_ind:] = 0  # [SEP] and [PAD] have token_type_id of 0
            attention_mask[cutoff_ind + 1:] = 0  # After [SEP], we mask attention

            new_batch.setdefault("input_ids", []).append(input_ids.tolist())
            new_batch.setdefault("token_type_ids", []).append(token_type_ids.tolist())
            new_batch.setdefault("attention_mask", []).append(attention_mask.tolist())
            new_batch.setdefault("labels", []).append(label)
        
        new_batch = {k: torch.as_tensor(np.array(v)) for k, v in new_batch.items()}
        return new_batch   


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BATCH_SIZE = 128

LEARNING_RATE = args.lr  # By default 1e-4

EPOCHS = args.epochs  # By default 25

DROPOUT_PROB = 0.3

DIM = 768

INITIALIZER_STD = 0.02

WEIGHT_DECAY = 1e-6

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
    for metric in [accuracy_metric, precision_metric, recall_metric]:
        metrics.update(metric.compute(predictions=preds, references=labels))
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
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits, labels)

        preds = torch.round(torch.sigmoid(logits))
        acc = accuracy_metric.compute(predictions=preds.detach().cpu().numpy(), references=labels.cpu().numpy())
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


def hp_space(trial):
        return {
            "num_train_epochs": trial.suggest_int("num_train_epochs", 1, 15, step=1),
            "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
            "per_device_train_batch_size": trial.suggest_categorical("per_device_train_batch_size", [64, 80, 96]),
            "hidden_dropout_prob":  trial.suggest_categorical("hidden_dropout_prob", [0.1, 0.2, 0.3, 0.4, 0.5]),
            "attention_probs_dropout_prob": trial.suggest_categorical("attention_probs_dropout_prob", [0.1, 0.2, 0.3, 0.4, 0.5]),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
        }


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('/root/data/tokenizer-mimic-iv-icd-final')
    dataset_train = Dataset.load_from_disk('/root/data/mimic-iv_data/ood/hf_dataset_finetune_mortality_prediction/train')
    dataset_test = Dataset.load_from_disk('/root/data/mimic-iv_data/ood/hf_dataset_finetune_mortality_prediction/test')

  
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
        **vsa_args )

    seeds = [42, 123, 500, 87, 11, 5, 55, 57, 100, 301]

    def model_init():
        if args.model_checkpoint is None:
            print("Instantiating a new model...")
            model = BerthaForSequenceClassification(config, num_labels=1)
        else:
            print(f"Loading pre-trained model from {args.model_checkpoint}")
            model = BerthaForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=1)
        model.to(device)
        return model


    for trial in range(args.num_trials):
        output_log_folder = '~/output/mortality_finetune_optuna/'
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
            do_eval = True,
            do_train=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
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


        best_run = trainer.hyperparameter_search(
            hp_space = hp_space,
            backend='optuna',
            direction="maximize",
            n_trials=30,
        )

        print(best_run)
