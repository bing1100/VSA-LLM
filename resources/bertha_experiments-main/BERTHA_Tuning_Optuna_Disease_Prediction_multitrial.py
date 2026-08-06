import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from torch.optim import Adam
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import medair_models
from medair_models.bertha.modelling_bertha import BerthaForSequenceClassification
from medair_models.bertha.configuration_bertha import BerthaConfig

from transformers import get_constant_schedule
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
from evaluate import load
import pickle

from math import floor
from random import randrange
from torch import LongTensor

import os
import argparse
import time

parser = argparse.ArgumentParser(description='choose VSA embedding variables')
parser.add_argument('--snomed_group', type=str, choices=['ignore', 'group_zero', 'group_vectors'], default='ignore')
parser.add_argument('--dp_type', type=str, choices=['words', 'words_rv', 'atomic'], default='atomic')
parser.add_argument('--dp_composition', type=str, choices=['icd_name', 'snomed_name', 'snomed_all'], default='snomed_all')
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--num_trials', type=int, choices=range(1, 15), default=3)

# Using random embeddings overrides VSA options above
use_vsa_parser = parser.add_mutually_exclusive_group(required=False)
use_vsa_parser.add_argument('--use_vsa', dest='use_vsa', action='store_true')
use_vsa_parser.add_argument('--use_random_embeddings', dest='use_vsa', action='store_false')
parser.set_defaults(use_vsa=True)

args, unknown = parser.parse_known_args()

BATCH_SIZE = 128

LEARNING_RATE = args.lr  # By default 1e-4

EPOCHS = args.epochs  # By default 25

DROPOUT_PROB = 0.3

DIM = 768

INITIALIZER_STD = 0.02

WEIGHT_DECAY = 1e-6


def compute_metrics(eval_pred):
    """
    Calculate metrics for model predictions during evaluation.

    Parameters
    ----------
    eval_pred: 
        Predictions from the model including the logits or preds, and the corresponding labels

    Returns
    -------
    metric.compute: float
        Computes accuracy
    
    """
    preds, labels = eval_pred
    preds = preds[0]
    active_mask = labels.flatten() != -100
    active_labels = labels.flatten()[active_mask]
    active_preds = preds.flatten()[active_mask]
    return metric.compute(predictions=active_preds, references=active_labels)
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
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        #logits = logits.permute(0, 2, 1)
        # compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        preds = torch.argmax(logits, axis=-1)
        # Get active labels
        active_mask = labels.view(-1) != -100
        active_labels = torch.masked_select(labels.view(-1), active_mask)
        active_preds = torch.masked_select(preds.view(-1), active_mask)
        acc = metric.compute(predictions=active_preds.cpu().numpy(), references=active_labels.cpu().numpy())
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
        self.optimizer = Adam(params=self.model.parameters(), lr=self.args.learning_rate, weight_decay=WEIGHT_DECAY)
        self.lr_scheduler = get_constant_schedule(
            optimizer=self.optimizer)

from transformers.data.data_collator import DataCollatorMixin, _torch_collate_batch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


@dataclass
class DataCollatorForNextDiseasePrediction(DataCollatorMixin):
    """
    Data collator used for next disease prediction. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
    """

    tokenizer: PreTrainedTokenizerBase
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

        batch["input_ids"], batch['token_type_ids'], batch['attention_mask'], batch["labels"] = \
            self.torch_mask_tokens(
            batch["input_ids"], batch["token_type_ids"], batch['attention_mask']
        )
        return batch

    def torch_mask_tokens(self, inputs: Any, input_types: Any, attention_mask: Any) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels for next disease prediction: MASK all of last visit. Set labelss to only have first disease code of last visit
        """
        import torch

        # get the first disease from the last visit and set labels to only include that one token
        next_disease_indices = torch.argmax(input_types, dim=1)
        labels = torch.stack([t[i:i+1] for i, t in zip(next_disease_indices, inputs)])

        # turn the next visit into just a mask token then a sep token
        post_next_disease_indices = torch.stack([torch.arange(0, inputs.shape[1]) >= i for i in next_disease_indices])
        inputs[post_next_disease_indices] = self.tokenizer.pad_token_id
        input_types[post_next_disease_indices] = self.tokenizer.pad_token_type_id
        attention_mask[post_next_disease_indices] = 0
        for i, j in enumerate(next_disease_indices):
            inputs[i, j] = self.tokenizer.sep_token_id
            attention_mask[i, j] = 1

        return inputs, input_types, attention_mask, labels

@dataclass
class DataCollatorForICD10(DataCollatorMixin):
    """
    Data collator. Accepts batches of ids for patients (input_ids, token_type_ids, attention_mask). Selects random visit number, masks tokens after (and including) that visit. 
    Produces ICD-10 label.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
    """
    
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

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

        batch = self.batch_data_collator(batch)
        return batch

    def torch_mask_tokens(self, inputs: Any, input_types: Any, attention_mask: Any) -> Tuple[Any, Any, Any, Any]:
        """
        Prepare masked tokens inputs/labels for next disease prediction: MASK all of last visit. Set labelss to only have first disease code of last visit
        """
        import torch

        # get the first disease from the last visit and set labels to only include that one token
        next_disease_indices = torch.argmax(input_types, dim=1)
        labels = torch.stack([t[i:i+1] for i, t in zip(next_disease_indices, inputs)])

        # turn the next visit into just a mask token then a sep token
        post_next_disease_indices = torch.stack([torch.arange(0, inputs.shape[1]) >= i for i in next_disease_indices])
        inputs[post_next_disease_indices] = self.tokenizer.pad_token_id
        input_types[post_next_disease_indices] = self.tokenizer.pad_token_type_id
        attention_mask[post_next_disease_indices] = 0
        for i, j in enumerate(next_disease_indices):
            inputs[i, j] = self.tokenizer.sep_token_id
            attention_mask[i, j] = 1

        return inputs, input_types, attention_mask, labels
    
    def lookup_ICD_10_chapter(self, code):
        """
        Accepts ICD-9 or ICD-10 codes, returns ICD-10 chapter.
        """
        if code[-2:] == '-9':
            if code[0].isalpha():
                chapter = '20' 
                return chapter
            code = code[:-2]    # 2765-9 -> 2765
            if len(code) == 5:
                code = str(code)[:-2] + "." + str(code)[-2:]    # 27939 -> 279.39  
            if len(code) == 4:
                code = str(code)[:-1] + "." + str(code)[-1]     # 0032 -> 003.2  
            while code[0] == "0":  
                code = code[1:]  # 0032 -> 3.2
            code = float(code)
            code_dict = {'1': [1,139], '2': [140,239], '3': [240,279], '4': [280,289], '5': [290,319], '6': [320,359],'7': [360,379],
                        '8': [380,389],'9': [390,459], '10': [460,519], '11': [520,579], '12': [580,629], '13': [630,679], '14': [680,709],
                        '15': [710,739], '16': [740,759], '17': [760,779], '18': [780,799], '19': [800,999]}                                                                                                  
            for i in range(1, len(code_dict) + 1):
                if code_dict[str(i)][0] <= floor(code) <= code_dict[str(i)][1]:
                    chapter = str(i)
                    return chapter

        if code[-3:] == '-10':
            if code[:3] == 'O9A':
                chapter = '19'
                return chapter
            code = code[:3]
            code_dict = {'1': ['A00', 'B99'], '2': ['C00', 'D49'], '3': ['D50', 'D99'], '4': ['E00', 'E99'], '5': ['F00', 'F99'], '6': ['G00', 'G99'],'7': ['H00', 'H59'],
                                '8': ['H60', 'H99'],'9': ['I00', 'I99'], '10': ['J00', 'J99'], '11': ['K00', 'K99'], '12': ['L00', 'L99'], '13': ['M00', 'M99'], '14': ['N00', 'N99'], 
                                '15': ['O00', 'O99'], '16': ['P00', 'P99'], '17': ['Q00', 'Q99'], '18': ['R00', 'R99'], '19': ['S00', 'T99'], '20': ['V00', 'Y99'], '21': ['Z00', 'Z99'], 
                                '22': ['U00', 'U99']}  
            for i in range(1, len(code_dict)):
                if code_dict[str(i)][0] <= code <= code_dict[str(i)][1]:
                    chapter = str(i)
                    return chapter   

    def data_collator(self, batch, i):
        current = {'input_ids':batch['input_ids'][i], 'token_type_ids':batch['token_type_ids'][i], 'attention_mask':batch['attention_mask'][i]}
        max_token = max(current['token_type_ids'].tolist())
        if max_token == 1:
            batch['labels'].append(-100)
            return batch
        else:
            visit_to_not_keep = randrange(2,max_token+1) 
            first_index_to_not_keep = current['token_type_ids'].tolist().index(visit_to_not_keep)
            label = self.tokenizer.convert_ids_to_tokens(current['input_ids'][first_index_to_not_keep].item())
            label = self.lookup_ICD_10_chapter(label)
            batch['labels'].append(int(label))
            current['input_ids'][first_index_to_not_keep] = self.tokenizer.convert_tokens_to_ids('[SEP]')
            current['input_ids'][first_index_to_not_keep+1:] = self.tokenizer.convert_tokens_to_ids('[PAD]')
            current['token_type_ids'][first_index_to_not_keep:] = 0
            current['attention_mask'][first_index_to_not_keep:] = 0
            batch['input_ids'][i] = current['input_ids']
            batch['token_type_ids'][i] = current['token_type_ids']
            batch['attention_mask'][i] = current['attention_mask']
            return batch   

    def batch_data_collator(self, batch):
        batch_size = len(batch['input_ids'])
        batch['labels'] = []
        for i in range(batch_size):
            batch = self.data_collator(batch, i)
        batch['labels'] = LongTensor(batch['labels'])
        return batch

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
    pred_ids = torch.argmax(logits, dim=-1)
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
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'

    # local_rank = int(os.environ["LOCAL_RANK"])

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # torch.distributed.init_process_group("nccl")

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    tokenizer = AutoTokenizer.from_pretrained('/root/data/tokenizer-mimic-iv-icd-final')
    dataset_train = Dataset.load_from_disk('/root/data/mimic-iv_data/ood/hf_dataset_finetune_disease_prediction/train')
    dataset_test = Dataset.load_from_disk('/root/data/mimic-iv_data/ood/hf_dataset_finetune_disease_prediction/test')

  
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

    metric = load('accuracy')

    seeds = [42, 123, 500, 87, 11, 5, 55, 57, 100, 301]

    def model_init():
        model = BerthaForSequenceClassification.from_pretrained("/root/medair_skynet_01/output-2023-01-20-ood-pretraining/ood_pretraining/2023_01_20-220015_ignore_atomic_snomed_all/output/checkpoint-41000", num_labels = 22)
        model.to(device)
        return model


    for trial in range(args.num_trials):
        output_log_folder = '~/ryan/ood_pretraining/'
        timestr = time.strftime("%Y_%m_%d-%H%M%S_")
        

        trainingArgs = TrainingArguments(
            seed = seeds[trial],
            do_eval = True,
            do_train=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            load_best_model_at_end=True,
            learning_rate=LEARNING_RATE,
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
            ddp_find_unused_parameters=False,
            output_dir=output_log_folder + timestr + "fifty trials",
            logging_dir=output_log_folder + timestr + "fifty trials"
        )


        trainer = CustomTrainer(
            model_init=model_init,
            args=trainingArgs,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=DataCollatorForICD10(tokenizer),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )


        best_run = trainer.hyperparameter_search(
            hp_space = hp_space,
            backend='optuna',
            direction="maximize",
            n_trials=30,
        )

        print(best_run)
