import os

import argparse
import time
import logging
import sys

from datasets import Dataset
from medair_models.bertha.modelling_bertha import BerthaForMaskedLM
from medair_models.bertha.configuration_bertha import BerthaConfig
from torch.optim import Adam
from transformers import (
    get_cosine_schedule_with_warmup, 
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    DataCollatorForLanguageModeling
)
from evaluate import load

import torch
import numpy as np

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
    active_mask = labels.flatten() != -100
    active_labels = labels.flatten()[active_mask]
    active_preds = preds.flatten()[active_mask]
    return accuracy_metric.compute(predictions=active_preds, references=active_labels)

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
        logits = logits.permute(0, 2, 1)
        # compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        preds = torch.argmax(logits, axis=1)
        # Get active labels
        active_mask = labels.view(-1) != -100
        active_labels = torch.masked_select(labels.view(-1), active_mask)
        active_preds = torch.masked_select(preds.view(-1), active_mask)
        acc = accuracy_metric.compute(predictions=active_preds.cpu().numpy(), references=active_labels.cpu().numpy())
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
        self.optimizer = Adam(params=self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self.optimizer, 
            num_warmup_steps=int(0.1*num_training_steps), 
            num_training_steps=num_training_steps)

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


if __name__ == "__main__":

    # arg parser
    parser = argparse.ArgumentParser(description='choose VSA embedding variables')

    # Arguments that may be needed for AWS
    # parser.add_argument('--input_model_dir', type=str, default=os.environ['SM_CHANNEL_MODEL'])
    parser.add_argument('--output_model_dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--output_dir', type=str, default=os.environ['SM_OUTPUT_DIR'])
    parser.add_argument('--output_data_dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    # parser.add_argument("--n_gpus", type=str, default=os.environ["SM_NUM_GPUS"])

    parser.add_argument('--tokenizer_path', type=str, default=os.environ['SM_CHANNEL_TOKENIZER'])
    parser.add_argument('--vsa_parser_path', type=str, default=None)

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dropout_prob', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=4e-6)
    parser.add_argument('--init_std', type=float, default=0.02)
    parser.add_argument('--dim', type=int, default=768)

    parser.add_argument('--num_trials', type=int, choices=range(1, 15), default=1)

    parser.add_argument('--snomed_group', type=str, choices=['ignore', 'group_zero', 'group_vectors'], default='ignore')
    parser.add_argument('--dp_type', type=str, choices=['words', 'words_rv', 'atomic'], default='atomic')
    parser.add_argument('--dp_composition', type=str, choices=['icd_name', 'snomed_name', 'snomed_all'], default='snomed_all')

    # parser.add_argument('--model_checkpoint', type=str, default=os.environ['SM_MODEL_CHKPT'])
 
    # Using random embeddings overrides VSA options above
    use_vsa_parser = parser.add_mutually_exclusive_group(required=False)
    use_vsa_parser.add_argument('--use_vsa', dest='use_vsa', action='store_true')
    use_vsa_parser.add_argument('--use_random_embeddings', dest='use_vsa', action='store_false')
    parser.set_defaults(use_vsa=False)

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

    #torch.backends.cuda.matmul.allow_tf32 = True
    #torch.backends.cudnn.allow_tf32 = True

    BATCH_SIZE = args.batch_size

    LEARNING_RATE = args.lr  # By default 1e-4

    EPOCHS = args.epochs  # By default 50

    DROPOUT_PROB = args.dropout_prob

    DIM = args.dim
    INITIALIZER_STD = args.init_std

    WEIGHT_DECAY = args.weight_decay

    # tokenizer and datasets

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    dataset_train = Dataset.load_from_disk(args.train) 
    dataset_test = Dataset.load_from_disk(args.test)

    logger.info(f" loaded dataset_train length is: {len(dataset_train)}")
    logger.info(f" loaded dataset_test length is: {len(dataset_test)}")
  
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
        path_to_vsa_parser_data=args.vsa_parser_path,
        tokenizer_vocab=tokenizer.vocab,
        use_vsa=args.use_vsa,
        **vsa_args
    )

    def model_init():
        model = BerthaForMaskedLM(config)
        model.to(device)
        return model

    seeds = [42, 123, 500, 87, 11, 5, 55, 57, 100, 301, 32, 77, 97, 117, 3]

    for trial in range(args.num_trials):
        
        output_log_folder = args.output_data_dir

        timestr = time.strftime("%Y_%m_%d-%H%M%S_")

        if args.use_vsa:
            output_dir = output_log_folder + "/" + timestr + f'{args.snomed_group}_{args.dp_type}_{args.dp_composition}/output'
            logging_dir = output_log_folder + "/" + timestr + f'{args.snomed_group}_{args.dp_type}_{args.dp_composition}/logs'
        else:
            output_dir = output_log_folder + "/" + timestr + "no_vsa/output"
            logging_dir = output_log_folder + "/" + timestr + "no_vsa/logs"
        

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
            #tf32=True,
            ddp_find_unused_parameters=False
        )


        trainer = CustomTrainer(
            model_init=model_init,
            args=trainingArgs,
            train_dataset=dataset_train,
            eval_dataset=dataset_test,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            data_collator=DataCollatorForLanguageModeling(tokenizer),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        logger.info("Starting training.")
        trainer.train()
        logger.info("Ending training.")

        # Save model to s3
        trainer.save_model(args.output_model_dir)
