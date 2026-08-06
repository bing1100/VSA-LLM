import torch
from torch.optim import Adam
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import os
from medair_models.bertha.modelling_bertha import BerthaForMaskedLM
from medair_models.bertha.configuration_bertha import BerthaConfig
from transformers import get_cosine_schedule_with_warmup
from transformers import Trainer
from transformers import TrainingArguments
from evaluate import load
import argparse
import time

#/ Intended Functionality of the Script:
# Read arguments from command
# Pass path for .npy to the BERTHA config (path_to_pretrained_embeddings)
# Train


parser = argparse.ArgumentParser(description='choose hyperparameter and VSA embedding variables')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--position_embedding_type', type=str, choices=['absolute', 'rotary'], default='absolute')
parser.add_argument('--layer_norm_position', type=str, choices=['pre', 'post'], default='post')
parser.add_argument('--gradient_accumulation_steps', type=int, choices=range(1,9), default=1)
parser.add_argument('--snomed_group', type=str, choices=['ignore', 'group_zero', 'group_vectors'], default='ignore')
parser.add_argument('--dp_type', type=str, choices=['words', 'words_rv', 'atomic'], default='atomic')
parser.add_argument('--dp_composition', type=str, choices=['icd_name', 'snomed_name', 'snomed_all'], default='snomed_all')
parser.add_argument('--path_to_model_checkpoint', type=str, default=None)
parser.add_argument('--num_trials', type=int, choices=range(1, 15), default=3)
parser.add_argument('--freeze_layers', type=int, choices=range(12), default=0)
parser.add_argument('--num_saved_checkpoints', type=int, choices=range(1, 20), default=2)
parser.add_argument('--save_name', type=str, default=None)
parser.add_argument('--embedding_mlp', type=str, default=None)

norm_parser = parser.add_mutually_exclusive_group(required=False)
norm_parser.add_argument('--do_normalization', dest='normalization', action='store_true')
norm_parser.add_argument('--no_normalization', dest='normalization', action='store_false')
parser.set_defaults(normalization=True)

embeddings_parser = parser.add_mutually_exclusive_group(required=False)
embeddings_parser.add_argument('--pretrained_embeddings', dest='use_pretrained_embeddings', action='store_true')
embeddings_parser.add_argument('--no_pretrained_embeddings', dest='use_pretrained_embeddings', action='store_false')
parser.set_defaults(use_pretrained_embeddings=True)

# Only relevant to models without pre-trained embeddings
freeze_embedding_parser = parser.add_mutually_exclusive_group(required=False)
freeze_embedding_parser.add_argument('--freeze_embedding', dest='freeze_embedding', action='store_true')
freeze_embedding_parser.add_argument('--no_freeze_embedding', dest='freeze_embedding', action='store_false')
parser.set_defaults(freeze_embedding=False)

# For adding extra configurations not handled by other combinations of arguments
parser.add_argument('--path_to_embeddings', type=str, default=None)

args = parser.parse_args()

BATCH_SIZE = 128

LEARNING_RATE = args.lr

EPOCHS = args.epochs

DROPOUT_PROB = 0

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
        logits = logits.permute(0, 2, 1)
        # compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        preds = torch.argmax(logits, axis=1)
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
        self.optimizer = Adam(params=self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)

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

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.distributed.init_process_group("nccl")

    if not args.use_pretrained_embeddings:
          path_to_pretrained_embeddings = None
    elif args.use_pretrained_embeddings and args.path_to_embeddings is not None:
        path_to_pretrained_embeddings = args.path_to_embeddings
    else:
        embeddings_directory = '/root/data/HRR_Embeddings_v4_768_stopwords_rescale_0.01/'
        if args.dp_type == "atomic":
            path_to_pretrained_embeddings = embeddings_directory + f'vecs_{args.dp_type}_{args.snomed_group}_norm_{args.normalization}.tsv.npy'
        else:
            path_to_pretrained_embeddings = embeddings_directory + f'vecs_{args.dp_composition}_{args.dp_type}_{args.snomed_group}_norm_{args.normalization}.tsv.npy'        

    tokenization_params = {
        'max_length': 128,
        'truncation': True,
        'padding': 'max_length',
        'is_split_into_words': True,
        'return_special_tokens_mask': True
    }

    tokenizer = AutoTokenizer.from_pretrained('/root/data/tokenizer-mimic-iv-icd-final')
    dataset_train = Dataset.load_from_disk('/root/data/mimic-iv_data/mimic_icd_hf_dataset_pretraining/train')
    dataset_test = Dataset.load_from_disk('/root/data/mimic-iv_data/mimic_icd_hf_dataset_pretraining/test')

    config = BerthaConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dropout_prob=DROPOUT_PROB,
        attention_probs_dropout_prob=DROPOUT_PROB,
        max_position_embeddings=tokenization_params['max_length'],
        type_vocab_size=100,
        initializer_range=0.01,
        layer_norm_position=args.layer_norm_position,
        position_embedding_type=args.position_embedding_type,
        path_to_pretrained_embeddings=path_to_pretrained_embeddings, 
        apply_ffn_to_embeddings=args.embedding_mlp,
    )
    
    metric = load('accuracy')


    # model.to(device)
    seeds = [42, 123, 500, 87, 11, 5, 55, 57, 100, 301]

    def model_init():
        if args.path_to_model_checkpoint is None:
            model = BerthaForMaskedLM(config)
        else: 
            model = BerthaForMaskedLM.from_pretrained(args.path_to_model_checkpoint)
            
            if args.freeze_layers != 0:
                for li in range(args.freeze_layers):
                    model.bertha.encoder.layer[li].BerthaAttention.self_attention.query.weight.requires_grad = False
                    model.bertha.encoder.layer[li].BerthaAttention.self_attention.key.weight.requires_grad = False
                    model.bertha.encoder.layer[li].BerthaAttention.output.dense.weight.requires_grad = False

            if args.freeze_embedding:
                model.bertha.embeddings.word_embeddings.weight.requires_grad = False

        model.to(device)
        return model


    for trial in range(args.num_trials):
        output_log_folder = '/root/output/'
        timestr = time.strftime("%Y_%m_%d-%H%M%S_")
        
        if args.save_name is not None:
            output_dir = output_log_folder + args.save_name + '/output'
            logging_dir = output_log_folder + args.save_name + '/logs'
        elif args.use_pretrained_embeddings and args.path_to_embeddings is None:
            output_dir = output_log_folder + timestr + f'{args.lr}_{args.epochs}_{args.position_embedding_type}_{args.layer_norm_position}_{args.gradient_accumulation_steps}_{args.snomed_group}_{args.dp_type}_{args.dp_composition}_{args.normalization}_{args.use_pretrained_embeddings}/output'
            logging_dir = output_log_folder + timestr + f'{args.lr}_{args.epochs}_{args.position_embedding_type}_{args.layer_norm_position}_{args.gradient_accumulation_steps}_{args.snomed_group}_{args.dp_type}_{args.dp_composition}_{args.normalization}_{args.use_pretrained_embeddings}/logs'
        elif args.path_to_embeddings is not None:
            output_dir = output_log_folder + timestr + (args.path_to_embeddings.split('/')[-1]).split('.tsv.npy')[0] + '/output'
            logging_dir = output_log_folder + timestr + (args.path_to_embeddings.split('/')[-1]).split('.tsv.npy')[0] + '/logs'
        else:
            output_dir = output_log_folder + timestr + f'{args.lr}_{args.epochs}_{args.position_embedding_type}_{args.layer_norm_position}_{args.gradient_accumulation_steps}_{args.use_pretrained_embeddings}/output'
            logging_dir = output_log_folder + timestr + f'{args.lr}_{args.epochs}_{args.position_embedding_type}_{args.layer_norm_position}_{args.gradient_accumulation_steps}_{args.use_pretrained_embeddings}/logs'

        trainingArgs = TrainingArguments(
            output_dir=output_dir,
            logging_dir=logging_dir,
            seed = seeds[trial],
            do_eval = True,
            do_train=True,
            evaluation_strategy="epoch",
            eval_steps=50,
            save_strategy="epoch",
            save_total_limit=args.num_saved_checkpoints,
            load_best_model_at_end=True,
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE, 
            per_device_eval_batch_size=8,
            eval_accumulation_steps=20,
            num_train_epochs=EPOCHS,
            logging_strategy="steps",
            logging_steps=32, 
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            fp16 = False,
            bf16 = True,
            tf32=True,
            ddp_find_unused_parameters=False,

        )

        trainer = CustomTrainer(
            model_init = model_init,
            args = trainingArgs,
            train_dataset = dataset_train,
            eval_dataset=dataset_test,
            compute_metrics=compute_metrics,
            tokenizer = tokenizer,
            data_collator = DataCollatorForLanguageModeling(tokenizer),
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
        )

        
        trainer.train()
