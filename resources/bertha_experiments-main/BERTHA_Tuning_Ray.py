import torch
from torch.optim import Adam
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from datasets import Dataset
import apex
import os

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# os.environ['MASTER_ADDR'] = 'localhost'
# os.environ['MASTER_PORT'] = '12355'

# local_rank = int(os.environ["LOCAL_RANK"])
# torch.distributed.init_process_group("nccl")

tokenization_params = {
    'max_length': 128,
    'truncation': True,
    'padding': 'max_length',
    'is_split_into_words': True,
    'return_special_tokens_mask': True
}

tokenizer = AutoTokenizer.from_pretrained('./data/tokenizer-icd-final')
dataset = Dataset.load_from_disk('/root/data/mimic-iv_data/mimic_icd_hf_dataset')
dataset = dataset.train_test_split(test_size=0.1)

BATCH_SIZE = 128

LEARNING_RATE = 1e-4

EPOCHS = 10

DROPOUT_PROB = 0

import medair_models
from medair_models.bertha.modelling_bertha import BerthaForMaskedLM
from medair_models.bertha.configuration_bertha import BerthaConfig

config = BerthaConfig(
    vocab_size=tokenizer.vocab_size,
    hidden_dropout_prob=DROPOUT_PROB,
    attention_probs_dropout_prob=DROPOUT_PROB,
    max_position_embeddings=tokenization_params['max_length'],
    type_vocab_size=100,
    initializer_range=0.01,
    layer_norm_position='post',  # defaults to post
    position_embedding_type='rotary' # defaults to prepared(?)

)

# model = BerthaForMaskedLM(config)
# model.to(device)

from transformers import get_cosine_schedule_with_warmup
from transformers import Trainer
from transformers import TrainingArguments
import numpy as np
from evaluate import load

metric = load('accuracy')

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    # preds = np.argmax(logits, axis=-1)
    preds= preds[0]
    active_mask = labels.flatten() != -100
    active_labels = labels.flatten()[active_mask]
    active_preds = preds.flatten()[active_mask]
    return metric.compute(predictions=active_preds, references=active_labels)

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        logits = logits.permute(0, 2, 1)
        # compute custom loss (suppose one has 3 labels with different weights)
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        preds = torch.argmax(logits, axis=1)
        # Get active labels
        active_mask = labels.view(-1) != -100
        active_labels = torch.masked_select(labels.view(-1), active_mask)
        active_preds = torch.masked_select(preds.view(-1), active_mask)
        acc = metric.compute(predictions=active_preds.cpu().numpy(), references=active_labels.cpu().numpy())
        # print(f'loss: {loss}, accuracy: {acc}')
        self.log(acc)
        return (loss, outputs) if return_outputs else loss

    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = Adam(params=self.model.parameters(), lr=self.args.learning_rate)
        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer, num_warmup_steps=750, num_training_steps=num_training_steps)

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids, labels

import ray
from ray import tune
from ray.tune import CLIReporter
from transformers import AutoConfig
from ray.tune.schedulers import PopulationBasedTraining, HyperBandScheduler, ASHAScheduler
def tune_transformer(num_samples=189988, gpus_per_trial=2, smoke_test=False, train_dataset=dataset['train'], eval_dataset=dataset['test'], config=config, tokenizer=tokenizer):
    data_dir_name = "./data" if not smoke_test else "./test_data"
    data_dir = os.path.abspath(os.path.join(os.getcwd(), data_dir_name))
    if not os.path.exists(data_dir):
        os.mkdir(data_dir, 0o755)

    model_name = "BerthaForMaskedLM"
    task_name = "mlm"

    task_data_dir = os.path.join(data_dir, task_name.upper())

    def model_init():
        model = BerthaForMaskedLM(config)
        model.to(device)
        return model

    training_args = TrainingArguments(
        output_dir='.',
        learning_rate=LEARNING_RATE,  # config
        do_train=True,
        do_eval=True,
        no_cuda=gpus_per_trial <= 0,
        evaluation_strategy="epoch",
        eval_accumulation_steps=20,
        save_strategy="epoch",
        save_total_limit=2,
        logging_strategy="steps", 
        logging_steps=32,
        load_best_model_at_end=True,
        num_train_epochs=EPOCHS,  # config
        max_steps=-1,
        per_device_train_batch_size=BATCH_SIZE,  # config
        per_device_eval_batch_size=8,  # config
        logging_dir="./logs",
        skip_memory_metrics=True,
        report_to="none",
        gradient_accumulation_steps=1,
        fp16 = False,
        bf16 = True,
        tf32=True,
        ddp_find_unused_parameters=False,
    )

    trainer = CustomTrainer(
        model_init=model_init,
        args = training_args,
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer = tokenizer,
        # optimizers=(optimizer, lr_scheduler),
        data_collator = DataCollatorForLanguageModeling(tokenizer),
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    asha_scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="eval_loss",
        mode="min",

    )

    pbt_scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="eval_loss",
        mode="min",
        perturbation_interval=1, # for population based scheduler
        hyperparam_mutations={
            "learning_rate": tune.choice([1e-6, 1e-5, 5e-5, 1e-4, 4e-4, 1e-3] ),
            "gradient_accumulation_steps": tune.choice([1,2,4,8]),

        },
    )

    tune_config = {
        "per_device_train_batch_size": BATCH_SIZE,
        "num_train_epochs":EPOCHS,
        "learning_rate":tune.choice([1e-6, 1e-5, 5e-5, 1e-4, 4e-4, 1e-3 ]),
        "gradient_accumulation_steps":tune.choice([1,2,4,8]),
        "max_steps": 1 if smoke_test else -1,  # Used for smoke test.
        # "scheduler":asha_scheduler,
    }

    

    reporter = CLIReporter(
        parameter_columns={
            "learning_rate": "lr",
            "gradient_accumation_steps": "grad_acc_steps"
        },
        metric_columns=["eval_loss", "eval_accuracy", "epoch", "training_iteration"],
    )


    trainer.hyperparameter_search(
        hp_space=lambda _: tune_config,
        backend="ray",
        n_trials=24,
        resources_per_trial={"cpu": 1, "gpu": gpus_per_trial},
        scheduler=asha_scheduler,
        keep_checkpoints_num=1,
        checkpoint_score_attr="training_iteration",
        stop={"training_iteration": 1} if smoke_test else None,
        progress_reporter=reporter,
        local_dir="~/ray_results/",
        name="tune_bertha_rotary_post",
        log_to_file=True,
    )

tune_transformer()

