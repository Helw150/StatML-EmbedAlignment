import sys
import datasets
import numpy as np
import pandas as pd
from datasets import load_dataset

import evaluate
import transformers

from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


def load_embeddings(filename):
    df = pd.read_csv(filename, sep=",")
    # Remove Column Numbers
    np_array = df.values[:, 1:]
    return torch.tensor(np_array)


model = sys.argv[1]
embeddings = sys.argv[2]
lang = sys.argv[3]

if model == "mono":
    if lang == "en":
        model_url = "WillHeld/en-bert-xnli"
    else:
        model_url = "WillHeld/es-bert-xnli"
elif model == "multi":
    model_url = "WillHeld/multi-bert-xnli"
else:
    model_url = "WillHeld/en-bert-xnli"


eval_dataset = load_dataset(
    "xnli",
    lang,
    split="validation",
    use_auth_token=None,
)

label_list = eval_dataset.features["label"].names

num_labels = len(label_list)

config = AutoConfig.from_pretrained(
    model_url,
    num_labels=num_labels,
    finetuning_task="xnli",
    use_auth_token=None,
)


model = AutoModelForSequenceClassification.from_pretrained(model_url)
if embeddings == "align" and lang == "es" and model == "align":
    tokenizer = AutoTokenizer.from_pretrained("WillHeld/es-bert-xnli")
    model.base_model.embeddings.word_embeddings.weight = load_embeddings(
        "embedding_files/aligned_es_embeddings.txt"
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(model_url)


def preprocess_function(examples):
    # Tokenize the texts
    return tokenizer(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        max_length=128,
        truncation=True,
    )


eval_dataset = eval_dataset.map(
    preprocess_function,
    batched=True,
    desc="Running tokenizer on validation dataset",
)

# Get the metric function
metric = evaluate.load("xnli")


def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return metric.compute(predictions=preds, references=p.label_ids)


# Initialize our Trainer without training data for evaluation
trainer = Trainer(
    model=model,
    args=None,
    train_dataset=None,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=default_data_collator,
)

metrics = trainer.evaluate(eval_dataset=eval_dataset)

metrics["eval_samples"] = len(eval_dataset)

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
print(metrics)
