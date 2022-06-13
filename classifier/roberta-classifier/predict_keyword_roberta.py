#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

import datasets
import numpy as np
from datasets import Dataset
from datasets import load_dataset, load_metric
from datasets.utils import disable_progress_bar

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint


label2id =  {
  "attraction-location": 0,
  "attraction-others": 1,
  "attraction-type": 2,
  "hotel-service": 3,
  "hotel-travel": 4,
  "movie-attribute": 5,
  "movie-theater": 6,
  "movie-type": 7,
  "restaurant-cooking": 8,
  "restaurant-dessert": 9,
  "restaurant-eat": 10,
  "restaurant-meal": 11,
  "restaurant-service": 12,
  "restaurant-type": 13,
  "song-attributes": 14,
  "song-method": 15,
  "song-performer": 16,
  "song-type": 17,
  "transportation-others": 18,
  "transportation-traffic": 19,
  "transportation-type": 20
}
id2label = {v:k for k, v in label2id.items()}

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: "},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default = "roberta_model/pytorch_model.bin",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default = "roberta_model/config.json",
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default = "roberta_model/",
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )

@dataclass
class MyTrainingArguments(TrainingArguments):
    output_dir: str = field(
        default = "./predict",
        metadata = {"help": "The output directory where the model predictions and checkpoints will be written."},
    )
    log_level: Optional[str] = field(
        default="error",
        metadata={
            "help": "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug', 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and lets the application set the level. Defaults to 'passive'."
        },
    )
    disable_tqdm: Optional[bool] = field(
        default=True, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )


parser = HfArgumentParser((ModelArguments, DataTrainingArguments, MyTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Load pretrained model and tokenizer
#
# In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
# download model & vocab.
config = AutoConfig.from_pretrained(
    model_args.config_name,
    finetuning_task=data_args.task_name,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
)
model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    config=config,
    cache_dir=model_args.cache_dir,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    ignore_mismatched_sizes=model_args.ignore_mismatched_sizes,
)

# Padding strategy
if data_args.pad_to_max_length:
    padding = "max_length"
else:
    # We will pad later, dynamically at batch creation, to the max sequence length in each batch
    padding = False

max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

def preprocess_function(examples):
    # Tokenize the texts
    contexts = []
    for example in examples['dialog']:
        contexts.append('<s>' + '</s> <s>'.join(example) + '</s>')
    result = tokenizer(contexts, padding=padding, max_length=max_seq_length, truncation=True)
    # Map labels to IDs (not necessary for GLUE tasks)
    # result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples[label_name]]
    return result

# Get the metric function
metric = load_metric("accuracy")

# You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.argmax(preds, axis=1)
    return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

# Data collator will default to DataCollatorWithPadding when the tokenizer is passed to Trainer, so we change it if
# we already did the padding.
if data_args.pad_to_max_length:
    data_collator = default_data_collator
elif training_args.fp16:
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
else:
    data_collator = None

# Initialize our Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


def predict_keyword_roberta(dialog_list):
    dialog_list = [dialog_list]
    input_dict = {"dialog": dialog_list}
    predict_dataset = Dataset.from_dict(input_dict)
    disable_progress_bar()
    datasets.logging.get_verbosity = lambda: logging.NOTSET
    predict_dataset = predict_dataset.map(
        preprocess_function,
        batched=True,
    )
    predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
    predictions = np.argmax(predictions, axis=1)
    with open("subdomain.json", 'r') as f:
        jsonObj = json.load(f)
    subdomain = id2label[predictions[0]]
    sample = np.random.choice(jsonObj[subdomain])
    return sample

if __name__ == "__main__":
    test_dialog_list = [
        "Chocolate is my favorite too! It's so sweet and delicious. It's also one of the most popular desserts in the world.",
        "I like that too! I also like to make my own chocolate by grinding up cocoa beans.",
        "That sounds really good, I'll have to try that sometime. Do you have a favorite dessert?"
    ]
    test_subdomain = "restaurant-dessert"
    result = predict_keyword_roberta([test_dialog_list])
    print('result:', result)