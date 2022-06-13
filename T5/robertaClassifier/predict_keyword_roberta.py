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
import random
from dataclasses import dataclass, field
from typing import Optional
import json

import datasets
import numpy as np
from datasets import Dataset
from datasets import load_metric
from datasets.utils import disable_progress_bar

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)


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

config = AutoConfig.from_pretrained(
    "./pretrained/Robertaclassifier/config.json",
    use_auth_token = None,
)
tokenizer = AutoTokenizer.from_pretrained(
    "./pretrained/Robertaclassifier/",
    use_fast = True,
    revision = "main",
    use_auth_token = None,
)
model = AutoModelForSequenceClassification.from_pretrained(
    "./pretrained/Robertaclassifier/pytorch_model.bin",
    config=config,
    use_auth_token = None,
    ignore_mismatched_sizes = False,
)

# Padding strategy
padding = "max_length"

max_seq_length = min(128, tokenizer.model_max_length)

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
data_collator = default_data_collator

training_args = MyTrainingArguments()
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
    with open("./pretrained/subdomain.json", 'r') as f:
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