import torch
import csv
import os 

def preprocess_function(examples, tokenizer, max_input_length=60, max_target_length=30):
    inputs = examples['inputs']
    targets = examples['target']

    model_inputs = tokenizer(
        inputs, 
        max_length=max_input_length, 
        truncation=True,
        padding='max_length',
        add_special_tokens=True,
        return_tensors="pt",
    )

    if targets is not None:
        # Set up the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, 
                max_length=max_target_length, 
                truncation=True, 
                padding='max_length',
                add_special_tokens=True,
                return_tensors="pt",
            )

        model_inputs["labels"] = labels["input_ids"]
        model_inputs["targets"] = targets
    return model_inputs

def read_data(data_dir, tokenizer, args):
    splits = ['train', 'dev', 'test']
    datasets = {}
    for split in splits:
        directory = data_dir / split / 'text.csv'
        if os.path.isfile(directory):
            with open(directory, newline='') as csvfile:
                rows = csv.reader(csvfile)
                data = []
                for i, row in enumerate(rows):
                    if i > 0:
                        data.append(row)
                datasets[split] = OTTersDataset(
                    preprocess_function({
                        'inputs' : [row[0] for row in data],
                        'target' : [row[1] for row in data] if split != 'test' else None,
                    }, tokenizer, args.max_input_len, args.max_target_len),
                    split=split
                )
        else:
            datasets[split] = None
    return datasets['train'], datasets['dev'], datasets['test']

class OTTersDataset(torch.utils.data.Dataset):
    def __init__(self, data, split):
        self.split = split
        self.data = data

    def __getitem__(self, idx):
        if self.split != 'test':
            return self.data['input_ids'][idx], self.data['attention_mask'][idx], self.data['labels'][idx], self.data['targets'][idx]
        else:
            return self.data['input_ids'][idx], self.data['attention_mask'][idx]

    def __len__(self):
        return len(self.data['input_ids'])