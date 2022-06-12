from unittest import result
import numpy as np
import random
import json
import time
import torch
from torch.utils.data import DataLoader
from argparse import ArgumentParser, Namespace
from pathlib import Path
from tqdm import tqdm

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import AdamW, Adafactor, get_scheduler
from accelerate import Accelerator
from datasets import load_metric

from dataset import read_data


def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True

def train(model, optimizer, lr_scheduler, train_loader, valid_loader, accelerator, tokenizer, metric, args):
    rouge1_curve, rouge2_curve, rougel_curve = [], [], []

    for epoch in range(args.num_epoch):
        train_loss = 0
        model.train()
        with tqdm(total=len(train_loader), desc=f"Epoch #{epoch}", disable=(accelerator.is_main_process == False)) as t:
            for step, (input_ids, attention_mask, labels, target) in enumerate(train_loader):
                if (step + 1) % args.gradient_accumulation_steps != 0:
                    with model.no_sync():
                        output = model(
                            input_ids = input_ids,
                            attention_mask = attention_mask,
                            labels = labels
                        )
                        loss = output.loss
                        loss = loss / args.gradient_accumulation_steps
                        accelerator.backward(loss)
                else:
                    output = model(
                        input_ids = input_ids,
                        attention_mask = attention_mask,
                        labels = labels
                    )
                    loss = output.loss
                    loss = loss / args.gradient_accumulation_steps
                    accelerator.backward(loss)

                train_loss += loss.item()
                if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
            
                if (step + 1) % args.logging_step == 0:
                    t.set_postfix({
                        'loss' : train_loss / args.logging_step,
                        'lr' : optimizer.param_groups[0]['lr']
                    })
                    train_loss = 0
                t.update(1)            
        
        if (epoch + 1) % args.validation_frequency == 0:
            model.eval()
            decoded_reesult, reference = [], []
            with torch.no_grad():
                for step, (input_ids, attention_mask, _, targets) in enumerate(tqdm(valid_loader)):
                    generated_tokens = accelerator.unwrap_model(model).generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=args.max_target_len,
                        do_sample=True,
                        top_k=50, 
                        top_p=1, 
                    )
                    generated_tokens = accelerator.pad_across_processes(
                        generated_tokens, 
                        dim=1, 
                        pad_index=tokenizer.pad_token_id
                    )
                    generated_tokens = accelerator.gather(generated_tokens)
                    decoded_preds = tokenizer.batch_decode(
                        generated_tokens, 
                        skip_special_tokens=True
                    )
                    for pred in decoded_preds:
                        decoded_reesult.append(pred.strip())
                    for target in targets:
                        reference.append(target)
                
            for res in decoded_reesult[:10]:
                print(res)
            result = metric.compute(predictions=[decoded_reesult], references=[reference])
            # print(json.dumps(rouge, indent=2))
            # rouge1_curve.append(rouge['rouge-1']['f'])
            # rouge2_curve.append(rouge['rouge-2']['f'])
            # rougel_curve.append(rouge['rouge-l']['f'])
            print({"bleu" : result['score']})
        
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            state_dict_path = args.ckpt_dir / f'Epoch{epoch + 1}'
            state_dict_path.mkdir(parents=True, exist_ok=True)
            unwrapped_model.save_pretrained(
                state_dict_path / 'model', 
                save_function=accelerator.save, 
                state_dict=accelerator.get_state_dict(model)
            )
            tokenizer.save_pretrained(state_dict_path / 'tokenizer')
        #accelerator.save(unwrapped_model.state_dict(), state_dict_path / 'model.pt')
        #unwrapped_model.config.to_json_file(state_dict_path / 'config.json')
    
    # accelerator.wait_for_everyone()
    # np.save(args.plot_dir / 'rouge-1.npy', np.array(rouge1_curve))
    # np.save(args.plot_dir / 'rouge-2.npy', np.array(rouge2_curve))
    # np.save(args.plot_dir / 'rouge-l.npy', np.array(rougel_curve))

def main(args):
    # Save Path
    timestr = time.strftime("%Y-%m-%d-%H:%M:%S")
    args.plot_dir = args.plot_dir / (timestr + "-npys")
    args.ckpt_dir = args.ckpt_dir / (timestr + "-ckpts")
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Set seed
    same_seeds(args.seed)
    # Set device
    accelerator = Accelerator()   

    # Model declaration
    print('----- Start Model Initialization -----')
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    ###########################################
    tokenizer.pad_token = tokenizer.eos_token #
    tokenizer.add_tokens(['@', '<s>', '</s>'], special_tokens=True)
    new_tokens = []
    g = open("../data/relation2text.json")
    relation = json.load(g)
    for key in relation.keys():
        new_tokens.append(key.lower())
    tokenizer.add_tokens(new_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))
    # print("# ", num_added_tokens, " tokens added")
    ###########################################
    print('----- Finish Model Initialization -----')

    # dataloader
    print('----- Start Reading Data -----')
    train_dataset, eval_dataset, test_dataset = read_data(args.data_dir, tokenizer, args)
    print('----- Finish Reading Data -----')

    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        pin_memory=True
    )
    valid_loader = DataLoader(
        eval_dataset, 
        batch_size=args.batch_size * 2, 
        shuffle=False, 
        pin_memory=True
    )
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    lr_scheduler = get_scheduler(
        'linear', 
        optimizer, 
        num_warmup_steps=len(train_loader) // (args.gradient_accumulation_steps * args.warmup_ratio) + 1,
        num_training_steps=(len(train_loader) // (args.gradient_accumulation_steps) + 1) * args.num_epoch,
    )
    metric = load_metric("sacrebleu")
    # optimizer = Adafactor(
    #     model.parameters(),
    #     lr=2e-4,
    #     eps=(1e-30, 1e-3),
    #     clip_threshold=1.0,
    #     decay_rate=-0.8,
    #     beta1=None,
    #     weight_decay=0.0,
    #     relative_step=False,
    #     scale_parameter=False,
    #     warmup_init=False,
    # )
    # optimizer = Adafactor(
    #     model.parameters(), 
    #     scale_parameter=True, 
    #     relative_step=True, 
    #     warmup_init=True, 
    #     lr=None
    # )
    # lr_scheduler = AdafactorSchedule(optimizer)
    model, optimizer, train_loader, valid_loader, lr_scheduler = accelerator.prepare(model, optimizer, train_loader, valid_loader, lr_scheduler)
    train(model, optimizer, lr_scheduler, train_loader, valid_loader, accelerator, tokenizer, metric, args)
    
def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="../data/out_of_domain/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/",
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./plot/",
    )
    # BERT max length
    parser.add_argument("--max_input_len", type=int, default=256)
    parser.add_argument("--max_target_len", type=int, default=128)

    # optimizer
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--lr", type=float, default=3e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=4)

    # training
    parser.add_argument("--num_epoch", type=int, default=16)
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--validation_frequency", type=int, default=8)
    parser.add_argument('--warmup_ratio', type=int, default=4)
    parser.add_argument('--phase', type=int, default=2)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)