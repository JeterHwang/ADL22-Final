import argparse
from ast import keyword
import json
import random
from pathlib import Path
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)
from bot import GPT2bot, GPT5bot, T5bot
from lstmClassifier import predict_keyword_lstm
from robertaClassifier import predict_keyword_roberta

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="facebook/blenderbot-400M-distill",
        type=str,
        help="model to chat with simulator",
    )

    parser.add_argument("--num_chats", default=5, type=int, help="the number of round")

    parser.add_argument("--split", default="train", type=str, help="split")

    parser.add_argument("--seed", default=26, type=int, help="random seed")

    parser.add_argument(
        "--interactive_mode",
        action="store_true",
        help="make the simualtor interact with the user (type 'stop' to stop the program, type 'exit' to leave current round)",
    )

    parser.add_argument(
        "--output",
        default="output.jsonl",
        type=str,
        help="file to save the dialogs",
    )

    parser.add_argument(
        "--disable_output_dialog",
        action="store_true",
        help="whether output the dialogs to the command line",
    )
    parser.add_argument("--bot", type=str, default="T5bot", choices=['T5bot', 'GPT5bot', 'GPT2bot'])
    parser.add_argument("--rel2text_path", type=Path, default="./pretrained/relation2text.json")
    parser.add_argument("--counts_path", type=Path, default="./pretrained/counts.txt")
    parser.add_argument("--T5model1_path", type=Path, default='./pretrained/T5model1')
    parser.add_argument("--T5model2_path", type=Path, default='./pretrained/T5model2')
    parser.add_argument("--T5tokenizer2_path", type=str, default='./pretrained/T5model2')
    parser.add_argument("--GPT2model1_path", type=Path, default='./pretrained/GPT2model1')
    parser.add_argument("--GPT2model2_path", type=Path, default='./pretrained/GPT2model2')
    parser.add_argument("--GPT2tokenizer2_path", type=str, default='./pretrained/GPT2model2')
    # parser.add_argument("--keywords_path", type=Path, default='./pretrained/keywords.json')
    parser.add_argument("--subdomain_path", type=Path, default='./pretrained/subdomain.json')
    parser.add_argument("--classifier_path", type=Path, default='./pretrained/LSTMclassifier')
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--device", type=str, default='cuda:0')
    args = parser.parse_args()

    return args


def preprocess(example):

    example["personas"] = [f"your persona: {p}" for p in example["personas"]]
    example["context"] = "\n".join(
        example["personas"]
        + (["additional_context"] if example["additional_context"] else [])
        + example["previous_utterance"]
    )

    return example


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    mname = "facebook/blenderbot-400M-distill"
    simulator = BlenderbotForConditionalGeneration.from_pretrained(mname).to(args.device)
    simulator_tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    print("start token = ", simulator_tokenizer.bos_token)

    casualLM = AutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path).to(args.device)
    casualLM_tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # load your bot
    if args.bot == 'T5bot':
        bot = T5bot.from_pretrained(
            args.T5model1_path,
            args.T5model2_path,
            args.T5model2_path,
            args.device,
        )
    elif args.bot == 'GPT5bot':
        bot = GPT5bot.from_pretrained(
            args.GPT2model1_path,
            args.T5model2_path,
            args.T5model2_path,
            args.rel2text_path,
            args.counts_path,
            args.device,
        )
    else:
        bot = GPT2bot.from_pretrained(
            args.GPT2model1_path,
            args.GPT2model2_path,
            args.GPT2tokenizer2_path,
            args.rel2text_path,
            args.counts_path,
            args.device,
        )

    dataset = load_dataset("blended_skill_talk", split=args.split)
    dataset = dataset.map(
        preprocess,
        remove_columns=[
            "free_messages",
            "guided_messages",
            "suggestions",
            "personas",
            "additional_context",
            "previous_utterance",
        ],
    )

    if args.interactive_mode:
        for _ in range(args.num_chats):
            dialog = ["hi"]
            while True:
                inputs = simulator_tokenizer(
                    ["</s> <s>".join(dialog[-3:])], return_tensors="pt", truncation=True
                ).to(args.device)
                # print("bello", "</s> <s>".join(dialog[-3:]))
                reply_ids = simulator.generate(**inputs, do_sample=True, top_p=0.8)
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                dialog.append(text)
                print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")

                text = input(f"\033[0;33;49m {'you: ': ^11}")
                dialog.append(text)
                if text in ["stop", "exit"]:
                    break
            if text == "stop":
                break
            print()
    else:
        assert args.num_chats <= len(
            dataset
        ), f"--num_chats needs to be smaller than dataset (<={len(dataset)})"
        dataset = dataset.select(random.sample(range(len(dataset)), args.num_chats))

        output = []
        for index, context in enumerate(
            tqdm(dataset["context"], disable=(not args.disable_output_dialog))
        ):
            dialog = []
            if not args.disable_output_dialog:
                print(f" dialog id: {index}")
            for _round in range(6):
                inputs = simulator_tokenizer(
                    [
                        "</s> <s>".join(
                            ([context] + dialog if len(dialog) < 3 else dialog[-3:])
                        )
                    ],
                    return_tensors="pt",
                    truncation=True,
                ).to(args.device)
                reply_ids = simulator.generate(**inputs)
                text = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()
                dialog.append(text)
                if not args.disable_output_dialog:
                    print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")
                if _round == 5:
                    continue
                # you might need to change this line due to the model you use
                inputs = casualLM_tokenizer(
                    ["</s> <s>".join(dialog[-3:])], return_tensors="pt", truncation=True
                ).to(args.device)
                reply_ids = casualLM.generate(**inputs)
                normal_conversation = casualLM_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[
                    0
                ].strip()
                # print(f"Intermediate : {normal_conversation}")
                ###############################################################
                keyword = predict_keyword_lstm(
                    dialog + [normal_conversation], 
                    args.classifier_path / "vocab.pkl", 
                    args.classifier_path / "embeddings.pt", 
                    args.classifier_path / "model.pkl", 
                    args.subdomain_path,
                )
                # keyword = predict_keyword_lstm(dialog + [normal_conversation])
                ################################################################
                # print(keyword)
                bot.target = keyword
                topic_transfer = bot.generate(
                    normal_conversation,
                    dialog,
                    args.max_input_len
                )
                dialog.append(topic_transfer)
                if not args.disable_output_dialog:
                    print(f"\033[0;33;49m {'bot: ': ^11}{topic_transfer} \033[0;0m")

            output.append(dialog)
            if not args.disable_output_dialog:
                print()

        with open(args.output, "w") as f:
            for idx, dialog in enumerate(output):
                f.write(json.dumps({"id": idx, "dialog": dialog}) + "\n")
