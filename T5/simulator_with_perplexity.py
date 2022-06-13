import argparse
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
from bot import T5bot, GPT5bot
from perplexity import perplexity

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

    parser.add_argument("--bot", type=str, default='T5bot', choices=['T5bot', 'GPT5bot'])
    parser.add_argument("--commensense_model_path", type=Path, default="./pretrained/checkpoints_6lendict_wcontains")
    parser.add_argument("--rel2text_path", type=Path, default="./pretrained/relation2text.json")
    parser.add_argument("--counts_path", type=Path, default="./pretrained/counts.txt")
    parser.add_argument("--model1_path", type=Path, default='./pretrained/model1')
    parser.add_argument("--model2_path", type=Path, default='./pretrained/model2')
    parser.add_argument("--tokenizer2_path", type=str, default='t5-small')
    parser.add_argument("--keywords_path", type=Path, default='./pretrained/keywords.json')
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
    perplex = []

    # load your bot
    if args.bot == 'T5bot':
        bot = T5bot.from_pretrained(
            args.model1_path, 
            args.model2_path, 
            args.tokenizer2_path, 
            args.keywords_path,
            args.device
        )
    else:
        bot = GPT5bot.from_pretrained(
            args.model2_path,
            args.tokenizer2_path,
            args.commensense_model_path,
            args.keywords_path,
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
            bot.choose_target() ## Choose a topic to transfer in this dialogue
            print(bot.target)
            dialog = []
            if not args.disable_output_dialog:
                print(f" dialog id: {index}")
            for _ in range(5):
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


                # generate a reply based on the conversation
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
                reply = simulator_tokenizer.batch_decode(
                    reply_ids, skip_special_tokens=True
                )[0].strip()

                # you might need to change this line due to the model you use
                text = bot.generate(
                    reply,
                    #dialog[-1],
                    args.max_input_len
                )
                dialog.append(text)
                if not args.disable_output_dialog:
                    print(f"\033[0;33;49m {'bot: ': ^11}{text} \033[0;0m")
            # print("sentences = ", "".join(dialog))
            print("perplexity = ", perplexity("".join(dialog)))
            perplex.append(perplexity("".join(dialog)))
            output.append(dialog)
            if not args.disable_output_dialog:
                print()

        print("overall perplexity = ", sum(perplex) / len(perplex))
        with open(args.output, "w") as f:
            for idx, dialog in enumerate(output):
                f.write(json.dumps({"id": idx, "dialog": dialog}) + "\n")
