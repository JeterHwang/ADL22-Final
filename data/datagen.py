import argparse
import csv
import json
import random

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BlenderbotForConditionalGeneration,
    BlenderbotTokenizer,
)


def preprocess(example):

    example["personas"] = [f"your persona: {p}" for p in example["personas"]]
    example["context"] = "\n".join(
        example["personas"]
        + (["additional_context"] if example["additional_context"] else [])
        + example["previous_utterance"]
    )

    return example


if __name__ == "__main__":
    random.seed(26)
    num_chats = 500 # can modify num chats here

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mname = "facebook/blenderbot-400M-distill"
    simulator = BlenderbotForConditionalGeneration.from_pretrained(mname).to(device)
    simulator_tokenizer = BlenderbotTokenizer.from_pretrained(mname)
    # load your bot
    bot = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill").to(device)
    bot_tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")

    f = open("../final_project_scripts/keywords.json")
    keywords_f = json.load(f)
    f.close()
    keywords, data, history = [], [], []
    for field in keywords_f.keys():
        for word in keywords_f[field]:
            keywords.append(word)
    # print("keywords =", keywords) 

    dataset = load_dataset("blended_skill_talk", split="train")

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

    
    print("dataset len = ", len(dataset))
    assert num_chats <= len(
        dataset
    ), f"--num_chats needs to be smaller than dataset (<={len(dataset)})"
    dataset = dataset.select(random.sample(range(len(dataset)), num_chats))

    output = []
    for index, context in enumerate(
        tqdm(dataset["context"])
    ):
        dialog = []
        # if not args.disable_output_dialog:
        #     print(f" dialog id: {index}")
        for _ in range(10):
            inputs = simulator_tokenizer(
                [
                    "</s> <s>".join(
                        ([context] + dialog if len(dialog) < 3 else dialog[-3:])
                    )
                ],
                return_tensors="pt",
                truncation=True,
            ).to(device)
            reply_ids = simulator.generate(**inputs)
            text = simulator_tokenizer.batch_decode(
                reply_ids, skip_special_tokens=True
            )[0].strip()
            dialog.append(text)
            if len(dialog) > 10:
                dialog = dialog[1:]
            if len(dialog) >= 10:
                for word in keywords:
                    if word in text:
                        data.append(dialog)


            # if not args.disable_output_dialog:
            #     print(f"\033[0;32;49m {'simulator: ': ^11}{text} \033[0;0m")

            # you might need to change this line due to the model you use
            inputs = bot_tokenizer(
                ["</s> <s>".join(dialog[-3:])], return_tensors="pt", truncation=True
            ).to(device)
            reply_ids = bot.generate(**inputs)
            text = bot_tokenizer.batch_decode(reply_ids, skip_special_tokens=True)[
                0
            ].strip()
            # if not args.disable_output_dialog:
            #     print(f"\033[0;33;49m {'bot: ': ^11}{text} \033[0;0m")

        output.append(dialog)
        # if not args.disable_output_dialog:
        #     print()

    idx = 0
    source, target = "", ""
    with open("train/text.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["inputs", "target"])
        for text in data:
            for i in range(1, len(text) - 2, 2):
                source, target = "", ""
                # print(type(target))
                target = text[i].strip("\"")
                
                for j in range(len(text)):
                    if i != j:
                        source += text[j].strip("\"")
                    else:
                        source += "@"
                    source += " "
                writer.writerow([source, target])
                    # print(len(text))
                idx += 1
