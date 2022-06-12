import jsonlines
import csv
import json

# phase1
in_file_names = ["augfp4_in_domain_train_out_goldkeywords.jsonl", "augfp4_in_domain_valid_fromtrain.jsonl", "augfp4_out_of_domain_train_out_goldkeywords.jsonl", "augfp4_out_of_domain_dev_out_goldkeywords.jsonl"]
out_file_names = ["in_domain/train/phase1_text.csv", "in_domain/dev/phase1_text.csv", "out_of_domain/train/phase1_text.csv", "out_of_domain/dev/phase1_text.csv"]


g = open("relation2text.json")
relation_ = json.load(g)
longest = 0
relation = []

for key, val in relation_.items():
    length = len(val.split())
    if length > longest:
        longest = length
count = 0
for i in range(longest, 0, -1):
    for key, val in relation_.items():
        length = len(val.split())
        if i == length:
            relation.append((key, val))

for i in range(4):
    data = []
    with jsonlines.open(in_file_names[i]) as reader:
        for obj in reader:
            data.append(obj)

    idx = 0
    with open(out_file_names[i], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["inputs", "target"])
        for d in data:
            if idx != 0:
                inputs = "context : " + d["context"] + " @ path_tailentity : " + d["path_tailentity"]
                path = d["path"]
                for key, val in relation:
                    path = path.replace(val, "")
                    path = path.replace(key, "")
                target = path
                writer.writerow([inputs, target])
            idx += 1



# phase2
out_file_names = ["in_domain/train/phase2_text.csv", "in_domain/dev/phase2_text.csv", "out_of_domain/train/phase2_text.csv", "out_of_domain/dev/phase2_text.csv"]
for i in range(4):
    data = []
    with jsonlines.open(in_file_names[i]) as reader:
        for obj in reader:
            data.append(obj)

    idx = 0
    with open(out_file_names[i], 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["inputs", "target"])
        for d in data:
            if idx != 0:
                path = d["path"]
                for key, val in relation:
                    path = path.replace(val, "")
                    path = path.replace(key, "")
                target = path
                inputs = "context : " + d["context"] + " @ path_tailentity : " + d["path_tailentity"] + " @ path : " + path
                target = d["response"]
            idx += 1
            writer.writerow([inputs, target])