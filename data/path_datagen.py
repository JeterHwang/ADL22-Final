import jsonlines
import csv
import json

def get_key(val, my_dict):
    for key, value in my_dict.items():
        if value == val:
            return key
    return "errrrrr!!!! key not found" 

# phase1
in_file_names = ["augfp4_in_domain_train_out_goldkeywords.jsonl", "augfp4_in_domain_valid_fromtrain.jsonl", "augfp4_out_of_domain_train_out_goldkeywords.jsonl", "augfp4_out_of_domain_dev_out_goldkeywords.jsonl"]
out_file_names = ["in_domain/train/phase1_text.csv", "in_domain/dev/phase1_text.csv", "out_of_domain/train/phase1_text.csv", "out_of_domain/dev/phase1_text.csv"]


g = open("relation2text.json")
relation = json.load(g)

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
                for key, val in relation.items():
                    if path.find(val) >= 0:
                        path.replace(val, key)
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
                for key, val in relation.items():
                    if path.find(val) >= 0:
                        path.replace(val, key)
                        print(val, key)
                target = path
                inputs = "context : " + d["context"] + " @ path_tailentity : " + d["path_tailentity"] + " @ path : " + path
                target = d["response"]
            idx += 1
            writer.writerow([inputs, target])