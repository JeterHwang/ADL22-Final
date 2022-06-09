from asyncore import write
import csv

data = []
with open("train_data.csv", newline='') as f:
    rows = csv.reader(f)
    for row in rows:
        data.append(row)
with open("train/text.csv", 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["inputs", "target"])
    for d in data:
        writer.writerow([d[1], d[2]])