import csv
from pathlib import Path
from argparse import ArgumentParser, Namespace

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument('--data', type=Path, default='../../OTTers/data/in_domain/test/source.csv')
    parser.add_argument('--pred', type=Path, default='./runs/finetune/generated_predictions.txt')
    parser.add_argument('--concat', type=Path, default='./T5-concat.txt')
    args = parser.parse_args()
    return args

def main(args):
    prediction, inputs = [], []
    with open(args.pred, 'r') as f:
        line = f.readline()
        while line:
            prediction.append(line.strip())
            line = f.readline()
    with open(args.data, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row, pred in zip(reader, prediction):
            inputs.append(row[1] + ' ' + pred + ' ' + row[2])
    with open(args.concat, 'w') as f:
        for row in inputs:
            f.write(row + '\n')

if __name__ == '__main__':
    args = parse_arg()
    main(args)