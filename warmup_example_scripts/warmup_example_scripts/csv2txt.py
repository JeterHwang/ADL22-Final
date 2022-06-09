import csv
from pathlib import Path
from argparse import ArgumentParser, Namespace

def parse_arg():
    parser = ArgumentParser()
    parser.add_argument('--csv', type=Path, default='../../OTTers/data/in_domain/test/target.csv')
    parser.add_argument('--txt', type=Path, default='./reference.txt')
    args = parser.parse_args()
    return args

def main(args):
    with open(args.csv, 'r') as csvfile, open(args.txt, 'w') as txtfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            txtfile.write(row[1] + '\n')

if __name__ == '__main__':
    args = parse_arg()
    main(args)