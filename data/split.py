import argparse, random

def main(args):
    with open(args.dataset, "r") as f:
        lines = f.readlines()
    random.shuffle(lines)
    split = [0.9, 0.1]
    start = 0
    end = int(len(lines)*split[0])
    with open(args.dataset + "_train", 'w') as f:
        for line in lines[start: end]:
            f.write(line)
        start = end
        end = start + int(len(lines)*split[1])
    with open(args.dataset + "_valid", 'w') as f:
        for line in lines[start:]:
            f.write(line)
        start = end
        end = start + int(len(lines)*split[1])



if __name__ == "__main__":
    args = argparse.ArgumentParser(description='Split dataset')
    args.add_argument('--dataset', default=None, type=str, required=True, help='indices of GPUs to enable (default: all)')
    args.add_argument('--option', default=None, type=str, help='indices of GPUs to enable (default: all)')
    args = args.parse_args()
    main(args)