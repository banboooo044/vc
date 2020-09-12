from argparse import ArgumentParser, Namespace
import torch
from evaluator import Evaluator
import yaml
import sys

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    parser.add_argument('-data_dir', '-d', default='')
    parser.add_argument('-test_set', default='out_test')
    parser.add_argument('-test_index_file', default='out_test_sample_64.json')
    parser.add_argument('-load_model_path', default='')
    parser.add_argument('-attr', default='')
    parser.add_argument('-tag', '-t', default='init')

    args = parser.parse_args()

    # load config file
    with open(args.config) as f:
        config = yaml.load(f)

    evaluator = Evaluator(config=config, args=args)

    evaluator.eval_rec()
