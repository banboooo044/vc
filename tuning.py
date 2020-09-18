import optuna
from argparse import ArgumentParser, Namespace
import torch
from solver2 import Solver
import yaml
import sys

# 目的関数
def objective(trial):
    '''
    最適化対象のコード
    '''
    # VQ
    commitment_cost = trial.suggest_loguniform('commitment_cost', 1e-3, 100.0)
    epsilon = trial.suggest_loguniform('epsilon', 1e-7, 1e-3)
    # optimizer
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-2)
    # lambda
    lambda_rec = trial.suggest_loguniform('lambda_rec', 1e-3, 100.0)

    config['ContentEncoder']['commitment_cost'] = commitment_cost
    config['ContentEncoder']['epsilon'] = epsilon
    config['optimizer']['lr'] = lr
    config['lambda']['lambda_rec'] = lambda_rec
    solver = Solver(config=config, args=args)
    loss = solver.train(n_iterations=args.iters)
    return loss


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config2.yaml')
    parser.add_argument('-data_dir', '-d', default='')
    parser.add_argument('-train_set', default='train')
    parser.add_argument('-train_index_file', default='train_samples_64.json')
    parser.add_argument('--use_eval_set', action='store_true')
    parser.add_argument('-eval_set', default='in_test')
    parser.add_argument('-eval_index_file', default='train_samples_64.json')
    parser.add_argument('--use_test_set', action='store_true')
    parser.add_argument('-test_set', default='out_test')
    parser.add_argument('-test_index_file', default='out_test_sample_64.json')
    parser.add_argument('-logdir', default='log/')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--load_opt', action='store_true')
    parser.add_argument('-store_model_path', default='/storage/model/adaptive_vc/model')
    parser.add_argument('-load_model_path', default='/storage/model/adaptive_vc/model')
    parser.add_argument('-summary_steps', default=100, type=int)
    parser.add_argument('-save_steps', default=5000, type=int)
    parser.add_argument('-tag', '-t', default='init')
    parser.add_argument('-iters', default=0, type=int)

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)

    # 最適化（score：最小化, トライアル数：80）
    study = optuna.create_study()
    study.optimize(objective, n_trials=40)


    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))