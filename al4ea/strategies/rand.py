# -*- coding: utf-8 -*-



from al4ea.al_modules import *
import argparse
from al4ea.util import seed_everything
import random


class RandomStrategy(Strategy):
    def __init__(self):
        super(RandomStrategy, self).__init__()

    def measure_informativeness(self, unlabeled_ent1_list):
        random.shuffle(unlabeled_ent1_list)
        return unlabeled_ent1_list



def al_process():
    # initialize settings and modules
    al_settings = ALSettings()
    al_settings.ea_model_arg_fn = args.arg_fn
    al_settings.edge_mode = "origin" # args.edge_mode

    pool = Pool(args.data_dir)
    ea_module = EAModule(data_dir=args.data_dir, ea_arg_fn=al_settings.ea_model_arg_fn)
    strategy = RandomStrategy()
    oracle = Oracle(args.data_dir)
    anno_data = AnnoData(args.data_dir)

    al_settings.query_num_per_iteration = int(al_settings.budget * pool.initial_size_of_pool) if isinstance(al_settings.budget, float) else al_settings.budget

    general_al_process(args.data_dir, al_settings, pool, strategy, oracle, anno_data, ea_module)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, help="directory of data")
    ap.add_argument("--arg_fn", type=str, help="filepath of config file")
    ap.add_argument("--seed", type=int, help="seed of random")

    args, _ = ap.parse_known_args()
    seed_everything(args.seed)
    al_process()


