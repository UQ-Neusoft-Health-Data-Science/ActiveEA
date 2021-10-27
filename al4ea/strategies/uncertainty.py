# -*- coding: utf-8 -*-


from al4ea.al_modules import *
import pandas as pd
import argparse
from al4ea.util import seed_everything
from al4ea.strategies.strategy_util import measure_uncertainty


class UncertaintyStrategy(Strategy):
    def __init__(self, ea_module: EAModule, pool: Pool, al_settings: ALSettings):
        super(UncertaintyStrategy, self).__init__()
        self.ea_module = ea_module
        self.pool = pool
        self.al_settings = al_settings

    def measure_informativeness(self, unlabeled_ent1_list):
        unlabeled_ent2_list = list(self.pool.ent2_in_pool)

        if not self.ea_module.trained:
            uncertainty = np.ones(shape=(len(unlabeled_ent1_list)), dtype=np.float)
        else:
            simi_mtx = self.ea_module.predict(unlabeled_ent1_list, unlabeled_ent2_list)  # pass data, output_dir
            uncertainty = measure_uncertainty(simi_mtx, topK=10, measure=self.al_settings.uncertainty_measure)  # each node gets an uncertainty score

        # sort nodes, select a batch, and then update the dataset
        ent1_uncertainty_map = {unlabeled_ent1_list[i]: uncertainty[i] for i in range(len(unlabeled_ent1_list))}
        sorted_unlabeled_ent_list = sorted(unlabeled_ent1_list, key=lambda item: -ent1_uncertainty_map.get(item))
        return sorted_unlabeled_ent_list

    def update(self, sampling_iteration=0):
        self.ea_module.update_model()



def al_process_with_struct_uncertainty():
    # initialize settings and modules
    al_settings = ALSettings()
    al_settings.ea_model_arg_fn = args.arg_fn
    al_settings.query_num_per_iteration = args.sample_num_per_ite
    al_settings.uncertainty_measure = args.uncertainty_measure

    pool = Pool(args.data_dir)
    ea_module = EAModule(data_dir=args.data_dir, ea_arg_fn=al_settings.ea_model_arg_fn)
    strategy = UncertaintyStrategy(ea_module=ea_module, pool=pool, al_settings=al_settings)
    oracle = Oracle(args.data_dir)
    anno_data = AnnoData(args.data_dir)

    general_al_process(args.data_dir, al_settings, pool, strategy, oracle, anno_data, ea_module)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, help="directory of data")
    ap.add_argument("--arg_fn", type=str, help="filepath of config file")
    ap.add_argument("--uncertainty_measure", type=str, choices=["entropy", "margin", "variation_ratio", "similarity"], help="measure of uncertainty")
    ap.add_argument("--sample_num_per_ite", type=int, default=1000)
    ap.add_argument("--seed", type=int, help="seed of random")

    args, _ = ap.parse_known_args()
    seed_everything(args.seed)
    al_process_with_struct_uncertainty()


