# -*- coding: utf-8 -*-


import shutil

from al4ea.al_modules import *
import argparse
from al4ea.util import seed_everything
from al4ea.bachelor_recognizer.gcn_simi_thr_cv import GCNSimiThrBachRecogCV
from al4ea.strategies.struct_uncertainty_bachelor_recog_simi_thr import StructUncertaintySimiThrBachRecogStrategy


class StructUncertaintySimiThrBachRecogCVStrategy(StructUncertaintySimiThrBachRecogStrategy):
    def __init__(self, data_dir, ea_module: EAModule, pool: Pool, oracle: Oracle, al_settings: ALSettings):
        super(StructUncertaintySimiThrBachRecogCVStrategy, self).__init__(data_dir, ea_module, pool, oracle, al_settings)
        self.ent_recog = GCNSimiThrBachRecogCV(data_dir=data_dir, division="temp", arg_fn=self.al_settings.bach_recog_arg_fn)


def al_process():
    # initialize settings and modules
    al_settings = ALSettings()
    al_settings.ea_model_arg_fn = args.arg_fn
    al_settings.query_num_per_iteration = args.sample_num_per_ite
    al_settings.bach_recog_arg_fn = args.bach_recog_arg_fn
    al_settings.uncertainty_measure = args.uncertainty_measure
    al_settings.pr_alpha = args.pr_alpha
    al_settings.edge_mode = args.edge_mode

    pool = Pool(args.data_dir)
    ea_module = EAModule(data_dir=args.data_dir, ea_arg_fn=al_settings.ea_model_arg_fn)
    oracle = Oracle(args.data_dir)
    strategy = StructUncertaintySimiThrBachRecogCVStrategy(data_dir=args.data_dir, ea_module=ea_module, pool=pool, oracle=oracle, al_settings=al_settings)
    anno_data = AnnoData(args.data_dir)

    general_al_process(args.data_dir, al_settings, pool, strategy, oracle, anno_data, ea_module)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, help="directory of data")
    ap.add_argument("--arg_fn", type=str, help="filepath of config file")
    ap.add_argument("--bach_recog_arg_fn", type=str, help="filepath of config file of bachelor recognizer")
    ap.add_argument("--uncertainty_measure", type=str, choices=["entropy", "margin", "variation_ratio", "similarity"], help="measure of uncertainty")
    ap.add_argument("--pr_alpha", type=float, default=0.5, help="parameter alpha of pagerank")
    ap.add_argument("--edge_mode", type=str, default="basic", choices=["basic", "origin", "add_inverse", "basic_func", "add_inverse_func"], help="edge mode")
    ap.add_argument("--sample_num_per_ite", type=int, default=1000)
    ap.add_argument("--seed", type=int, help="seed of random")

    args, _ = ap.parse_known_args()
    seed_everything(args.seed)
    al_process()






