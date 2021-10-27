# -*- coding: utf-8 -*-



from al4ea.al_modules import *
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import argparse
import pandas as pd
from al4ea.util import seed_everything
from al4ea.strategies.strategy_util import measure_uncertainty, construct_graph


class StructUncertaintyStrategy(Strategy):
    def __init__(self, data_dir, ea_module: EAModule, pool: Pool, al_settings: ALSettings):
        super(StructUncertaintyStrategy, self).__init__()
        self.data_dir = data_dir
        self.ea_module = ea_module
        self.pool = pool
        self.al_settings = al_settings
        self.graph = construct_graph(self.data_dir, self.al_settings.edge_mode)

    def measure_informativeness(self, unlabeled_ent1_list):
        # unlabeled_ent1_list = list(self.pool.ent1_in_pool)
        unlabeled_ent2_list = list(self.pool.ent2_in_pool)

        if not self.ea_module.trained:
            uncertainty = np.ones(shape=(len(unlabeled_ent1_list)), dtype=np.float)
        else:
            simi_mtx = self.ea_module.predict( unlabeled_ent1_list, unlabeled_ent2_list)  # pass data, output_dir
            uncertainty = measure_uncertainty(simi_mtx, topK=10, measure=self.al_settings.uncertainty_measure)  # each node gets an uncertainty score
        # sort nodes, select a batch, and then update the dataset
        ent1_uncertainty_map = {unlabeled_ent1_list[i]: uncertainty[i] for i in range(len(unlabeled_ent1_list))}
        nodes = self.graph.nodes()
        node2weight_map = {n: ent1_uncertainty_map.get(n, 0.0) for n in nodes}
        new_weights = pagerank(self.graph, alpha=self.al_settings.pr_alpha, personalization=node2weight_map, nstart=node2weight_map, dangling=None)  # todo: I dont need to use dangling. how to disable it? it will use personalization if I set is as None
        ent1_influence_map = {n: new_weights[n] for n in unlabeled_ent1_list}
        sorted_unlabeled_ent_list = sorted(unlabeled_ent1_list, key=lambda item: -ent1_influence_map.get(item))
        return sorted_unlabeled_ent_list

    def update(self, sampling_iteration=0):
        self.ea_module.update_model()


def al_process_with_struct_uncertainty():
    # initialize settings and modules
    al_settings = ALSettings()
    al_settings.ea_model_arg_fn = args.arg_fn
    al_settings.query_num_per_iteration = args.sample_num_per_ite
    al_settings.uncertainty_measure = args.uncertainty_measure
    al_settings.pr_alpha = args.pr_alpha
    al_settings.edge_mode = args.edge_mode

    pool = Pool(args.data_dir)
    ea_module = EAModule(data_dir=args.data_dir, ea_arg_fn=al_settings.ea_model_arg_fn)
    strategy = StructUncertaintyStrategy(data_dir=args.data_dir, ea_module=ea_module, pool=pool, al_settings=al_settings)
    oracle = Oracle(args.data_dir)
    anno_data = AnnoData(args.data_dir)

    general_al_process(args.data_dir, al_settings, pool, strategy, oracle, anno_data, ea_module)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, help="directory of data")
    ap.add_argument("--arg_fn", type=str, help="filepath of config file")
    ap.add_argument("--uncertainty_measure", type=str, choices=["entropy", "margin", "variation_ratio", "similarity"], help="measure of uncertainty")
    ap.add_argument("--pr_alpha", type=float, default=0.5, help="parameter alpha of pagerank")
    ap.add_argument("--edge_mode", type=str, default="basic", choices=["basic", "origin", "add_inverse", "basic_func", "add_inverse_func"], help="edge mode")
    ap.add_argument("--sample_num_per_ite", type=int, default=1000)
    ap.add_argument("--seed", type=int, help="seed of random")

    args, _ = ap.parse_known_args()
    seed_everything(args.seed)
    al_process_with_struct_uncertainty()






