# -*- coding: utf-8 -*-



from al4ea.al_modules import *
import networkx as nx
from networkx.algorithms.link_analysis.pagerank_alg import pagerank
import argparse
import pandas as pd
from al4ea.util import seed_everything
from al4ea.strategies.struct_uncertainty import StructUncertaintyStrategy
from al4ea.strategies.strategy_util import measure_uncertainty, construct_graph


class StructUncertaintyBachRecogSimuStrategy(StructUncertaintyStrategy):
    def __init__(self, data_dir, having_mate_recall, bachelor_recall, ea_module: EAModule, pool: Pool, oracle: Oracle, al_settings: ALSettings):
        super(StructUncertaintyBachRecogSimuStrategy, self).__init__(data_dir=data_dir, ea_module=ea_module, pool=pool, al_settings=al_settings)
        self.ea_module = ea_module
        self.pool = pool
        self.oracle = oracle
        self.al_settings = al_settings
        self.graph = construct_graph(self.data_dir, self.al_settings.edge_mode)
        self.having_mate_recall = having_mate_recall
        self.bachelor_recall = bachelor_recall

        self.ent1_to_having_mate = self.recognize_bachelor()

    def recognize_bachelor(self):
        ### simulate bachelor recognition results
        ent1_to_having_mate = {}
        having_mate_recall = self.having_mate_recall
        bachelor_recall = self.bachelor_recall
        for ent1 in self.pool.pool:
            if ent1 in self.oracle.oracle_head2tail_map:
                res = np.random.choice([0, 1], size=1, replace=True, p=(1 - having_mate_recall, having_mate_recall))[0]
            else:
                res = np.random.choice([0, 1], size=1, replace=True, p=(bachelor_recall, 1 - bachelor_recall))[0]
            ent1_to_having_mate[ent1] = res
        return ent1_to_having_mate


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
        ent1_influence_map = {n: new_weights[n]*self.ent1_to_having_mate[n] for n in unlabeled_ent1_list}
        sorted_unlabeled_ent_list = sorted(unlabeled_ent1_list, key=lambda item: -ent1_influence_map.get(item))
        return sorted_unlabeled_ent_list

    def update(self, sampling_iteration=0):
        self.ea_module.update_model()


def al_process():
    # initialize settings and modules
    al_settings = ALSettings()
    al_settings.ea_model_arg_fn = args.arg_fn
    al_settings.query_num_per_iteration = args.sample_num_per_ite
    al_settings.uncertainty_measure = args.uncertainty_measure
    al_settings.pr_alpha = args.pr_alpha
    al_settings.edge_mode = args.edge_mode

    pool = Pool(args.data_dir)
    ea_module = EAModule(data_dir=args.data_dir, ea_arg_fn=al_settings.ea_model_arg_fn)
    oracle = Oracle(args.data_dir)
    strategy = StructUncertaintyBachRecogSimuStrategy(data_dir=args.data_dir, having_mate_recall=args.having_mate_recall, bachelor_recall=args.bachelor_recall, ea_module=ea_module, pool=pool, oracle=oracle, al_settings=al_settings)
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
    ap.add_argument("--having_mate_recall", type=float)
    ap.add_argument("--bachelor_recall", type=float)

    seed_everything()

    args, _ = ap.parse_known_args()
    al_process()






