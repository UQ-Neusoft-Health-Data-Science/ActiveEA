# -*- coding: utf-8 -*-



from al4ea.al_modules import *
import networkx as nx
import pandas as pd


def construct_graph(data_dir, edge_mode):
    # construct graph
    kg1, kg2, _ = read_kgs_n_links(data_dir)
    g = nx.Graph()
    g.add_nodes_from(kg1.entities_list)
    self_ent_pair = [(ent, ent) for ent in kg1.entities_list]
    triples_list = kg1.relation_triples_list

    if edge_mode == "add_inverse" or edge_mode == "add_inverse_func":
        inv_triple_list = [(tri[2], "inv_" + tri[1], tri[0]) for tri in triples_list]
        triples_list += inv_triple_list

    if edge_mode == "origin":
        ent_pairs = [(tri[0], tri[2]) for tri in triples_list]
    else:
        ent_pairs = [(tri[2], tri[0]) for tri in triples_list]

    if edge_mode == "basic_func" or edge_mode == "add_inverse_func":
        triple_df = pd.DataFrame(triples_list, columns=["head", "relation", "tail"])
        relation_types = triple_df["relation"].unique()
        rel2func_map = dict()
        for rel in relation_types:
            triples_of_rel = triple_df[triple_df["relation"] == rel]
            func = len(triples_of_rel["head"].unique()) / len(triples_of_rel)
            rel2func_map[rel] = func
        all_pairs = ent_pairs + self_ent_pair
        edge_weights = [rel2func_map[tri[1]] for tri in triples_list] + [1.0] * len(self_ent_pair)
        g.add_edges_from(all_pairs, weight=edge_weights)
    else:
        g.add_edges_from(ent_pairs + self_ent_pair)
    return g



def measure_uncertainty(simi_mtx, topK=5, measure="entropy"):  # measure options: entropy, margin, variation_ratio, similarity
    sorted_simi_mtx = np.sort(simi_mtx, axis=-1)
    if measure == "entropy":
        topk_simi_mtx = sorted_simi_mtx[:, -topK:]
        prob_mtx = topk_simi_mtx / topk_simi_mtx.sum(axis=1, keepdims=True)
        uncertainty = - np.sum(prob_mtx*np.log2(prob_mtx), axis=1)
    elif measure == "margin":
        margin = sorted_simi_mtx[:, -1] - sorted_simi_mtx[:, -2]
        uncertainty = - margin  # larger margin means small uncertainty
        uncertainty = uncertainty - uncertainty.min()
    elif measure == "variation_ratio":
        topk_simi_mtx = sorted_simi_mtx[:, -topK:]
        prob_mtx = topk_simi_mtx / topk_simi_mtx.sum(axis=1, keepdims=True)
        uncertainty = 1.0 - prob_mtx[:, -1]
    elif measure == "similarity":
        uncertainty = - sorted_simi_mtx[:, -1]
    else:
        raise Exception("unknown uncertainty measure")
    return uncertainty


