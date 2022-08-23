# -*- coding: utf-8 -*-


from al4ea.reader import read_kgs_n_links
import random
import os
from al4ea.reader import save_links
import argparse
from al4ea.util import seed_everything
import shutil


def generate_bachelors(data_dir, bachelor_ratio):
    kg1, kg2, links = read_kgs_n_links(data_dir)

    random.shuffle(links)

    drop_num = int(len(links) * bachelor_ratio)
    drop_links = links[:drop_num]
    kept_links = links[drop_num:]
    kg2_dropped_entities = [p[1] for p in drop_links]
    ent2drop_map = {ent: True for ent in kg2_dropped_entities}

    kg2_left_rel_triple_list = [tri for tri in kg2.relation_triples_list if not ent2drop_map.get(tri[0], False) and not ent2drop_map.get(tri[2], False)]
    kg2_left_ent_list = []
    for tri in kg2_left_rel_triple_list:
        kg2_left_ent_list.append(tri[0])
        kg2_left_ent_list.append(tri[2])
    kg2_left_ent_list = list(set(kg2_left_ent_list))

    # kg2_dropped_rel_triple_list = [tri for tri in kg2.relation_triples_list if ent2drop_map.get(tri[0], False) or ent2drop_map.get(tri[2], False)]
    # kg2_dropped_entities = []
    # for tri in kg2_dropped_rel_triple_list:
    #     kg2_dropped_entities.append(tri[0])
    #     kg2_dropped_entities.append(tri[2])
    ent2left_map = {ent: True for ent in kg2_left_ent_list}
    kg2_left_att_triple_list = [tri for tri in kg2.attribute_triples_list if ent2left_map.get(tri[0], False)]

    print("left triple num:")
    print(len(kg2.relation_triples_list), "==>", len(kg2_left_rel_triple_list))
    print(len(kg2.attribute_triples_list), "==>", len(kg2_left_att_triple_list))

    save_links(kept_links, os.path.join(data_dir, "ent_links"))

    with open(os.path.join(data_dir, "rel_triples_2"), "w+") as file:
        for tri in kg2_left_rel_triple_list:
            file.write("%s\t%s\t%s\n"%tri)

    with open(os.path.join(data_dir, "attr_triples_2"), "w+") as file:
        for tri in kg2_left_att_triple_list:
            file.write("%s\t%s\t%s\n" % tri)


def generate_al_settings(data_dir):
    kg1, kg2, links = read_kgs_n_links(data_dir)
    ent1_in_pool = kg1.entities_list
    ent2_in_pool = kg2.entities_list
    with open(os.path.join(data_dir, "ent1_in_pool.txt"), "w+") as file:
        file.write("\n".join(ent1_in_pool))
    with open(os.path.join(data_dir, "ent2_in_pool.txt"), "w+") as file:
        file.write("\n".join(ent2_in_pool))
    shutil.copy(os.path.join(data_dir, "ent_links"), os.path.join(data_dir, "oracle.txt"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, help="directory of data")
    ap.add_argument("--bachelor_ratio", default=0.0, type=float, help="bachelor ratio")
    ap.add_argument("--seed", default=1011, type=int, required=False, help="seed of random")
    args, _ = ap.parse_known_args()
    data_dir = args.data_dir

    seed_everything(args.seed)
    print("generate bachelors")
    generate_bachelors(data_dir, bachelor_ratio=args.bachelor_ratio)
    print("set up AL settings")
    generate_al_settings(data_dir=data_dir)


