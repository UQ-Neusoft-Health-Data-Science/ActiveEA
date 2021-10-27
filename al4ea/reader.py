# -*- coding: utf-8 -*-


from openea.modules.load.read import read_relation_triples, read_attribute_triples, read_links
from openea.modules.load.kg import KG
from openea.modules.load.kgs import remove_unlinked_triples
import os
import json


read_links = read_links  # this is to let you know you can import read_links from reader.py


def read_kgs_n_links(data_folder, remove_unlinked=False):

    kg1_relation_triples, _, _ = read_relation_triples(os.path.join(data_folder, 'rel_triples_1'))
    kg2_relation_triples, _, _ = read_relation_triples(os.path.join(data_folder, 'rel_triples_2'))
    kg1_attribute_triples, _, _ = read_attribute_triples(os.path.join(data_folder, 'attr_triples_1'))
    kg2_attribute_triples, _, _ = read_attribute_triples(os.path.join(data_folder, 'attr_triples_2'))

    links = read_links(os.path.join(data_folder, 'ent_links'))

    if remove_unlinked:
        kg1_relation_triples = remove_unlinked_triples(kg1_relation_triples, links)
        kg2_relation_triples = remove_unlinked_triples(kg2_relation_triples, links)

    kg1 = KG(kg1_relation_triples, kg1_attribute_triples)
    kg2 = KG(kg2_relation_triples, kg2_attribute_triples)
    return kg1, kg2, links


def save_links(links, out_fn):
    with open(out_fn, "w+") as file:
        for link in links:
            file.write("\t".join(link) + "\n")


def save_annotation(anno, out_fn):
    with open(out_fn, "w+") as file:
        file.write(json.dumps(anno))


def load_al_settings(fn):
    with open(fn) as file:
        obj = json.loads(file.read())
    return obj


def read_links_with_steps(fn):
    with open(fn) as file:
        obj = json.loads(file.read())
    return obj


def save_links_with_steps(links_with_steps, out_fn):
    with open(out_fn, "w+") as file:
        file.write(json.dumps(links_with_steps))
