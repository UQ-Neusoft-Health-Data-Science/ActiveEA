# -*- coding: utf-8 -*-


import os
from al4ea.reader import save_links, read_links, read_kgs_n_links
import numpy as np
import time
from openea.modules.args.args_hander import load_args
from openea.modules.load.kgs import read_kgs_from_folder
import tensorflow as tf
from openea_run.main_from_args import get_model
from openea.modules.load.read import load_embeddings, read_dict
from openea.modules.load.read import uris_list_2ids
import shutil
import json


class Pool:
    def __init__(self, data_dir):
        with open(os.path.join(data_dir, "ent1_in_pool.txt")) as file:
            self.ent1_in_pool = set(file.read().strip().split("\n"))
        with open(os.path.join(data_dir, "ent2_in_pool.txt")) as file:
            self.ent2_in_pool = set(file.read().strip().split("\n"))
        self.pool = self.ent1_in_pool
        self.initial_size_of_pool = len(self.pool)

    def initial_sampling(self, num=100):
        np.random.seed(1011)
        l = list(self.pool)
        np.random.shuffle(l)
        selected_entities = l[:num]
        return selected_entities

    def sample(self, sel_ent1_list, sel_ent2_list):
        self.ent1_in_pool = self.ent1_in_pool.difference(set(sel_ent1_list))
        self.ent2_in_pool = self.ent2_in_pool.difference(set(sel_ent2_list))
        self.pool = self.ent1_in_pool

    def unlabelled_entities(self):
        return list(self.pool)


class Oracle:
    def __init__(self, data_dir):
        self.oracle_links = read_links(os.path.join(data_dir, "oracle.txt"))
        self.oracle_head2tail_map = dict(self.oracle_links)

    def annotate(self, ent_list):
        new_anno_links = []
        for ent1 in ent_list:
            ent2 = self.oracle_head2tail_map.get(ent1, "null")
            new_anno_links.append((ent1, ent2))
        return new_anno_links


class AnnoData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.tmp_division_dir = os.path.join(data_dir, "temp")
        if os.path.exists(self.tmp_division_dir):
            shutil.rmtree(self.tmp_division_dir)
        os.mkdir(self.tmp_division_dir)
        self.all_anno_links = []
        self.training_anno_links = []
        self.valid_anno_links = []
        self.test_anno_links = []
        self.oracle_links = self.links_of_ent1()

    def update_training_data(self, new_links):
        self.all_anno_links.extend(new_links)
        np.random.shuffle(new_links)
        new_num = len(new_links)
        new_train_anno_num = int(new_num * 0.8)
        self.training_anno_links.extend(new_links[:new_train_anno_num])
        self.valid_anno_links.extend(new_links[new_train_anno_num:])
        self.test_anno_links = list(set(self.oracle_links).difference(set(self.all_anno_links)))

        effective_training_links = [l for l in self.training_anno_links if l[1] != "null"]
        effective_valid_links = [l for l in self.valid_anno_links if l[1] != "null"]
        effective_test_links = [l for l in self.test_anno_links if l[1] != "null"]
        save_links(effective_training_links, os.path.join(self.tmp_division_dir, "train_links"))
        save_links(effective_valid_links, os.path.join(self.tmp_division_dir, "valid_links"))
        save_links(effective_test_links, os.path.join(self.tmp_division_dir, "test_links"))
        save_links(self.all_anno_links, os.path.join(self.data_dir, "annotations.txt"))
        save_links(self.training_anno_links+self.valid_anno_links, os.path.join(self.tmp_division_dir, "train_valid_anno_links"))
        save_links(self.test_anno_links, os.path.join(self.tmp_division_dir, "test_anno_links"))

    def existing_annotations(self):
        anno_links = read_links(os.path.join(self.data_dir, "annotations.txt"))
        return anno_links

    def links_of_ent1(self):
        pool = Pool(self.data_dir)
        oracle = Oracle(self.data_dir)
        oracle_links = oracle.annotate(list(pool.ent1_in_pool))
        return oracle_links

    def entities_in_kg1(self):
        pool = Pool(self.data_dir)
        return list(pool.ent1_in_pool)

    def entities_in_kg2(self):
        pool = Pool(self.data_dir)
        return list(pool.ent2_in_pool)



class EAModule:
    def __init__(self, data_dir, ea_arg_fn):
        division = "temp"
        self.model_dir = os.path.join(data_dir, "model")
        if os.path.exists(self.model_dir):
            shutil.rmtree(self.model_dir)
        os.makedirs(self.model_dir)
        self.args = load_args(ea_arg_fn)
        self.args.training_data = data_dir if data_dir.endswith("/") else data_dir + "/"
        self.args.dataset_division = division if division.endswith("/") else division + "/"
        self.args.output = self.model_dir if self.model_dir.endswith("/") else self.model_dir + "/"
        print(self.args.embedding_module)
        print(self.args)
        self.model = None
        self.trained = False

        self.use_mc_dropout = ("dropout" in self.args.__dict__) and (self.args.dropout > 0)

        if self.args.embedding_module == "RDGCN":
            print("caching global word embs for RDGCN")
            from openea.approaches.rdgcn import RDGCN
            self.rdgcn_model = RDGCN()
            self.rdgcn_model.cache_word_embs(self.args.training_data)
            print("finish caching global word embs for RDGCN")

    def update_model(self):
        t = time.time()
        remove_unlinked = False
        if self.args.embedding_module == "RSN4EA":
            remove_unlinked = True

        kgs = read_kgs_from_folder(self.args.training_data, self.args.dataset_division, self.args.alignment_module,
                                   self.args.ordered, remove_unlinked=remove_unlinked)
        if self.model is None:
            tf.reset_default_graph()
            self.model = get_model(self.args.embedding_module)()
            self.model.set_args(self.args)
            self.model.set_kgs(kgs)
            self.model.init()
            self.model.restore()
        else:
            self.model.set_kgs(kgs)
            self.model.ref_ent1 = kgs.valid_entities1 + kgs.test_entities1
            self.model.ref_ent2 = kgs.valid_entities2 + kgs.test_entities2
            self.model.reset_early_stop()
        self.model.valid("hits1")

        #### add this snippet to form the implementation 2 ####
        # # train model with new links
        # if new_train_links is not None:
        #     print("train model with new train links")
        #     new_ent1_list, new_ent2_list = zip(*new_train_links)
        #     train_model_with_new_data(model, kgs.kg1, kgs.kg2, new_ent1_list, new_ent2_list)

        # triain model with all data
        print("train model with all data")
        self.model.run()
        # self.model.test()
        self.model.save()
        print("Total run time = {:.3f} s.".format(time.time() - t))
        del self.model
        self.model = None

        if not self.trained:
            self.trained = True

    def predict(self, ent1_list, ent2_list):
        if self.use_mc_dropout:
            simi_mtx_sum = None
            for i in range(10):
                sub_dir = os.path.join(self.model_dir, "model_drop_%02d" % i)
                ent_embs = load_embeddings(os.path.join(sub_dir, "ent_embeds.npy"))
                kg1_ent_ids = read_dict(os.path.join(sub_dir, "kg1_ent_ids"))
                kg2_ent_ids = read_dict(os.path.join(sub_dir, "kg2_ent_ids"))
                ent1_id_list = [kg1_ent_ids[uri] for uri in ent1_list]
                ent2_id_list = [kg2_ent_ids[uri] for uri in ent2_list]
                ent1_embs = ent_embs[ent1_id_list]
                ent2_embs = ent_embs[ent2_id_list]
                simi_mtx = np.matmul(ent1_embs, np.transpose(ent2_embs))
                if simi_mtx_sum is None:
                    simi_mtx_sum = simi_mtx
                else:
                    simi_mtx_sum += simi_mtx
            simi_mtx = simi_mtx_sum / 10.0
        else:
            ent_embs = load_embeddings(os.path.join(self.model_dir, "ent_embeds.npy"))
            kg1_ent_ids = read_dict(os.path.join(self.model_dir, "kg1_ent_ids"))
            kg2_ent_ids = read_dict(os.path.join(self.model_dir, "kg2_ent_ids"))
            ent1_id_list = [kg1_ent_ids[uri] for uri in ent1_list]
            ent2_id_list = [kg2_ent_ids[uri] for uri in ent2_list]
            ent1_embs = ent_embs[ent1_id_list]
            ent2_embs = ent_embs[ent2_id_list]
            simi_mtx = np.matmul(ent1_embs, np.transpose(ent2_embs))
            return simi_mtx
        return simi_mtx



        # ent_embs = load_embeddings(os.path.join(self.model_dir, "ent_embeds.npy"))
        # kg1_ent_ids = read_dict(os.path.join(self.model_dir, "kg1_ent_ids"))
        # kg2_ent_ids = read_dict(os.path.join(self.model_dir, "kg2_ent_ids"))
        # ent1_id_list = [kg1_ent_ids[uri] for uri in ent1_list]
        # ent2_id_list = [kg2_ent_ids[uri] for uri in ent2_list]
        # ent1_embs = ent_embs[ent1_id_list]
        # ent2_embs = ent_embs[ent2_id_list]
        # simi_mtx = np.matmul(ent1_embs, np.transpose(ent2_embs))
        # return simi_mtx


class Strategy:
    def __init__(self):
        pass

    def measure_informativeness(self, unlabelled_entities):
        return list()

    def update(self, sampling_iteration=0):
        pass



class ALSettings:
    def __init__(self):
        self.initial_query_num = 100
        self.query_num_per_iteration = 100
        self.budget = 0.5  # int (number) or float (percentage)
        self.ea_model_arg_fn = None

        # uncertainty sampling
        self.uncertainty_measure = "entropy"

        # topology-based sampling
        self.edge_mode = "basic"

        # structure-aware uncertainty sampling
        self.pr_alpha = 0.1

        # bachelor recognizer
        self.bach_recog_arg_fn = ""



def general_al_process(data_dir, al_settings: ALSettings, pool: Pool, strategy: Strategy, oracle: Oracle, anno_data: AnnoData, ea_module: EAModule):

    with open(os.path.join(data_dir, "al_settings.json"), "w+") as file:
        file.write(json.dumps(al_settings.__dict__))

    ## status
    max_budget = al_settings.budget if isinstance(al_settings.budget, int) else int(pool.initial_size_of_pool*al_settings.budget)
    consumed_budget = 0

    sampling_iteration = 1
    # start AL process
    while consumed_budget < max_budget:
        print(f"=======>>>>>>>>>> NEW SAMPLING ITERATION: {consumed_budget}/{max_budget} <<<<<<<<<<========")
        ## select entities
        print(f"=======>>>>>>>>>> SAMPLING <<<<<<<<<<========")
        sorted_entities = strategy.measure_informativeness(pool.unlabelled_entities())
        sample_num = min(al_settings.query_num_per_iteration, max_budget-consumed_budget)
        new_selected_entities = sorted_entities[: sample_num]
        # new_selected_entities = pool.initial_sampling(al_settings.initial_query_num)

        ## annotate
        new_links = oracle.annotate(new_selected_entities)
        ## update annotation data and pool
        anno_data.update_training_data(new_links)
        sel_ent2_list = [link[1] for link in new_links]
        pool.sample(new_selected_entities, sel_ent2_list)  # remove selected entities from pool
        # ## update model
        # print(f"=======>>>>>>>>>> UPDATE EA MODEL <<<<<<<<<<========")
        # ea_module.update_model()
        print(f"=======>>>>>>>>>> UPDATE STRATEGY <<<<<<<<<<========")
        strategy.update(sampling_iteration)

        consumed_budget += sample_num
        sampling_iteration += 1


    # generate datasets with different annotation percentages
    anno_percentages = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    anno_numbers = [al_settings.initial_query_num] + [int(per*pool.initial_size_of_pool) for per in anno_percentages]
    anno_percentages = [0.00] + anno_percentages
    anno_links = anno_data.all_anno_links
    all_effective_links = oracle.oracle_links
    for idx in range(len(anno_percentages)):
        ratio = anno_percentages[idx]

        train_valid_num = anno_numbers[idx]
        train_num = int(train_valid_num * 0.8)
        valid_num = train_valid_num - train_num

        train_valid_links = anno_links[:train_valid_num]
        idxes = list(range(train_valid_num))
        np.random.shuffle(idxes)
        train_valid_links_arr = np.array(train_valid_links)
        train_links_arr = train_valid_links_arr[idxes[:train_num]]
        valid_links_arr = train_valid_links_arr[idxes[train_num:]]
        train_links = train_links_arr.tolist()
        valid_links = valid_links_arr.tolist()
        train_links = [l for l in train_links if l[1] != "null"]
        valid_links = [l for l in valid_links if l[1] != "null"]
        test_links = list(set(all_effective_links).difference(set(train_valid_links)))

        out_dir = os.path.join(data_dir, "%.2f" % ratio)
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        save_links(train_links, os.path.join(out_dir, "train_links"))
        save_links(valid_links, os.path.join(out_dir, "valid_links"))
        save_links(test_links, os.path.join(out_dir, "test_links"))



