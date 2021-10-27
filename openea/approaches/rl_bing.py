# -*- coding: utf-8 -*-
"""
Created on 8/11/20 11:43 pm.

@author: Bing Liu (bing.liu@uq.edu.au, liubing.csai@gmail.com).

@desc:
"""

import gc
import math
import multiprocessing as mp
import time
import os

from openea.modules.finding.evaluation import early_stop
import openea.modules.train.batch as bat
from openea.modules.utils.util import task_divide
import tensorflow as tf

import gym
from gym.core import Env
import numpy as np
from openea.approaches.bootea import BootEA, bootstrapping
from spinup import vpg_pytorch as vpg
import pickle
import openea.modules.load.read as rd
from openea.modules.args.args_hander import load_args
from openea.modules.load.kgs import read_kgs_from_folder
import torch as th
import random


class CusBootEA(BootEA):
    def __init__(self):
        super(CusBootEA, self).__init__()
        self.newseed_labeled_align, self.newseed_entities1, self.newseed_entities2 = None, None, None

    def run(self):
        t = time.time()
        triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        manager = mp.Manager()
        training_batch_queue = manager.Queue()
        neighbors1, neighbors2 = None, None
        labeled_align = set()
        sub_num = self.args.sub_epoch
        iter_nums = self.args.max_epoch // sub_num
        for i in range(1, iter_nums + 1):
            print("\niteration", i)
            self.launch_training_k_epo(i, sub_num, triple_steps, steps_tasks, training_batch_queue, neighbors1,
                                       neighbors2)
            if i * sub_num >= self.args.start_valid:
                flag = self.valid(self.args.stop_metric)
                self.flag1, self.flag2, self.early_stop = early_stop(self.flag1, self.flag2, flag)
                if self.early_stop or i == iter_nums:
                    break
            labeled_align, entities1, entities2 = bootstrapping(self.eval_ref_sim_mat(),
                                                                self.ref_ent1, self.ref_ent2, labeled_align,
                                                                self.args.sim_th, self.args.k)
            self.train_alignment(self.kgs.kg1, self.kgs.kg2, entities1, entities2, 1)
            # self.likelihood(labeled_align)
            if i * sub_num >= self.args.start_valid:
                self.valid(self.args.stop_metric)
            t1 = time.time()
            assert 0.0 < self.args.truncated_epsilon < 1.0
            neighbors_num1 = int((1 - self.args.truncated_epsilon) * self.kgs.kg1.entities_num)
            neighbors_num2 = int((1 - self.args.truncated_epsilon) * self.kgs.kg2.entities_num)
            if neighbors1 is not None:
                del neighbors1, neighbors2
            gc.collect()
            neighbors1 = bat.generate_neighbours_single_thread(self.eval_kg1_useful_ent_embeddings(),
                                                               self.kgs.useful_entities_list1,
                                                               neighbors_num1, self.args.batch_threads_num)
            neighbors2 = bat.generate_neighbours_single_thread(self.eval_kg2_useful_ent_embeddings(),
                                                               self.kgs.useful_entities_list2,
                                                               neighbors_num2, self.args.batch_threads_num)
            ent_num = len(self.kgs.kg1.entities_list) + len(self.kgs.kg2.entities_list)
            print("generating neighbors of {} entities costs {:.3f} s.".format(ent_num, time.time() - t1))
            self.newseed_labeled_align, self.newseed_entities1, self.newseed_entities2 = labeled_align, entities1, entities2
        print("Training ends. Total time = {:.3f} s.".format(time.time() - t))

    def save(self):
        super(CusBootEA, self).save()
        labeled_align_cache = {"labeled_align": self.newseed_labeled_align, "entities1": self.newseed_entities1, "entities2": self.newseed_entities2}
        with open(os.path.join(self.out_folder, "labeled_align.pkl"), "wb") as f:
            pickle.dump(labeled_align_cache, f)


class RLEA2(CusBootEA):
    def __init__(self, model_cache_dir=None):
        super(RLEA2, self).__init__()

        if model_cache_dir:
            self.load_model(model_cache_dir)

    def load_model(self, cache_dir):  # load model and labeled_align
        print("loading parameters from cache ...")
        ent_emb_fn = os.path.join(cache_dir, "ent_embeds.npy")
        rel_emb_fn = os.path.join(cache_dir, "rel_embeds.npy")
        labeled_align_fn = os.path.join(cache_dir, "labeled_align.pkl")
        ent_emb = rd.load_embeddings(ent_emb_fn)
        rel_emb = rd.load_embeddings(rel_emb_fn)
        with open(labeled_align_fn, "rb") as f:
            labeled_align_result = pickle.load(f)
        self.newseed_labeled_align, self.newseed_entities1, self.newseed_entities2 = labeled_align_result["labeled_align"], \
                                                             labeled_align_result["entities1"], \
                                                             labeled_align_result["entities2"]
        self.start_ent_emb = ent_emb
        self.start_rel_emb = rel_emb


    def init(self):
        super(RLEA2, self).init()

        with tf.variable_scope('relational' + 'embeddings', reuse=True):
            self.ent_emb_var = tf.get_variable(name="ent_embeds")
            self.rel_emb_var = tf.get_variable(name="rel_embeds")
        self.restore_ent_op = self.ent_emb_var.assign(self.start_ent_emb)
        self.restore_rel_op = self.rel_emb_var.assign(self.start_rel_emb)
        self.simi_fea_op = self.similarity_feature()
        self.rollback()

    def train_with_new_seeds(self, iteration, indication_list):
        # triples_num = self.kgs.kg1.relation_triples_num + self.kgs.kg2.relation_triples_num
        # triple_steps = int(math.ceil(triples_num / self.args.batch_size))
        # steps_tasks = task_divide(list(range(triple_steps)), self.args.batch_threads_num)
        # manager = mp.Manager()
        # training_batch_queue = manager.Queue()
        # neighbors1, neighbors2 = None, None
        # sub_num = self.args.sub_epoch
        # self.launch_training_k_epo(iteration, sub_num, triple_steps, steps_tasks, training_batch_queue, neighbors1,
        #                            neighbors2)

        # labeled_align = [self.labeled_align[idx] for idx in range(len(indication_list)) if indication_list[idx]]
        entities1 = [self.newseed_entities1[idx] for idx in range(len(indication_list)) if indication_list[idx]]
        entities2 = [self.newseed_entities2[idx] for idx in range(len(indication_list)) if indication_list[idx]]
        self.train_alignment(self.kgs.kg1, self.kgs.kg2, entities1, entities2, 1)  # just run transe for new seed

    def similarity_feature(self):
        ent1_embs = tf.nn.embedding_lookup(self.ent_embeds, self.newseed_entities1)
        ent2_embs = tf.nn.embedding_lookup(self.ent_embeds, self.newseed_entities2)
        sim_op = tf.reduce_sum(tf.multiply(ent1_embs, ent2_embs), axis=1)
        return sim_op

    def get_new_seed_candidates(self):
        test_sim = self.session.run(self.simi_fea_op)
        # test_prob = self.calculate_holistic_mat(outvec_e, rel_prob, entities1, entities2)
        test_prob = np.zeros(len(test_sim))  # todo: for speed up debug
        test_sample = np.concatenate((test_sim.reshape(-1, 1), test_prob.reshape(-1, 1)), axis=1)
        return test_sample

    def rollback(self):
        self.session.run(self.restore_ent_op)
        self.session.run(self.restore_rel_op)

    def analyze(self):  # to explore how to design reward

        met_name = "mrr"
        init_metric = self.valid(met_name)

        # self.rollback()
        # self.train_alignment(self.kgs.kg1, self.kgs.kg2, self.seed_entities1, self.seed_entities2, 1)
        # all_metric = self.valid("hits5")

        newseed_labeled_align = list(self.newseed_labeled_align)
        best_ent1 = [self.newseed_entities1[i] for i in range(len(newseed_labeled_align)) if newseed_labeled_align[i][0]==newseed_labeled_align[i][1]]
        best_ent2 = [self.newseed_entities2[i] for i in range(len(newseed_labeled_align)) if newseed_labeled_align[i][0]==newseed_labeled_align[i][1]]
        wrong_ent1 = [self.newseed_entities1[i] for i in range(len(newseed_labeled_align)) if newseed_labeled_align[i][0] != newseed_labeled_align[i][1]]
        wrong_ent2 = [self.newseed_entities2[i] for i in range(len(newseed_labeled_align)) if newseed_labeled_align[i][0] != newseed_labeled_align[i][1]]

        per0_ent1 = best_ent1 + wrong_ent1[:int(len(wrong_ent1)*0.0)]
        per0_ent2 = best_ent2 + wrong_ent2[:int(len(wrong_ent2) * 0.0)]
        self.rollback()
        self.train_alignment(self.kgs.kg1, self.kgs.kg2, per0_ent1, per0_ent2, 1)
        per0_metric = self.valid(met_name)

        per03_ent1 = best_ent1 + wrong_ent1[:int(len(wrong_ent1) * 0.3)]
        per03_ent2 = best_ent2 + wrong_ent2[:int(len(wrong_ent2) * 0.3)]
        self.rollback()
        self.train_alignment(self.kgs.kg1, self.kgs.kg2, per03_ent1, per03_ent2, 1)
        per03_metric = self.valid(met_name)

        per06_ent1 = best_ent1 + wrong_ent1[:int(len(wrong_ent1) * 0.6)]
        per06_ent2 = best_ent2 + wrong_ent2[:int(len(wrong_ent2) * 0.6)]
        self.rollback()
        self.train_alignment(self.kgs.kg1, self.kgs.kg2, per06_ent1, per06_ent2, 1)
        per06_metric = self.valid(met_name)

        per1_ent1 = best_ent1 + wrong_ent1[:int(len(wrong_ent1) * 1)]
        per1_ent2 = best_ent2 + wrong_ent2[:int(len(wrong_ent2) * 1)]
        self.rollback()
        self.train_alignment(self.kgs.kg1, self.kgs.kg2, per1_ent1, per1_ent2, 1)
        per1_metric = self.valid(met_name)

        self.rollback()
        self.train_alignment(self.kgs.kg1, self.kgs.kg2, self.newseed_entities1, self.newseed_entities2, 1)
        all_metric = self.valid(met_name)

        print(f"init_metric: {init_metric}, "
              f"per0_metric: {per0_metric}, "
              f"per03_metric: {per03_metric}, "
              f"per06_metric: {per06_metric}, "
              f"per1_metric: {per1_metric}, "
              f"all_metric: {all_metric}")

    def analyze_effect_of_order(self, noise_per=0.5):
        newseed_labeled_align = list(self.newseed_labeled_align)
        best_ent1 = [self.newseed_entities1[i] for i in range(len(newseed_labeled_align)) if
                     newseed_labeled_align[i][0] == newseed_labeled_align[i][1]]
        best_ent2 = [self.newseed_entities2[i] for i in range(len(newseed_labeled_align)) if
                     newseed_labeled_align[i][0] == newseed_labeled_align[i][1]]
        wrong_ent1 = [self.newseed_entities1[i] for i in range(len(newseed_labeled_align)) if
                      newseed_labeled_align[i][0] != newseed_labeled_align[i][1]]
        wrong_ent2 = [self.newseed_entities2[i] for i in range(len(newseed_labeled_align)) if
                      newseed_labeled_align[i][0] != newseed_labeled_align[i][1]]

        newseed_entities1 = best_ent1 + wrong_ent1[:int(len(wrong_ent1) * noise_per)]
        newseed_entities2 = best_ent2 + wrong_ent2[:int(len(wrong_ent2) * noise_per)]


        met_name = "mrr"
        newseed_ent1 = np.array(newseed_entities1)
        newseed_ent2 = np.array(newseed_entities2)
        idx = np.arange(0, len(newseed_entities2))

        met_list = []
        for _ in range(10):
            # np.random.shuffle(idx)
            ent1 = newseed_ent1[idx]
            ent2 = newseed_ent2[idx]
            self.rollback()
            self.train_alignment(self.kgs.kg1, self.kgs.kg2, ent1, ent2, 1)
            metric = self.valid(met_name)
            met_list.append(metric)
        mean_met = np.mean(met_list)
        var_met = np.std(met_list)
        print(met_list)
        print(f"mean: {mean_met}, var: {var_met}")




class EAEnv(Env):
    def __init__(self, rlea: RLEA2):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]))

        self.met_name = "mrr"
        self.rlea = rlea
        self.rlea.rollback()
        self.rlea.train_with_new_seeds(1, np.ones(shape=len(self.rlea.newseed_labeled_align)))
        self.init_valid_metric = self.rlea.valid(self.met_name)
        # nodes and selections
        self.nodes = rlea.get_new_seed_candidates()
        self.sel_actions = []
        self.trajectory_idx = 0

    def step(self, action):
        self.sel_actions.append(action)
        self.trajectory_idx += 1
        done = self.trajectory_idx == len(self.nodes)
        if done:
            self.rlea.rollback()
            self.rlea.train_with_new_seeds(1, self.sel_actions)
            new_metric = self.rlea.valid(self.met_name)
            reward = (new_metric - self.init_valid_metric) * 10000
            print("======== %.4f -> %.4f ========" % (self.init_valid_metric, new_metric))
            obs = self.nodes[0]
        else:
            obs = self.nodes[self.trajectory_idx]
            reward = 0.0
        info = {}
        return obs, reward, done, info

    def reset(self):
        self.sel_actions = []
        self.trajectory_idx = 0
        obs = self.nodes[0]
        return obs

    def render(self, mode='human'):
        pass



def train_rlea(log_dir):
    arg_fn = "/home/uqbliu3/experiments/OpenEA-master/run/args/rlea_args_15K.json"
    data_name = "D_W_15K_V1"
    split_no = "721_5fold/1/"
    args = load_args(arg_fn)
    args.training_data = "/home/uqbliu3/experiments/datasets/" + data_name + '/'
    args.dataset_division = split_no
    remove_unlinked = False
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=remove_unlinked)

    rlea_model_dir = "/home/uqbliu3/experiments/output/results/CusBootEA/D_W_15K_V1/721_5fold/1/20201119170452"
    rlea = RLEA2(model_cache_dir=rlea_model_dir)
    rlea.set_args(args)
    rlea.set_kgs(kgs)
    rlea.init()

    env_fn = lambda: EAEnv(rlea)
    ac_kwargs = dict(hidden_sizes=[4], activation=th.nn.ReLU)
    logger_kwargs = dict(output_dir=log_dir, exp_name='rlea')
    vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, max_ep_len=100000, steps_per_epoch=3084*10, epochs=200, logger_kwargs=logger_kwargs)


def analyze_reward():
    arg_fn = "/home/uqbliu3/experiments/OpenEA-master/run/args/rlea_bing_args_15K.json"
    data_name = "D_W_15K_V1"
    split_no = "721_5fold/1/"
    args = load_args(arg_fn)
    args.training_data = "/home/uqbliu3/experiments/datasets/" + data_name + '/'
    args.dataset_division = split_no
    remove_unlinked = False
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=remove_unlinked)

    rlea_model_dir = "/home/uqbliu3/experiments/output/results/CusBootEA/D_W_15K_V1/721_5fold/1/20201119170452"
    rlea = RLEA2(model_cache_dir=rlea_model_dir)
    rlea.set_args(args)
    rlea.set_kgs(kgs)
    rlea.init()

    rlea.analyze_effect_of_order(noise_per=0.9)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"

    # train_rlea("/home/uqbliu3/experiments/output/results/RLEA2/try01/")

    analyze_reward()





