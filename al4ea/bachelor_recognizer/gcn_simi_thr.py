# -*- coding: utf-8 -*-



from openea.modules.args.args_hander import load_args
from openea.modules.load.read import load_embeddings, read_dict
import numpy as np
import os
import time
from openea.modules.load.kgs import read_kgs_from_folder
from openea_run.main_for_al import get_model
import tensorflow as tf
from al4ea.al_modules import Pool, AnnoData
from al4ea.reader import save_links, read_links
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


class GCNSimiThrBachRecog:
    def __init__(self, data_dir, division, arg_fn):
        self.data_dir = data_dir
        self.division = division
        self.arg_fn = arg_fn
        self.division_dir = os.path.join(self.data_dir, division)
        bach_recog_division = division + "/bachelor_recog_data/"
        self.bach_recog_data_dir = os.path.join(self.data_dir, bach_recog_division)
        self.model_dir = os.path.join(data_dir, division, "bachelor_recog_model")
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.args = load_args(arg_fn)
        self.args.training_data = data_dir if data_dir.endswith("/") else data_dir + "/"
        self.args.dataset_division = bach_recog_division if bach_recog_division.endswith("/") else bach_recog_division + "/"
        self.args.output = self.model_dir if self.model_dir.endswith("/") else self.model_dir + "/"
        self.model = None

        anno_data = AnnoData(data_dir)
        self.ent2_list = anno_data.entities_in_kg2()

        self.sampling_iteration = 0

    def prepare_data(self):
        if not os.path.exists(self.bach_recog_data_dir):
            os.mkdir(self.bach_recog_data_dir)
        train_valid_anno_links = read_links(os.path.join(self.division_dir, "train_valid_anno_links"))
        test_anno_links = read_links(os.path.join(self.division_dir, "test_anno_links"))
        train_num = int(0.8 * len(train_valid_anno_links))
        train_anno_links = train_valid_anno_links[:train_num]
        valid_anno_links = train_valid_anno_links[train_num:]
        ea_train_links = [link for link in train_anno_links if link[1]!="null"]
        ea_valid_links = [link for link in valid_anno_links if link[1] != "null"]
        ea_test_links = [link for link in test_anno_links if link[1] != "null"]
        save_links(ea_train_links, os.path.join(self.bach_recog_data_dir, "train_links"))
        save_links(ea_valid_links, os.path.join(self.bach_recog_data_dir, "valid_links"))
        save_links(ea_test_links, os.path.join(self.bach_recog_data_dir, "test_links"))
        save_links(train_anno_links, os.path.join(self.bach_recog_data_dir, "train_anno_links"))
        save_links(valid_anno_links, os.path.join(self.bach_recog_data_dir, "valid_anno_links"))
        save_links(test_anno_links, os.path.join(self.bach_recog_data_dir, "test_anno_links"))


    def train_ea_model(self, ):
        t = time.time()
        remove_unlinked = False
        if self.args.embedding_module == "RSN4EA":
            remove_unlinked = True

        kgs = read_kgs_from_folder(self.args.training_data, self.args.dataset_division, self.args.alignment_module,
                                   self.args.ordered, remove_unlinked=remove_unlinked)
        # if self.model is None:
        #     tf.reset_default_graph()
        #     self.model = get_model(self.args.embedding_module)()
        #     self.model.set_args(self.args)
        #     self.model.set_kgs(kgs)
        #     self.model.init()
        #     # self.model.restore()
        # else:
        #     self.model.set_kgs(kgs)
        #     self.model.ref_ent1 = kgs.valid_entities1 + kgs.test_entities1
        #     self.model.ref_ent2 = kgs.valid_entities2 + kgs.test_entities2
        #     self.model.reset_early_stop()

        tf.reset_default_graph()
        self.model = get_model(self.args.embedding_module)()
        self.model.set_args(self.args)
        self.model.set_kgs(kgs)
        self.model.init()
        self.model.valid("hits1")

        print("train model with all data")
        self.model.run()
        # self.model.test()
        self.model.save()
        print("Total run time = {:.3f} s.".format(time.time() - t))
        del self.model
        self.model = None


    def predict_ea(self, ent1_list, ent2_list):
        ent_embs = load_embeddings(os.path.join(self.model_dir, "ent_embeds.npy"))
        kg1_ent_ids = read_dict(os.path.join(self.model_dir, "kg1_ent_ids"))
        kg2_ent_ids = read_dict(os.path.join(self.model_dir, "kg2_ent_ids"))
        ent1_id_list = [kg1_ent_ids[uri] for uri in ent1_list]
        ent2_id_list = [kg2_ent_ids[uri] for uri in ent2_list]
        ent1_embs = ent_embs[ent1_id_list]
        ent2_embs = ent_embs[ent2_id_list]
        simi_mtx = np.matmul(ent1_embs, np.transpose(ent2_embs))
        return simi_mtx

    def _load_data(self):
        train_anno_links = read_links(os.path.join(self.bach_recog_data_dir, "train_anno_links"))
        valid_anno_links = read_links(os.path.join(self.bach_recog_data_dir, "valid_anno_links"))
        test_anno_links = read_links(os.path.join(self.bach_recog_data_dir, "test_anno_links"))
        train_data = [(link[0], int(link[1]!="null")) for link in train_anno_links]
        valid_data = [(link[0], int(link[1]!="null")) for link in valid_anno_links]
        test_data = [(link[0], int(link[1]!="null")) for link in test_anno_links]
        return train_data, valid_data, test_data

    def train_calibration(self):
        train_data, valid_data, test_data = self._load_data()


        plt.figure()

        def draw(data, prefix):
            valid_df = pd.DataFrame(data=data, columns=["ent1", "label"])
            ent1_list = valid_df["ent1"].tolist()
            simi_mtx = self.predict_ea(ent1_list, self.ent2_list)
            max_simi_arr = simi_mtx.max(axis=1)
            valid_df["max_simi"] = max_simi_arr

            max_simi = valid_df["max_simi"].max()
            min_simi = valid_df["max_simi"].min()
            step = (max_simi - min_simi) / 20
            simi_ranges = list(np.arange(min_simi, max_simi, step))

            macro_f1_list = []
            micro_f1_list = []
            c0_recall_list = []
            c1_recall_list = []
            for thr in simi_ranges:
                pred = (valid_df["max_simi"] > thr).astype(np.int)
                plist, rlist, f1list, _ = precision_recall_fscore_support(y_true=valid_df["label"], y_pred=pred, average=None)
                p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(y_true=valid_df["label"], y_pred=pred, average="macro")
                p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(y_true=valid_df["label"], y_pred=pred, average="micro")
                c0_recall_list.append(rlist[0])
                c1_recall_list.append(rlist[1])
                macro_f1_list.append(f1_ma)
                micro_f1_list.append(f1_mi)


            # plt.plot(simi_ranges, c0_recall_list, label=prefix+"-c0_recall")
            # plt.plot(simi_ranges, c1_recall_list, label=prefix+"-c1_recall")
            # plt.plot(simi_ranges, ma_f1_list, label=prefix+"-ma_f1")
            plt.plot(simi_ranges, micro_f1_list, label=prefix+"-micro_f1")

            max_idx = np.argmax(micro_f1_list)
            max_thr = simi_ranges[max_idx]
            return max_thr

        draw(train_data, "train")
        thr = draw(valid_data, "valid")
        draw(test_data, "test")

        plt.plot([thr, thr], [0, 1], color="grey", linestyle="dotted", label="threshold")
        plt.xlabel("Similarity score")
        plt.ylabel("Performance of bachelor recognizer")
        plt.title(f"Annotation percentage: {self.division}")

        plt.legend()
        plt.savefig(os.path.join(self.model_dir, "figures_%03d.png" % self.sampling_iteration))

        with open(os.path.join(self.model_dir, "thr.txt"), "w+") as file:
            file.write(f"{thr}")

    def update_model(self, sampling_iteration=0):
        print("##===> update bachelor recognizer")
        self.sampling_iteration = sampling_iteration
        self.prepare_data()
        self.train_ea_model()
        self.train_calibration()

    def recognize(self, ent1_list):
        if not os.path.exists(os.path.join(self.model_dir, "thr.txt")):
            print("have not got a bachelor recognition model. All entities will be thought as normal.")
            pred = np.ones(shape=len(ent1_list), dtype=np.float)
        else:
            with open(os.path.join(self.model_dir, "thr.txt")) as file:
                thr = eval(file.read())
            simi_mtx = self.predict_ea(ent1_list, self.ent2_list)
            max_simi_arr = simi_mtx.max(axis=1)
            pred = (max_simi_arr > thr).astype(float)

        ent1_to_having_mate = dict(list(zip(ent1_list, pred)))
        return ent1_to_having_mate

    def decision_function(self, ent1_arr: np.ndarray):
        ent1_list = list(ent1_arr.reshape(-1))
        simi_mtx = self.predict_ea(ent1_list, self.ent2_list)
        max_simi_arr = simi_mtx.max(axis=1)
        return max_simi_arr





def generate_data_for_bach_recog():
    data_dir = "/home/uqbliu3/experiments/al4ea_code/dataset/bachelor_issue/D_W_15K_V2_BACH0.3_BootEA/STRUCT_UNCER_SIMI_THR_BACH_RECOG_0.1"
    division = "0.20"
    percentage = 0.20
    sel_num = int(15000 * percentage)
    # sel_num = 100
    anno_links = read_links(os.path.join(data_dir, "annotations.txt"))
    train_valid_anno_links = anno_links[:sel_num]

    anno_data = AnnoData(data_dir)
    oracle_links = anno_data.links_of_ent1()
    test_anno_links = list(set(oracle_links).difference(set(train_valid_anno_links)))

    save_links(train_valid_anno_links, os.path.join(data_dir, division, "train_valid_anno_links"))
    save_links(test_anno_links, os.path.join(data_dir, division, "test_anno_links"))


def evaluate_gcn_simi():
    data_dir = "/home/uqbliu3/experiments/al4ea_code/dataset/bachelor_issue/D_W_15K_V2_BACH0.3_BootEA/STRUCT_UNCER_SIMI_THR_BACH_RECOG_0.1"
    division = "0.20"
    ea_arg_fn = "/home/uqbliu3/experiments/al4ea_code/openea_run/args/entity_classifier1_args_15K.json"
    bachelor_recog = GCNSimiThrBachRecog(data_dir=data_dir, division=division, arg_fn=ea_arg_fn)
    bachelor_recog.prepare_data()
    bachelor_recog.train_ea_model()
    bachelor_recog.train_calibration()


if __name__ == "__main__":
    # generate_data_for_bach_recog()
    evaluate_gcn_simi()




