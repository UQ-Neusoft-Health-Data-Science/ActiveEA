# -*- coding: utf-8 -*-


from al4ea.bachelor_recognizer.gcn_simi_thr import GCNSimiThrBachRecog
import os
from al4ea.reader import save_links, read_links
from openea.modules.args.args_hander import load_args
from openea.modules.load.read import load_embeddings, read_dict
from tqdm import trange
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt
import numpy as np
import json


class GCNSimiThrBachRecogCV(GCNSimiThrBachRecog):
    def __init__(self, data_dir, division, arg_fn):
        super(GCNSimiThrBachRecogCV, self).__init__(data_dir, division, arg_fn)

    def train_ea_model_with_cv(self):
        bach_recog_dir = os.path.join(self.division_dir, "bach_recog_cv")
        if not os.path.exists(bach_recog_dir):
            os.mkdir(bach_recog_dir)

        cv_num = 5

        train_valid_anno_links = read_links(os.path.join(self.division_dir, "train_valid_anno_links"))
        test_anno_links = read_links(os.path.join(self.division_dir, "test_anno_links"))
        div_size = int(len(train_valid_anno_links) / cv_num)

        for cv_idx in trange(0, cv_num):
            bach_recog_division = self.division + "/bach_recog_cv/div_%d/" % cv_idx
            self.bach_recog_data_dir = os.path.join(self.data_dir, bach_recog_division)
            self.model_dir = os.path.join(self.division_dir, "bach_recog_cv/model_%d/" % cv_idx)
            if not os.path.exists(self.model_dir):
                os.makedirs(self.model_dir)
            self.args = load_args(self.arg_fn)
            self.args.training_data = self.data_dir if self.data_dir.endswith("/") else self.data_dir + "/"
            self.args.dataset_division = bach_recog_division if bach_recog_division.endswith(
                "/") else bach_recog_division + "/"
            self.args.output = self.model_dir if self.model_dir.endswith("/") else self.model_dir + "/"
            if self.model is not None:
                del self.model
                self.model = None


            # prepare data
            if not os.path.exists(self.bach_recog_data_dir):
                os.mkdir(self.bach_recog_data_dir)

            valid_anno_links = train_valid_anno_links[cv_idx * div_size:(cv_idx + 1) * div_size]
            train_anno_links = train_valid_anno_links[:cv_idx * div_size] + train_valid_anno_links[(cv_idx + 1) * div_size:]
            ea_train_links = [link for link in train_anno_links if link[1]!="null"]
            ea_valid_links = [link for link in valid_anno_links if link[1] != "null"]
            ea_test_links = [link for link in test_anno_links if link[1] != "null"]
            save_links(ea_train_links, os.path.join(self.bach_recog_data_dir, "train_links"))
            save_links(ea_valid_links, os.path.join(self.bach_recog_data_dir, "valid_links"))
            save_links(ea_test_links, os.path.join(self.bach_recog_data_dir, "test_links"))
            save_links(train_anno_links, os.path.join(self.bach_recog_data_dir, "train_anno_links"))
            save_links(valid_anno_links, os.path.join(self.bach_recog_data_dir, "valid_anno_links"))
            save_links(test_anno_links, os.path.join(self.bach_recog_data_dir, "test_anno_links"))

            # train model
            self.train_ea_model()
            self.train_calibration()

    def predict_ea_with_cv(self, ent1_list, ent2_list, cv_idx=0):
        bach_recog_division = self.division + "/bach_recog_cv/div_%d/" % cv_idx
        self.bach_recog_data_dir = os.path.join(self.data_dir, bach_recog_division)
        self.model_dir = os.path.join(self.division_dir, "bach_recog_cv/model_%d/" % cv_idx)
        self.args = load_args(self.arg_fn)
        self.args.training_data = self.data_dir if self.data_dir.endswith("/") else self.data_dir + "/"
        self.args.dataset_division = bach_recog_division if bach_recog_division.endswith(
            "/") else bach_recog_division + "/"
        self.args.output = self.model_dir if self.model_dir.endswith("/") else self.model_dir + "/"
        if self.model is not None:
            del self.model
            self.model = None

        simi_mtx = self.predict_ea(ent1_list, ent2_list)
        max_simi_arr = simi_mtx.max(axis=1)
        return max_simi_arr


    def train_calibration_with_cv(self):
        perf_logs = {"sampling_iteration": self.sampling_iteration}
        valid_df_list = []
        test_simi_arr_list = []
        for cv_idx in range(5):
            bach_recog_division = self.division + "/bach_recog_cv/div_%d/" % cv_idx
            self.bach_recog_data_dir = os.path.join(self.data_dir, bach_recog_division)
            # self.model_dir = os.path.join(self.division_dir, "bach_recog_cv/model_%d/" % cv_idx)
            # self.args = load_args(self.arg_fn)
            # self.args.training_data = self.data_dir if self.data_dir.endswith("/") else self.data_dir + "/"
            # self.args.dataset_division = bach_recog_division if bach_recog_division.endswith(
            #     "/") else bach_recog_division + "/"
            # self.args.output = self.model_dir if self.model_dir.endswith("/") else self.model_dir + "/"
            # if self.model is not None:
            #     del self.model
            #     self.model = None
            #
            train_data, valid_data, test_data = self._load_data()
            valid_df = pd.DataFrame(data=valid_data, columns=["ent1", "label"])
            ent1_list = valid_df["ent1"].tolist()
            # simi_mtx = self.predict_ea(ent1_list, self.ent2_list)
            # max_simi_arr = simi_mtx.max(axis=1)

            max_simi_arr = self.predict_ea_with_cv(ent1_list, self.ent2_list, cv_idx)
            valid_df["max_simi"] = max_simi_arr

            test_df = pd.DataFrame(data=test_data, columns=["ent1", "label"])
            ent1_list = test_df["ent1"].tolist()
            # simi_mtx = self.predict_ea(ent1_list, self.ent2_list)
            # max_simi_arr = simi_mtx.max(axis=1)
            max_simi_arr = self.predict_ea_with_cv(ent1_list, self.ent2_list, cv_idx)
            test_simi_arr_list.append(max_simi_arr)

            valid_df_list.append(valid_df)

        valid_df = pd.concat(valid_df_list, axis=0, ignore_index=True)
        _, _, test_data = self._load_data()
        test_df = pd.DataFrame(data=test_data, columns=["ent1", "label"])
        test_max_simi_arr = np.mean(test_simi_arr_list, axis=0)
        test_df["max_simi"] = test_max_simi_arr

        def draw(data_df, prefix):

            max_simi = data_df["max_simi"].max()
            min_simi = data_df["max_simi"].min()
            step = (max_simi - min_simi) / 20
            simi_ranges = list(np.arange(min_simi, max_simi, step))

            macro_f1_list = []
            micro_f1_list = []
            c0_recall_list = []
            c1_recall_list = []
            for thr in simi_ranges:
                pred = (data_df["max_simi"] > thr).astype(np.int)
                plist, rlist, f1list, _ = precision_recall_fscore_support(y_true=data_df["label"], y_pred=pred, average=None)
                p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(y_true=data_df["label"], y_pred=pred, average="macro")
                p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(y_true=data_df["label"], y_pred=pred, average="micro")
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

        def measure_bach_perf(data_df, simi_ranges):
            metric_list = []
            for thr in simi_ranges:
                pred = (data_df["max_simi"] > thr).astype(np.int)
                prfs = precision_recall_fscore_support(y_true=data_df["label"], y_pred=pred,
                                                                          average=None)
                macro_prfs = precision_recall_fscore_support(y_true=data_df["label"], y_pred=pred,
                                                                       average="macro")
                micro_prfs = precision_recall_fscore_support(y_true=data_df["label"], y_pred=pred,
                                                                       average="micro")
                prf = [list(v) for v in prfs[:3]]
                metric_list.append((thr, prf, macro_prfs[:3], micro_prfs[:3]))
            return metric_list


        max_simi = valid_df["max_simi"].max()
        min_simi = valid_df["max_simi"].min()
        step = (max_simi - min_simi) / 20
        simi_ranges = list(np.arange(min_simi, max_simi, step))

        v_metric_list = measure_bach_perf(valid_df, simi_ranges)
        t_metric_list = measure_bach_perf(test_df, simi_ranges)
        perf_logs["ave"] = {"valid_metrics": v_metric_list, "test_metrics": t_metric_list}


        plt.figure()
        thr = draw(valid_df, "valid")
        draw(test_df, "test")

        plt.plot([thr, thr], [0, 1], color="grey", linestyle="dotted", label="threshold")
        plt.xlabel("Similarity score")
        plt.ylabel("Performance of bachelor recognizer")
        plt.title(f"Annotation percentage: {self.division}")

        plt.legend()
        plt.savefig(os.path.join(self.division_dir, "bach_recog_cv/figures_ensemble_%03d.png" % self.sampling_iteration))

        with open(os.path.join(self.division_dir, "bach_recog_cv/thr.txt"), "w+") as file:
            file.write(f"{thr}")

        # log performance details
        for idx in range(len(valid_df_list)):
            valid_df = valid_df_list[idx]
            test_max_simi_arr = test_simi_arr_list[idx]
            test_df["max_simi"] = test_max_simi_arr

            v_metric_list = measure_bach_perf(valid_df, simi_ranges)
            t_metric_list = measure_bach_perf(test_df, simi_ranges)
            perf_logs[f"cv_{idx}"] = {"valid_metrics": v_metric_list, "test_metrics": t_metric_list}
        with open(os.path.join(self.division_dir, "bach_recog_cv", "bach_perf_detail_logs.json"), "a+") as file:
            file.write(json.dumps(perf_logs) + "\n")


    def update_model(self, sampling_iteration=0):
        self.sampling_iteration = sampling_iteration
        self.train_ea_model_with_cv()
        self.train_calibration_with_cv()

    def recognize(self, ent1_list):
        if not os.path.exists(os.path.join(self.model_dir, "thr.txt")):
            print("have not got a bachelor recognition model. All entities will be thought as normal.")
            pred = np.ones(shape=len(ent1_list), dtype=np.float)
        else:
            with open(os.path.join(self.division_dir, "bach_recog_cv/thr.txt")) as file:
                thr = eval(file.read())

            max_simi_arr_list = []
            for cv_idx in range(5):
                max_simi_arr = self.predict_ea_with_cv(ent1_list, self.ent2_list, cv_idx)
                max_simi_arr_list.append(max_simi_arr)

            max_simi_arr = np.mean(max_simi_arr_list, axis=0)
            pred = (max_simi_arr > thr).astype(float)

        ent1_to_having_mate = dict(list(zip(ent1_list, pred)))
        return ent1_to_having_mate




def evaluate_gcn_simi_cv():
    data_dir = "/home/uqbliu3/experiments/al4ea_code/dataset/bachelor_issue/D_W_15K_V2_BACH0.3_Alinet/STRUCT_UNCER_SIMI_THR_BACH_RECOG_0.1"
    division = "0.00"
    ea_arg_fn = "/home/uqbliu3/experiments/al4ea_code/openea_run/args/entity_classifier1_args_15K.json"
    bachelor_recog = GCNSimiThrBachRecogCV(data_dir=data_dir, division=division, arg_fn=ea_arg_fn)
    bachelor_recog.update_model()


if __name__ == "__main__":

    evaluate_gcn_simi_cv()
