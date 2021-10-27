# -*- coding: utf-8 -*-



import matplotlib.pyplot as plt
from al4ea.bachelor_recognizer.gcn_simi_thr import GCNSimiThrBachRecog
from al4ea.al_modules import Pool, Oracle, AnnoData
from al4ea.reader import read_links, save_links
import os
import pandas as pd
import numpy as np
from sklearn.calibration import calibration_curve


dataset_name = "D_W_15K_V2"
data_variant = "STRUCT_UNCER_SIMI_THR_BACH_RECOG_0.1"
model_name = "BootEA"
division = "0.20"
data_dir = "/home/uqbliu3/experiments/al4ea_code/dataset/bachelor_issue/%s_BACH0.3_%s/%s" % (dataset_name, model_name, data_variant)
bach_recog_arg_fn = "/home/uqbliu3/experiments/al4ea_code/openea_run/args/entity_classifier1_args_15K.json"

bach_recognizer = GCNSimiThrBachRecog(data_dir=data_dir, division=division, arg_fn=bach_recog_arg_fn)
oracle = Oracle(data_dir=data_dir)
anno_data = AnnoData(data_dir=data_dir)
ent1_list = anno_data.entities_in_kg1()
ent2_list = anno_data.entities_in_kg2()
anno_links = oracle.annotate(ent1_list)

simi_mtx = bach_recognizer.predict_ea(ent1_list, ent2_list)
max_simi_arr = simi_mtx.max(axis=1)
anno_link_df = pd.DataFrame(data=anno_links, columns=["ent1", "ent2"])
anno_link_df["max_simi"] = max_simi_arr
label_list = [int(link[1] != "null") for link in anno_links]
anno_link_df["label"] = label_list
anno_link_df.set_index(keys="ent1", drop=True, inplace=True)


train_anno_links = read_links(os.path.join(data_dir, division, "bachelor_recog_data/train_anno_links"))
valid_anno_links = read_links(os.path.join(data_dir, division, "bachelor_recog_data/valid_anno_links"))
test_anno_links = read_links(os.path.join(data_dir, division, "bachelor_recog_data/test_anno_links"))
train_ent_list = [link[0] for link in train_anno_links]
valid_ent_list = [link[0] for link in valid_anno_links]
test_ent_list = [link[0] for link in test_anno_links]
train_anno_link_df = anno_link_df.loc[train_ent_list]
valid_anno_link_df = anno_link_df.loc[valid_ent_list]
test_anno_link_df = anno_link_df.loc[test_ent_list]

#

max_simi = anno_link_df["max_simi"].max()
min_simi = anno_link_df["max_simi"].min()
step = (max_simi - min_simi) / 20
simi_ranges = list(np.arange(min_simi, max_simi, step))





def distri_of_ent():
    def draw_distri(link_df: pd.DataFrame, prefix):
        distri_of_ent_list = []
        for simi in simi_ranges:
            cond = (link_df["max_simi"] > simi) & (link_df["max_simi"] < (simi + step))
            slice_df = link_df[cond]
            percent_of_ent = len(slice_df) / len(link_df)
            distri_of_ent_list.append(percent_of_ent)
        plt.plot(simi_ranges, distri_of_ent_list, label=prefix + ": " + "distri_of_ent")

    plt.figure()
    draw_distri(anno_link_df.loc[train_ent_list], "train")
    draw_distri(anno_link_df.loc[valid_ent_list], "valid")
    draw_distri(anno_link_df.loc[test_ent_list], "test")
    plt.legend()
    plt.savefig(os.path.join(data_dir, division, "distri_of_ent.png"))



def percent_of_normal_ent():
    def draw_percentage_of_normal_ent(link_df: pd.DataFrame, prefix):
        percent_of_normal_ent_list = []
        effective_simi_range = []
        for simi in simi_ranges:
            cond = (link_df["max_simi"] > simi) & (link_df["max_simi"] < (simi + step))
            slice_df = link_df[cond]
            if len(slice_df) > 0:
                effective_simi_range.append(simi)
                percent_of_normal_ent = slice_df["label"].sum() / len(slice_df)
                percent_of_normal_ent_list.append(percent_of_normal_ent)
        plt.plot(effective_simi_range, percent_of_normal_ent_list, label=prefix + ": " + "per_of_normal_ent")

    plt.figure()
    draw_percentage_of_normal_ent(anno_link_df.loc[train_ent_list], "train")
    draw_percentage_of_normal_ent(anno_link_df.loc[valid_ent_list], "valid")
    draw_percentage_of_normal_ent(anno_link_df.loc[test_ent_list], "test")
    plt.legend()
    plt.savefig(os.path.join(data_dir, division, "percent_of_normal_ent.png"))


def calibrate_curve():
    # strategy = "quantile"
    strategy = "uniform"
    plt.figure()
    fop, mpv = calibration_curve(valid_anno_link_df["label"], valid_anno_link_df["max_simi"], n_bins=20, normalize=True, strategy=strategy)
    plt.plot(mpv, fop, marker='^', label="valid")
    fop, mpv = calibration_curve(test_anno_link_df["label"], test_anno_link_df["max_simi"], n_bins=20, normalize=True, strategy=strategy)
    plt.plot(mpv, fop, marker='.', label="test")
    plt.legend()
    plt.title(f"calibration curve ({strategy})")
    plt.savefig(os.path.join(data_dir, division, f"calibrate ({strategy}).png"))


def linear_calibrate():
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(anno_link_df["max_simi"].to_numpy().reshape((-1, 1)))
    valid_scale_simi = scaler.transform(valid_anno_link_df["max_simi"].to_numpy().reshape((-1, 1)))
    test_scale_simi = scaler.transform(test_anno_link_df["max_simi"].to_numpy().reshape((-1, 1)))

    plt.figure()
    fop, mpv = calibration_curve(valid_anno_link_df["label"], valid_scale_simi, n_bins=20, normalize=False)
    plt.plot(mpv, fop, marker='^', label="valid")
    fop, mpv = calibration_curve(test_anno_link_df["label"], test_scale_simi, n_bins=20, normalize=False)
    plt.plot(mpv, fop, marker='.', label="test")
    plt.legend()
    plt.title("linear calibration")
    plt.savefig(os.path.join(data_dir, division, "linear_calibrate.png"))


def sigmoid_calibrate():
    from sklearn.calibration import _SigmoidCalibration
    calibrator = _SigmoidCalibration()
    calibrator.fit(X=valid_anno_link_df["max_simi"], y=valid_anno_link_df["label"])

    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    yhat = calibrator.predict(valid_anno_link_df["max_simi"])
    fop, mpv = calibration_curve(valid_anno_link_df["label"], yhat, n_bins=20, normalize=False)
    plt.plot(mpv, fop, marker='^')
    yhat = calibrator.predict(test_anno_link_df["max_simi"])
    fop, mpv = calibration_curve(test_anno_link_df["label"], yhat, n_bins=20, normalize=False)
    plt.plot(mpv, fop, marker='.')
    plt.savefig(os.path.join(data_dir, division, "sigmoid_calibrate.png"))


def isotonic_calibrate():
    from sklearn.isotonic import IsotonicRegression


    valid_df = anno_link_df.loc[valid_ent_list]
    test_df = anno_link_df.loc[test_ent_list]

    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(X=valid_df["max_simi"], y=valid_df["label"])


    plt.figure()
    plt.plot([0, 1], [0, 1], linestyle='--')
    # plot model reliability
    yhat = calibrator.predict(valid_df["max_simi"])
    fop, mpv = calibration_curve(valid_df["label"], yhat, n_bins=20, normalize=False)
    plt.plot(mpv, fop, marker='^')
    yhat = calibrator.predict(test_df["max_simi"])
    fop, mpv = calibration_curve(test_df["label"], yhat, n_bins=20, normalize=False)
    plt.plot(mpv, fop, marker='.')
    plt.savefig(os.path.join(data_dir, division, "isotonic_calibrate.png"))




def visualize_recalls():

    plt.figure()

    def draw(unseen_ent1_list, prefix, color):
        unseen_ent1_pred_df = anno_link_df.loc[unseen_ent1_list]

        num_unseen_ent = len(unseen_ent1_pred_df)
        num_normal_ent = unseen_ent1_pred_df["label"].sum()
        num_bachelor = num_unseen_ent - num_normal_ent

        max_simi = unseen_ent1_pred_df["max_simi"].max()
        min_simi = unseen_ent1_pred_df["max_simi"].min()
        step = (max_simi - min_simi) / 20
        simi_ranges = list(np.arange(min_simi, max_simi, step))
        per_normal_ent_list = []
        per_of_ent_list = []
        recall_normal_ent_list = []
        prec_normal_ent_list = []
        f1_normal_ent_list = []
        recall_bachelor_ent_list = []
        for simi in simi_ranges:
            # cond = (unseen_ent1_pred_df["max_simi"]>simi) & (unseen_ent1_pred_df["max_simi"]<(simi+step))
            cond = unseen_ent1_pred_df["max_simi"] > simi
            ent_slice = unseen_ent1_pred_df[cond]
            num_normal_ent_in_slice = ent_slice["label"].sum()
            num_ent_in_slice = len(ent_slice)
            per_normal_ent = num_normal_ent_in_slice / num_ent_in_slice
            per_ent = num_ent_in_slice / num_unseen_ent
            recall_normal_ent = num_normal_ent_in_slice / num_normal_ent
            prec_normal_ent = num_normal_ent_in_slice / num_ent_in_slice
            f1_normal_ent = 2 * prec_normal_ent * recall_normal_ent / (prec_normal_ent + recall_normal_ent)
            recall_bachelor = 1 - (num_ent_in_slice - num_normal_ent_in_slice) / num_bachelor

            per_normal_ent_list.append(per_normal_ent)
            per_of_ent_list.append(per_ent)
            recall_normal_ent_list.append(recall_normal_ent)
            recall_bachelor_ent_list.append(recall_bachelor)
            prec_normal_ent_list.append(prec_normal_ent)
            f1_normal_ent_list.append(f1_normal_ent)


        # plt.plot(simi_ranges, per_normal_ent_list, linestyle="-.", marker="^", color=color, label=prefix+"percent of normal ent")
        plt.plot(simi_ranges, recall_normal_ent_list, linestyle="dotted",  color=color, label=prefix+"recall of normal ent")
        plt.plot(simi_ranges, recall_bachelor_ent_list, linestyle="dashed",  color=color, label=prefix+"recall of bachelor")
        # plt.plot([min_simi, max_simi], [0.6, 0.6], linestyle="--", color="grey", label="expected recall")


    # draw([link[0] for link in train_anno_links], "train: ", "blue")
    # draw([link[0] for link in valid_anno_links], "valid: ", "green")
    draw([link[0] for link in test_anno_links], "test: ", "red")

    plt.ylim(0, 1.1)
    plt.legend()
    plt.savefig(os.path.join(data_dir, division, "recalls.png"))

    def draw2(unseen_ent1_list, prefix, color):
        unseen_ent1_pred_df = anno_link_df.loc[unseen_ent1_list]
        fop, mpv = calibration_curve(y_true=unseen_ent1_pred_df["label"], y_prob=unseen_ent1_pred_df["max_simi"], normalize=True, n_bins=50)
        plt.plot(mpv, fop, color=color, label=prefix, marker=".")


    plt.figure()
    draw2([link[0] for link in train_anno_links], "train: ", "blue")
    draw2([link[0] for link in valid_anno_links], "valid: ", "green")
    draw2([link[0] for link in test_anno_links], "test: ", "red")
    plt.legend()
    plt.savefig(os.path.join(data_dir, division, "calibrate_curve.png"))



distri_of_ent()
percent_of_normal_ent()
calibrate_curve()
linear_calibrate()



