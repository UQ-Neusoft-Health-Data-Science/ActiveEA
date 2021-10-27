# -*- coding: utf-8 -*-



import os
import json
import pandas as pd
import numpy as np
import argparse
from al4ea.reader import load_al_settings


def gather_metrics(out_dir, gather_metric_fn):
    hit_list = []
    ndcg_list = []
    ratio_list = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    for ratio in ratio_list:
        dir_name = "res_%.2f" % ratio
        res_dir = os.path.join(out_dir, dir_name)
        metric_fn = os.path.join(res_dir, "metrics.json")
        with open(metric_fn) as file:
            metrics = json.loads(file.read())
            hit_list.append(metrics["cosine"]["hits"])
            ndcg_list.append(metrics["cosine"]["ndcgs"])
    topk = [1, 5, 10, 50]
    hit_arr = np.array(hit_list).transpose()
    ndcg_arr = np.array(ndcg_list).transpose()
    metrics_arr = np.concatenate([hit_arr, ndcg_arr], axis=0)
    metrics_comparison = pd.DataFrame(data=metrics_arr, columns=ratio_list, index=["hit@%d"%k for k in topk]+["ndcg@%d"%k for k in topk])
    metrics_comparison.to_csv(gather_metric_fn)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, help="directory of data")
    ap.add_argument("--metric_fn", type=str, help="file path of metric")
    args, _ = ap.parse_known_args()
    out_dir = args.out_dir
    gather_metric_fn = args.metric_fn
    gather_metrics(out_dir, gather_metric_fn)







