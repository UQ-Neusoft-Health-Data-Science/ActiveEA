# -*- coding: utf-8 -*-





from openea.modules.args.args_hander import load_args
from openea.modules.load.read import load_embeddings, read_dict
import numpy as np
import os


class EAModel:
    def __init__(self, data_dir, division, ea_arg_fn, model_dir):
        self.model_dir = model_dir
        self.args = load_args(ea_arg_fn)
        self.args.training_data = data_dir if data_dir.endswith("/") else data_dir + "/"
        self.args.dataset_division = division if division.endswith("/") else division + "/"
        self.args.output = self.model_dir if self.model_dir.endswith("/") else self.model_dir + "/"

        self.use_mc_dropout = ("dropout" in self.args.__dict__) and (self.args.dropout > 0)

    def predict(self, ent1_list, ent2_list):
        if self.use_mc_dropout:
            simi_mtx_list = []
            for i in range(10):
                sub_dir = os.path.join(self.model_dir, "model%02d" % i)
                ent_embs = load_embeddings(os.path.join(sub_dir, "ent_embeds.npy"))
                kg1_ent_ids = read_dict(os.path.join(sub_dir, "kg1_ent_ids"))
                kg2_ent_ids = read_dict(os.path.join(sub_dir, "kg2_ent_ids"))
                ent1_id_list = [kg1_ent_ids[uri] for uri in ent1_list]
                ent2_id_list = [kg2_ent_ids[uri] for uri in ent2_list]
                ent1_embs = ent_embs[ent1_id_list]
                ent2_embs = ent_embs[ent2_id_list]
                simi_mtx = np.matmul(ent1_embs, np.transpose(ent2_embs))
                simi_mtx_list.append(simi_mtx)
            simi_mtx = np.mean(simi_mtx_list, axis=0)
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




