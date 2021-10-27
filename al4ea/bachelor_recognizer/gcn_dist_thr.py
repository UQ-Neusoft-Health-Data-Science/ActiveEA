# -*- coding: utf-8 -*-



from openea.modules.load.read import load_embeddings, read_dict
import numpy as np
import os
from al4ea.bachelor_recognizer.gcn_simi_thr import GCNSimiThrBachRecog
from tqdm import trange


class GCNDistThrBachRecog(GCNSimiThrBachRecog):
    def __init__(self, data_dir, division, arg_fn):
        super(GCNDistThrBachRecog, self).__init__(data_dir, division, arg_fn)

    def predict_ea(self, ent1_list, ent2_list):
        ent_embs = load_embeddings(os.path.join(self.model_dir, "ent_embeds.npy"))
        kg1_ent_ids = read_dict(os.path.join(self.model_dir, "kg1_ent_ids"))
        kg2_ent_ids = read_dict(os.path.join(self.model_dir, "kg2_ent_ids"))
        ent1_id_list = [kg1_ent_ids[uri] for uri in ent1_list]
        ent2_id_list = [kg2_ent_ids[uri] for uri in ent2_list]
        ent1_embs = ent_embs[ent1_id_list]
        ent2_embs = ent_embs[ent2_id_list]
        # simi_mtx = np.matmul(ent1_embs, np.transpose(ent2_embs))
        dist_list = []
        for idx in trange(0, len(ent1_embs), 1000):
            ent1_embs_slice = ent1_embs[idx: idx+1000]
            delta = np.expand_dims(ent1_embs_slice, axis=1) - np.expand_dims(ent2_embs, axis=0)
            dist = np.linalg.norm(delta, axis=-1)
            dist_list.append(dist)
        dist = np.concatenate(dist_list, axis=0)
        simi_mtx = - dist
        return simi_mtx



