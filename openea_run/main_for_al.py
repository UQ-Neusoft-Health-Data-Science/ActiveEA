import argparse
import sys
import time

from openea.modules.args.args_hander import check_args, load_args
from openea.modules.load.kgs import read_kgs_from_folder
from openea.models.trans import TransD
from openea.models.trans import TransE
from openea.models.trans import TransH
from openea.models.trans import TransR
from openea.models.semantic import DistMult
from openea.models.semantic import HolE
from openea.models.semantic import SimplE
from openea.models.semantic import RotatE
from openea.models.neural import ConvE
from openea.models.neural import ProjE
from openea.approaches import AlignE
from openea.approaches import BootEA
from openea.approaches import JAPE
from openea.approaches import Attr2Vec
from openea.approaches import MTransE
from openea.approaches import IPTransE
from openea.approaches import GCN_Align
from openea.approaches import AttrE
from openea.approaches import IMUSE
from openea.approaches import SEA
from openea.approaches import MultiKE
from openea.approaches import RSN4EA
from openea.approaches import GMNN
from openea.approaches import KDCoE
from openea.approaches import RDGCN
from openea.approaches import BootEA_RotatE
from openea.approaches import BootEA_TransH
from openea.approaches import AliNet
from openea.models.basic_model import BasicModel
from openea.approaches.entity_classifier import EntityClassifier
from openea.approaches.entity_classifier1 import EntityClassifier1
import tensorflow as tf

class ModelFamily(object):
    BasicModel = BasicModel

    TransE = TransE
    TransD = TransD
    TransH = TransH
    TransR = TransR

    DistMult = DistMult
    HolE = HolE
    SimplE = SimplE
    RotatE = RotatE

    ProjE = ProjE
    ConvE = ConvE

    MTransE = MTransE
    IPTransE = IPTransE
    Attr2Vec = Attr2Vec
    JAPE = JAPE
    AlignE = AlignE
    BootEA = BootEA
    GCN_Align = GCN_Align
    GMNN = GMNN
    KDCoE = KDCoE

    AttrE = AttrE
    IMUSE = IMUSE
    SEA = SEA
    MultiKE = MultiKE
    RSN4EA = RSN4EA
    RDGCN = RDGCN
    BootEA_RotatE = BootEA_RotatE
    BootEA_TransH = BootEA_TransH
    AliNet = AliNet

    EntityClassifier = EntityClassifier
    EntityClassifier1 = EntityClassifier1


def get_model(model_name):
    return getattr(ModelFamily, model_name)


if __name__ == '__main__':
    t = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--arg_fn", type=str, help="file path of config file")
    ap.add_argument("--data_dir", type=str, help="directory of data")
    ap.add_argument("--division", type=str, help="division")
    ap.add_argument("--out_dir", type=str, help="directory of output")
    ap_args, _ = ap.parse_known_args()

    args = load_args(ap_args.arg_fn)
    args.training_data = ap_args.data_dir if ap_args.data_dir.endswith("/") else ap_args.data_dir+"/"
    args.dataset_division = ap_args.division if ap_args.division.endswith("/") else ap_args.division+"/"
    args.output = ap_args.out_dir if ap_args.out_dir else ap_args.out_dir+"/"
    print(args.embedding_module)
    print(args)
    remove_unlinked = False
    if args.embedding_module == "RSN4EA":
        remove_unlinked = True
    kgs = read_kgs_from_folder(args.training_data, args.dataset_division, args.alignment_module, args.ordered,
                               remove_unlinked=remove_unlinked)
    model = get_model(args.embedding_module)()
    model.set_args(args)
    model.set_kgs(kgs)
    model.init()
    # model.restore()
    model.run()
    model.save()
    model.test()
    print("Total run time = {:.3f} s.".format(time.time() - t))