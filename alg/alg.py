# coding=utf-8
from alg.algs.Our import Our
from alg.algs.ERM import ERM
from alg.algs.Fixed import Fixed
from alg.algs.CORAL import CORAL
from alg.algs.RSC import RSC
from alg.algs.DANN import DANN
from alg.algs.ANDMask import ANDMask
from alg.algs.GroupDRO import GroupDRO
from alg.algs.Mixup import Mixup


ALGORITHMS = [
    'Our',
    'ERM',
    'Fixed',
    'CORAL',
    'RSC',
    'DANN',
    'ANDMask',
    'GroupDRO',
    'Mixup',
]


def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
