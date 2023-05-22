# coding=utf-8
from alg.algs.ERM import ERM
from alg.algs.SCSN import SCSN

ALGORITHMS = [
    'ERM',
    'SCSN'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
