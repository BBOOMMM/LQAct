from collections import defaultdict
from collections.abc import Callable

from torch import Tensor

from ..linalg import *
from ..utils.threading import TaskProcessor
from ..utils.weakref import WeakHashKeyDictionary


COMPRESS_FUNC_MAPPING: dict[str, dict[str, Callable[..., tuple[Tensor, ...]]]] = {
    "lowrank": {
        "rqb": randomized_qb,
        "energy_rqb": energy_randomized_qb,
        "probing_rqb": probing_qb,
        "tsvd": truncated_svd,
        "rsvd": randomized_svd,
        "nyssvd": nystrom_svd,
    },
}

RECONSTRUCT_FUNC_MAPPING: dict[str, dict[str, Callable[..., Tensor]]] = {
    "lowrank": {
        "rqb": qb_reconstruct,
        "energy_rqb": energy_qb_reconstruct,
        "probing_rqb": probing_qb_reconstruct,
        "tsvd": svd_reconstruct,
        "rsvd": svd_reconstruct,
        "nyssvd": svd_reconstruct,
    },
}


_compress_cache = defaultdict(WeakHashKeyDictionary)
_quant_cache = defaultdict(WeakHashKeyDictionary)
_compress_processor = TaskProcessor()


def get_compress_cache():
    return _compress_cache


def get_quant_cache():
    return _quant_cache


def get_compress_processor():
    return _compress_processor
