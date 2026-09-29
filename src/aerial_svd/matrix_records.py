"""Framework-independent analysis output schemas."""
from dataclasses import asdict, dataclass
from typing import Tuple, Sequence


@dataclass
class MatrixRecord:
    name: str
    kind: str
    shape: Tuple[int, int]
    original_shape: Tuple[int, ...]
    params: int


@dataclass
class SpectralRecord:
    model: str
    checkpoint: str
    name: str
    kind: str
    m: int
    n: int
    params: int
    sigma_max: float
    sigma_min: float
    sigma_min_eps: float
    cond_eps: float
    fro_norm: float
    spectral_norm: float
    stable_rank: float
    numerical_rank_eps: int
    r90: int
    r95: int
    r99: int
    r90_ratio: float
    r95_ratio: float
    r99_ratio: float


@dataclass
class ApproximationRecord:
    model: str
    checkpoint: str
    name: str
    kind: str
    m: int
    n: int
    rank: int
    rank_ratio: float
    dense_storage: int
    lowrank_storage: int
    storage_ratio: float
    fro_error: float
    spectral_error: float
    retained_energy: float


@dataclass
class RandomizedSVDRecord:
    model: str
    checkpoint: str
    name: str
    kind: str
    m: int
    n: int
    rank: int
    oversampling: int
    n_iter: int
    full_svd_time_sec: float
    randomized_svd_time_sec: float
    sigma_relative_error: float
    approximation_fro_error: float
    randomized_fro_error: float



def dataclass_records_to_dicts(records: Sequence[object]) -> list[dict]:
    return [asdict(r) for r in records]
