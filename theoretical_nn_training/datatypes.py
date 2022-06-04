from typing import NamedTuple


class Resolution(NamedTuple):
    phi: int
    Nw: int
    eta_sp: int = 0


class Param(NamedTuple):
    min: float
    max: float
    mu: float = 0
    sigma: float = 0
    alpha: float = 0
    beta: float = 0
