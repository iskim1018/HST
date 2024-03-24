from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING
from hsable import HSable

detailed_str = False

if TYPE_CHECKING:
    import HST


class Vector(HSable):
    def __init__(self, vid: int, hst: HST, vec: np.ndarray):
        self.vid = vid
        self.v = vec
        self.hst = hst

    def get_centroid(self):
        return self.v

    def get_new_centroid(self, v: np.ndarray):
        return np.mean([self.v, v], axis=0)

    def get_radius(self):
        return 0

    def get_dist_max(self, v: np.ndarray):
        return self.hst.dist_func(self.v, v)

    def __str__(self):
        if detailed_str:
            return f"<{self.vid}>{self.v}"
        return f"<{self.vid}>"
