import numpy as np

from hsable import HSable

detailed_str = False


class Vector(HSable):
    def __init__(self, vid: int, vec: np.ndarray):
        self.vid = vid
        self.v = vec

    def get_centroid(self):
        return self.v

    def get_new_centroid(self, vec: np.ndarray):
        return np.mean([self.v, vec], axis=0)

    def get_radius(self):
        return 0

    def __str__(self):
        if detailed_str:
            return f"<{self.vid}>{self.v}"
        return f"<{self.vid}>"
