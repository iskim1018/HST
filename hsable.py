import numpy as np
from abc import ABC, abstractmethod


class HSable(ABC):
    @abstractmethod
    def get_centroid(self):
        pass

    @abstractmethod
    def get_new_centroid(self, vec: np.ndarray):
        pass

    @abstractmethod
    def get_radius(self):
        pass

    @abstractmethod
    def get_dist_max(self, v: np.ndarray):
        pass
