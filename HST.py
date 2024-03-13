from __future__ import annotations
import numpy as np
import pickle

from HS import HS


class HST:
    def __init__(self, n_dim: int, n_max_child: int):
        self.n_dim = n_dim
        self.n_max_child = n_max_child
        self.RHS = None
        pass

    def add(self, vec: np.ndarray):
        if self.RHS is None:
            self.RHS = HS(self, None, [vec])
        else:
            self.RHS.add(vec)

    def show(self):
        self.RHS.show("")

    @staticmethod
    def load(path) -> HST:
        with open(path, 'rb') as file:
            return pickle.load(file)

    def save(self, path):
        with open(path, 'wb') as file:
            pickle.dump(self, file)

    def __repr__(self):
        return "HST"
