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

    @staticmethod
    def get_dist(vec1: np.ndarray, vec2: np.ndarray):
        return np.linalg.norm(vec1 - vec2)

    def search(self, vec: np.ndarray):
        return self.RHS.search(vec)

    def search_dfs(self, vec: np.ndarray):
        return self.RHS.search_dfs(vec)

    def get_search_rank(self, vec: np.ndarray):
        vec_searched = self.search(vec)
        dist_searched = self.get_dist(vec, vec_searched)
        n_closer_vecs = self.RHS.search_n_closer_vecs(vec, dist_searched)
        return n_closer_vecs + 1

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
