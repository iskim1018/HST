from __future__ import annotations
import numpy as np
import pickle

import vector
from HS import HS
from vector import Vector
from hst_stat import HSTStat


class HST:
    def __init__(self, n_dim: int, n_max_child: int):
        self.n_dim = n_dim
        self.n_max_child = n_max_child
        self.RHS = None
        self.vid_start = 1

    def assign_vector(self, v: np.ndarray):
        vid = self.vid_start
        self.vid_start += 1
        return Vector(vid, v)

    def insert(self, v: np.ndarray):
        vec = self.assign_vector(v)
        if self.RHS is None:
            self.RHS = HS(self, None, [vec])
        else:
            self.RHS.insert(vec)

    @staticmethod
    def get_dist(vec1: np.ndarray, vec2: np.ndarray):
        return np.linalg.norm(vec1 - vec2)

    def search_pn(self, vec: np.ndarray):
        return self.RHS.search(vec)

    def search_nn(self, vec: np.ndarray):
        return self.RHS.search_nn(vec)

    def get_dist_rank(self, v: np.ndarray, dist: float):
        n_closer_vecs = self.RHS.get_n_closer_vecs(v, dist)
        return n_closer_vecs + 1

    def get_rank_vec_dist(self, vec: np.ndarray, rank: int):
        return self.RHS.get_rank_vec_dist(vec, rank)

    def get_nn_vec_dist(self, vec: np.ndarray):
        return self.RHS.get_rank_vec_dist(vec, 1)

    def get_nn_vec(self, vec: np.ndarray):
        return self.RHS.search_nn(vec, 1)

    def get_nn_vecs(self, v: np.ndarray, dist_nn: float):
        return self.RHS.search_vecs_nn(v, dist_nn)

    def get_pn_vec(self, v: np.ndarray, dist_pn: float, stat: HSTStat = None):
        return self.RHS.search_pn(v, dist_pn, stat)

    def get_summary(self):
        return self.RHS.get_summary()

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
