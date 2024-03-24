from __future__ import annotations
import numpy as np
from typing import List, Union, TYPE_CHECKING

from hsable import HSable
from vector import Vector
from hst_stat import HSTStat

if TYPE_CHECKING:
    import HST

detailed_str = False


class HS(HSable):
    children: List[Union[HS, Vector]]

    def __init__(self, hid: int, hst: HST, hs_parent: Union[HS, None], children: List[Union[HS, Vector]]):
        self.hid = hid
        self.centroid = 0
        self.radius = 0
        self.dist_max = 0
        self.hst = hst
        self.hs_parent = hs_parent
        self.children = children
        for child in children:
            if isinstance(child, HS):
                child.hs_parent = self
        self.setup()

    def get_level(self):
        if self.hs_parent is None:
            return 1
        return self.hs_parent.get_level() + 1

    def _get_child_centroids(self):
        return list(map(lambda child: child.get_centroid(), self.children))

    def _get_child_radii(self):
        return list(map(lambda child: child.get_radius(), self.children))

    def _get_child_new_centroids(self, v: np.ndarray):
        return list(map(lambda child: child.get_new_centroid(v), self.children))

    def get_radius(self):
        return self.radius

    def get_centroid(self):
        return self.centroid

    def get_new_centroid(self, vec):
        vecs = self._get_child_centroids()
        vecs.append(vec)
        return np.mean(vecs, axis=0)

    def get_dist_max(self, v: np.ndarray):
        dist = self.hst.dist_func(self.centroid, v)
        return dist + self.dist_max

    def _get_nearest_child_idx(self, vec: Vector):
        vecs = self._get_child_new_centroids(vec.v)
        idx_min = None
        dist_min = None
        idx = 0
        for v in vecs:
            dist = self.hst.dist_func(v, vec.v)
            if idx_min is None or dist < dist_min:
                idx_min = idx
                dist_min = dist
            idx += 1
        return idx_min

    def _get_dists(self, vs, v):
        dists = []
        for vv in vs:
            dists.append(self.hst.dist_func(vv, v))
        return dists

    def _get_candidate_children(self, vec: np.ndarray, dist_pn: float):
        candidates = []
        vecs = self._get_child_centroids()
        dists = self._get_dists(vecs, vec)
        for child, dist_centroid in zip(self.children, dists):
            if isinstance(child, HS):
                if child.dist_max >= dist_centroid or dist_centroid - child.dist_max < dist_pn:
                    candidates.append((child, dist_centroid))
            else:
                if dist_centroid < dist_pn:
                    candidates.append((child, dist_centroid / 2))
        candidates = sorted(candidates, key=lambda x: x[1])
        return [x[0] for x in candidates]

    def add(self, vec: Vector):
        if len(self.children) >= self.hst.n_max_child:
            idx = self._get_nearest_child_idx(vec)
            child = self.children[idx]
            if isinstance(child, HS):
                child.insert(vec)
            else:
                hs_new = self.hst.alloc_hs(self, [child, vec])
                self.children[idx] = hs_new
        else:
            self.children.append(vec)
        self._setup_radius()

    def insert(self, vec: Vector):
        dist = self.hst.dist_func(self.centroid, vec.v)
        if dist < self.dist_max:
            self.add(vec)
            return

        if len(self.children) >= self.hst.n_max_child:
            idx = self._get_nearest_child_idx(vec)
            child = self.children[idx]
            hs_new = self.hst.alloc_hs(self, [child, vec])
            if isinstance(child, HS):
                child.hs_parent = hs_new
            self.children[idx] = hs_new
            self.setup()
        else:
            self.children.append(vec)
            self.setup()

    def remove(self, child_removing):
        self.children.remove(child_removing)
        self._setup_radius()

    def _get_nearest_candidate_hs(self, v: np.ndarray):
        vecs = self._get_child_new_centroids(v)
        dists = self._get_dists(vecs, v)
        for idx, dist in sorted(enumerate(dists), key=lambda x: x[1]):
            child = self.children[idx]
            if isinstance(child, HS) and child.dist_max >= dist:
                return child
        return None

    def reparent(self, child_from, child_grand):
        v = child_grand.centroid if isinstance(child_grand, HS) else child_grand.v
        child_nn = self._get_nearest_candidate_hs(v)
        if child_nn is None or child_from == child_nn:
            return
        child_from.remove(child_grand)
        child_nn.add(child_grand)

    def try_to_reparent(self):
        if self.hs_parent is None:
            return
        for child in self.children:
            self.hs_parent.reparent(self, child)

    def _setup_dist_max(self):
        dist_max = 0
        for child in self.children:
            dist_max_child = child.get_dist_max(self.centroid)
            if dist_max_child > dist_max:
                dist_max = dist_max_child
        self.dist_max = dist_max

    def _setup_radius(self):
        centroids_child = self._get_child_centroids()
        radii_child = self._get_child_radii()
        self.radius = np.max(np.array(self._get_dists(centroids_child, self.centroid)) + radii_child)
        self._setup_dist_max()

    def setup(self):
        centroids_child = self._get_child_centroids()
        self.centroid = np.mean(centroids_child, axis=0)
        self._setup_radius()
        self.try_to_reparent()

    def search_pn(self, vec: np.ndarray, dist_pn: float, stat: HSTStat = None):
        if stat:
            stat.n_walks += 1
        candidates = self._get_candidate_children(vec, dist_pn)
        for child in candidates:
            if stat:
                stat.add_n_candidates_per_level(self.get_level(), 1)
            if isinstance(child, HS):
                vec_pn = child.search_pn(vec, dist_pn, stat)
                if vec_pn is not None:
                    return vec_pn
                if stat:
                    stat.n_backtracks += 1
            else:
                return child
        return None

    def search_nn(self, vec: np.ndarray):
        vec_min = None
        dist_min = 0
        for child in self.children:
            if isinstance(child, HS):
                vec_min_hs, dist_hs = child.search_nn(vec)
                if vec_min is None or dist_hs < dist_min:
                    dist_min = dist_hs
                    vec_min = vec_min_hs
            else:
                dist = self.hst.dist_func(child.v, vec)
                if dist < dist_min:
                    vec_min = child
                    dist_min = dist
        return vec_min, dist_min

    def _search_vecs_nn(self, v: np.ndarray, dist: float):
        vec_dists_nn = []
        for child in self.children:
            if isinstance(child, HS):
                vec_dists_nn_child = child._search_vecs_nn(v, dist)
                vec_dists_nn += vec_dists_nn_child
            else:
                dist_child = self.hst.dist_func(child.v, v)
                if dist_child < dist:
                    vec_dists_nn.append((child, dist_child))
        return vec_dists_nn

    def search_vecs_nn(self, v: np.ndarray, dist: float):
        vec_dists_nn = self._search_vecs_nn(v, dist)
        return [x[0] for x in sorted(vec_dists_nn, key=lambda x: x[1])]

    def get_n_closer_vecs(self, v: np.ndarray, dist: float):
        return len(self._search_vecs_nn(v, dist))

    def _search_top_k_vecs(self, v: np.ndarray, top_k: int, vec_dists: list):
        for child in self.children:
            if isinstance(child, HS):
                child._search_top_k_vecs(v, top_k, vec_dists)
            else:
                dist_child = self.hst.dist_func(child.v, v)
                rank = 0
                inserted = False
                for vec_dist in vec_dists:
                    if dist_child < vec_dist[1]:
                        vec_dists.insert(rank, (child, dist_child))
                        inserted = True
                        break
                    rank += 1
                if not inserted:
                    vec_dists.append((child, dist_child))
                if len(vec_dists) > top_k:
                    del vec_dists[top_k:]

    def get_rank_vec_dist(self, v: np.ndarray, rank: int):
        vec_dists = []
        self._search_top_k_vecs(v, rank, vec_dists)
        return vec_dists[-1][1]

    def get_summary(self):
        n_hs = 1
        n_vecs = 0
        n_levels_child_max = 1 if self.children else 0
        for child in self.children:
            if isinstance(child, HS):
                n_levels_child, n_hs_child, n_vecs_child = child.get_summary()
                if n_levels_child > n_levels_child_max:
                    n_levels_child_max = n_levels_child
                n_hs += n_hs_child
                n_vecs += n_vecs_child
            else:
                n_vecs += 1

        return n_levels_child_max + 1, n_hs, n_vecs

    def show(self, indent: str):
        print(f"{indent}{self}")
        for child in self.children:
            if isinstance(child, HS):
                child.show(indent + " ")
            else:
                if detailed_str:
                    print(f"{indent} {child}")

    def __repr__(self):
        return f"[{self.hid}]-{self.dist_max:.3f}-r:{self.radius:.3f}#{len(self.children)}"
