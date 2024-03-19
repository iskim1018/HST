from __future__ import annotations
import numpy as np
from typing import Union, TYPE_CHECKING
from sklearn.cluster import KMeans


if TYPE_CHECKING:
    import HST

terse_mode = False


class HS:
    def __init__(self, hst: HST, hs_parent: Union[HS, None], children: list):
        self.centroid = 0
        self.radius = 0
        self.hst = hst
        self.hs_parent = hs_parent
        self.children = children
        for child in children:
            if isinstance(child, HS):
                child.hs_parent = self
        self.setup()

    def _get_child_centroids(self):
        return list(map(lambda child: child.centroid if isinstance(child, HS) else child,
                        self.children))

    def _get_child_radii(self):
        return list(map(lambda child: child.radius if isinstance(child, HS) else 0,
                        self.children))

    def _get_child_new_centroids(self, vec):
        return list(map(lambda child: child._get_new_centroid(vec) if isinstance(child, HS) else
                        np.mean([child, vec], axis=0), self.children))

    def _get_new_centroid(self, vec):
        vecs = self._get_child_centroids()
        vecs.append(vec)
        return np.mean(vecs, axis=0)

    def _get_nearest_child_idx(self, vec: np.ndarray):
        vecs = self._get_child_new_centroids(vec)
        return np.argmin(np.linalg.norm(vecs - vec, axis=1))

    def _get_candidate_children(self, vec: np.ndarray, dist_pn: float):
        candidates = []
        vecs = self._get_child_centroids()
        dists = np.linalg.norm(vecs - vec, axis=1)
        for child, dist_centroid in zip(self.children, dists):
            if isinstance(child, HS):
                if child.radius >= dist_centroid or dist_centroid - child.radius < dist_pn:
                    candidates.append((child, dist_centroid))
            else:
                if dist_centroid < dist_pn:
                    candidates.append((child, dist_centroid / 2))
        candidates = sorted(candidates, key=lambda x: x[1])
        return [x[0] for x in candidates]

    def add(self, vec: np.ndarray) -> Union[HS, None]:
        if len(self.children) >= self.hst.n_max_child:
            idx = self._get_nearest_child_idx(vec)
            child = self.children[idx]
            if isinstance(child, HS):
                child.insert(vec)
            else:
                hs_new = HS(self.hst, self, [child, vec])
                self.children[idx] = hs_new
        else:
            self.children.append(vec)
        self._setup_radius()

    def insert(self, vec: np.ndarray):
        dist = np.linalg.norm(self.centroid - vec)
        if dist < self.radius:
            self.add(vec)
            return

        if len(self.children) >= self.hst.n_max_child:
            idx = self._get_nearest_child_idx(vec)
            child = self.children[idx]
            hs_new = HS(self.hst, self, [child, vec])
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

    def _get_nearest_candidate_hs(self, vec: np.ndarray):
        vecs = self._get_child_new_centroids(vec)
        dists = np.linalg.norm(vecs - vec, axis=1)
        for idx, dist in sorted(enumerate(dists), key=lambda x: x[1]):
            child = self.children[idx]
            if isinstance(child, HS) and child.radius <= dist:
                return child
        return None

    def reparent(self, child_from, child_grand):
        vec = child_grand.centroid if isinstance(child_grand, HS) else child_grand
        child_nn = self._get_nearest_candidate_hs(vec)
        if child_nn is None or child_from == child_nn:
            return
        child_from.remove(child_grand)
        child_nn.add(child_grand)

    def try_to_reparent(self):
        if self.hs_parent is None:
            return
        for child in self.children:
            self.hs_parent.reparent(self, child)

    def _setup_radius(self):
        centroids_child = self._get_child_centroids()
        radii_child = self._get_child_radii()
        self.radius = np.max(np.linalg.norm(np.array(centroids_child) - self.centroid) + radii_child)

    def setup(self):
        centroids_child = self._get_child_centroids()
        radii_child = self._get_child_radii()
        self.centroid = np.mean(centroids_child, axis=0)
        self.radius = np.max(np.linalg.norm(np.array(centroids_child) - self.centroid) + radii_child)
        self.try_to_reparent()

    def search_pn(self, vec: np.ndarray, dist_pn: float):
        candidates = self._get_candidate_children(vec, dist_pn)
        for child in candidates:
            if isinstance(child, HS):
                vec_pn = child.search_pn(vec, dist_pn)
                if vec_pn is not None:
                    return vec_pn
            else:
                return child
        return None

    def search_dfs(self, vec: np.ndarray):
        vec_min = None
        dist_min = 0
        for child in self.children:
            if isinstance(child, HS):
                vec_min_hs, dist_hs = child.search_dfs(vec)
                if vec_min is None or dist_hs < dist_min:
                    dist_min = dist_hs
                    vec_min = vec_min_hs
            else:
                dist = np.linalg.norm(child - vec)
                if dist < dist_min:
                    vec_min = child
                    dist_min = dist
        return vec_min, dist_min

    def search_n_closer_vecs(self, vec: np.ndarray, dist: int):
        n_closer_vecs = 0
        for child in self.children:
            if isinstance(child, HS):
                n_closer_vecs_hs = child.search_n_closer_vecs(vec, dist)
                n_closer_vecs += n_closer_vecs_hs
            else:
                dist_child = np.linalg.norm(child - vec)
                if dist_child < dist:
                    n_closer_vecs += 1
        return n_closer_vecs

    def _search_top_k_vecs(self, vec: np.ndarray, top_k: int, vec_dists: list):
        for child in self.children:
            if isinstance(child, HS):
                child._search_top_k_vecs(vec, top_k, vec_dists)
            else:
                dist_child = np.linalg.norm(child - vec)
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

    def get_rank_vec_dist(self, vec: np.ndarray, rank: int):
        vec_dists = []
        self._search_top_k_vecs(vec, rank, vec_dists)
        return vec_dists[-1][1]

    def show(self, indent: str):
        print(f"{indent}{self.__repr__()}")
        for child in self.children:
            if isinstance(child, HS):
                child.show(indent + " ")
            else:
                if not terse_mode:
                    print(indent, child)

    def __repr__(self):
        return f"r:{self.radius:.3f}#{len(self.children)}"
