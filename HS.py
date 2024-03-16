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

    def insert(self, vec: np.ndarray) -> Union[HS, None]:
        if len(self.children) >= self.hst.n_max_child:
            idx = self._get_nearest_child_idx(vec)
            child = self.children[idx]
            if isinstance(child, HS):
                hs_new = child.add(vec)
                if hs_new is not None:
                    self.children[idx] = hs_new
                    self.setup()
                    return self
                return None
            else:
                hs_new = HS(self.hst, self, [child, vec])
                self.children[idx] = hs_new
                self.setup()
                return self
        else:
            self.children.append(vec)
            self.setup()
            return self

    def add(self, vec: np.ndarray) -> Union[HS, None]:
        dist = np.linalg.norm(self.centroid - vec)
        if dist < self.radius:
            return self.insert(vec)

        if len(self.children) >= self.hst.n_max_child:
            idx = self._get_nearest_child_idx(vec)
            child = self.children[idx]
            hs_new = HS(self.hst, self, [vec, child])
            if isinstance(child, HS):
                child.hs_parent = hs_new
            self.children[idx] = hs_new
            self.setup()
        else:
            self.children.append(vec)
            self.setup()
        return self

    def search(self, vec: np.ndarray):
        idx = self._get_nearest_child_idx(vec)
        child = self.children[idx]
        if isinstance(child, HS):
            return child.search(vec)
        return child

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

    def setup(self):
        centroids_child = self._get_child_centroids()
        radii_child = self._get_child_radii()
        self.centroid = np.mean(centroids_child, axis=0)
        self.radius = np.max(np.linalg.norm(np.array(centroids_child) - self.centroid) + radii_child)

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
