from __future__ import annotations
import numpy as np
from typing import Union, TYPE_CHECKING
from sklearn.cluster import KMeans


if TYPE_CHECKING:
    import HST


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

    def add(self, vec: np.ndarray) -> Union[HS, None]:
        if len(self.children) >= self.hst.n_max_child:
            self.children.append(vec)
            vecs = self._get_child_centroids()
            kmeans = KMeans(n_clusters=2)
            kmeans.fit(vecs)

            children1 = [child for child, label in zip(self.children, kmeans.labels_) if label == 0]
            children2 = [child for child, label in zip(self.children, kmeans.labels_) if label == 1]
            hs_new = HS(self.hst, None, children1)
            self.children = children2
            self.setup()

            return hs_new
        else:
            dist = np.linalg.norm(self.centroid - vec)
            if dist < self.radius:
                vecs = self._get_child_centroids()
                idx = np.argmin(np.linalg.norm(vecs - vec))
                child = self.children[idx]
                if isinstance(child, HS):
                    hs_new = child.add(vec)
                    if hs_new is not None:
                        self.children.append(hs_new)
                        self.setup()
                    return None
            self.children.append(vec)
            self.setup()
            return None

    def setup(self):
        centroids_child = self._get_child_centroids()
        radii_child = self._get_child_radii()
        self.centroid = np.mean(centroids_child, axis=0)
        self.radius = np.max(np.linalg.norm(np.array(centroids_child) - self.centroid) + radii_child)

    def show(self, indent: str):
        print(f"{indent}r:{self.radius:.3f}")
        for child in self.children:
            if isinstance(child, HS):
                child.show(indent + " ")
            else:
                print(indent, child)

    def __repr__(self):
        return f"r:{self.radius:.3f}#{len(self.children)}"
