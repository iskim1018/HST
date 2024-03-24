import numpy as np
from scipy import spatial


def get_euclidean_dist(vec1: np.ndarray, vec2: np.ndarray):
    return np.linalg.norm(vec1 - vec2)


def get_cosine_dist(vec1: np.ndarray, vec2: np.ndarray):
    return spatial.distance.cosine(vec1, vec2)
