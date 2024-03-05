#!/usr/bin/python3

import sys
import getopt
import random
import logging
import math
import numpy as np
from scipy import spatial

vec_dim: int = 8
vec_range = (-1, 1)
n_vectors = 100
vectors = []
vec_centroid = []
similarity_func = None

def _usage_hs_test():
    print("""\
Usage: hs_test.py [<options>]
   <options>
   -h: help(this message)
   -d <vector dimension>: vector dimension
   -n <# of vectors>: default 100
   -r <vector range format>: eg: -1,1 default (-1, 1)
   -s <similarity>: euclidean(default), cosine
""")


logging.basicConfig(level=logging.DEBUG,  # 로그 레벨 설정
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 로그 포맷 설정
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


def gen_vectors():
    global vectors

    for i in range(n_vectors):
        vector = np.random.uniform(vec_range[0], vec_range[1], vec_dim)
        vectors.append(vector)


def get_centroids():
    global vec_centroid

    vec_centroid = np.mean(vectors, axis=0)


def _get_cosine(vec1, vec2, weights=None):
    return spatial.distance.cosine(vec1, vec2, weights)


def _get_euclidean(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)


def get_dist_from_centroid(vec):
    return similarity_func(vec_centroid, vec)


def show_euclidean_dist():
    dist_min = -1
    dist_max = 0
    for i in range(n_vectors):
        dist = get_dist_from_centroid(vectors[i])
        if dist_min < 0 or dist < dist_min:
            dist_min = dist
        if dist > dist_max:
            dist_max = dist
        #print(f"{dist:.3f}", end='')
        #print()
    print(f"Min: {dist_min}, Max: {dist_max}")


def _parse_args():
    global vec_dim, n_vectors, vec_range, similarity_func

    try:
        opts, args = getopt.getopt(sys.argv[1:], "d:n:r:s:h")
    except getopt.GetoptError:
        logger.error("invalid option")
        _usage_hs_test()
        exit(1)
    for o, a in opts:
        if o == '-h':
            _usage_hs_test()
            exit(0)
        elif o == '-d':
            vec_dim = int(a)
        elif o == '-n':
            n_vectors = int(a)
        elif o == '-r':
            try:
                vec_range = tuple(map(int, a.split(',')))
            except ValueError:
                print("Invalid format for range. Expected format: -r -1,1")
                exit(2)
        elif o == '-s':
            if a == "cosine":
                similarity_func = _get_cosine

    if similarity_func is None:
        similarity_func = _get_euclidean


if __name__ == "__main__":
    _parse_args()
    gen_vectors()
    get_centroids()
    show_euclidean_dist()
