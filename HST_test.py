#!/usr/bin/python3

import sys
import getopt
import logging
import numpy as np

import HS
from HST import HST


n_max_child: int = 5
vec_dim: int = 8
vec_range = (-1, 1)
n_vectors = 100
query_mode = False
query_dist = None
query_rank = None
query_pn_dist = None
path_save = None
path_load = None
seed = None


def _usage_hst_test():
    print("""\
Usage: HST_test.py [<options>]
   <options>
   -h: help(this message)
   -q <query type>: rank:<rank>, dist:<dist>, pn:<dist_threshold>
   -m <max child in HS>
   -d <vector dimension>: vector dimension
   -n <# of vectors>: default 100
   -r <vector range format>: eg: -1,1 default (-1, 1)
   -s <path>: save HST
   -l <path>: load HST
   -S: setting seed for numpy random
   -t: terse output 
""")


logging.basicConfig(level=logging.DEBUG,  # 로그 레벨 설정
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 로그 포맷 설정
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


def _parse_query_str(a):
    global query_dist, query_rank, query_pn_dist

    if a[0:5] == "rank:":
        query_rank = int(a[5:])
    elif a[0:5] == "dist:":
        query_dist = float(a[5:])
    elif a[0:3] == "pn:":
        query_pn_dist = float(a[3:])


def _parse_args():
    global n_max_child, vec_dim, n_vectors, vec_range, query_mode, path_load, path_save, seed

    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:d:n:q:r:s:l:S:th")
    except getopt.GetoptError:
        logger.error("invalid option")
        _usage_hst_test()
        exit(1)
    for o, a in opts:
        if o == '-h':
            _usage_hst_test()
            exit(0)
        elif o == '-m':
            n_max_child = int(a)
        elif o == '-d':
            vec_dim = int(a)
        elif o == '-n':
            n_vectors = int(a)
        elif o == '-q':
            query_mode = True
            _parse_query_str(a)
        elif o == '-s':
            path_save = a
        elif o == '-l':
            path_load = a
        elif o == '-S':
            seed = int(a)
        elif o == '-t':
            HS.terse_mode = True
        elif o == '-r':
            try:
                vec_range = tuple(map(int, a.split(',')))
            except ValueError:
                print("Invalid format for range. Expected format: -r -1,1")
                exit(2)


def run_query(hst, vec):
    if query_rank:
        dist = hst.get_rank_vec_dist(vec, query_rank)
        print(f"rank {query_rank}: {dist:.4f}")
    elif query_dist:
        dist = hst.get_nn_vec_dist(vec)
        print(f"nn vec: {dist:.4f}")
    elif query_pn_dist:
        vec_pn = hst.get_pn_vec(vec, query_pn_dist)
        if vec_pn is None:
            print("PN vector not found")
        else:
            dist_pn = hst.get_dist(vec, vec_pn)
            rank = hst.get_pn_dist_rank(vec, dist_pn)
            print(f"PN vec: rank: {rank}, dist:{dist_pn:.4f}")


def run_hst_test():
    if seed:
        np.random.seed(seed)
    if path_load:
        hst = HST.load(path_load)
    else:
        hst = HST(vec_dim, n_max_child)
    for i in range(n_vectors):
        vec = np.random.uniform(vec_range[0], vec_range[1], vec_dim)
        if query_mode:
            run_query(hst, vec)
        else:
            hst.insert(vec)
    if not query_mode:
        hst.show()
    if path_save:
        hst.save(path_save)


if __name__ == "__main__":
    _parse_args()
    run_hst_test()
