#!/usr/bin/python3

import sys
import getopt
import logging
import numpy as np

import HS
import vector
from HST import HST
from hst_stat import HSTStat

n_max_child: int = 5
vec_dim: int = 8
vec_range = (-1, 1)
n_vectors = 100
query_mode = False
query_rank = None
query_nn_dist = None
query_pn_dist = None
path_save = None
path_load = None
seed = None
verbose = ""

def _usage_hst_test():
    print("""\
Usage: HST_test.py [<options>]
   <options>
   -h: help(this message)
   -q <query type>: rank:<rank>, nn:<dist>, pn:<dist_threshold>
   -c <max child in HS>
   -d <vector dimension>: vector dimension
   -n <# of vectors>: default 100
   -r <vector range format>: eg: -1,1 default (-1, 1)
   -s <path>: save HST
   -l <path>: load HST
   -S: setting seed for numpy random
   -v <option>: verbose output, options: stvV
        s: summary, t: tree, v: vector, V: vector data 
""")


logging.basicConfig(level=logging.DEBUG,  # 로그 레벨 설정
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 로그 포맷 설정
                    handlers=[
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)


def _parse_query_str(a):
    global query_rank, query_nn_dist, query_pn_dist

    if a[0:5] == "rank:":
        query_rank = int(a[5:])
    elif a[0:3] == "nn:":
        query_nn_dist = float(a[3:])
    elif a[0:3] == "pn:":
        query_pn_dist = float(a[3:])


def _parse_args():
    global n_max_child, vec_dim, n_vectors, vec_range, query_mode, path_load, path_save, seed, verbose

    try:
        opts, args = getopt.getopt(sys.argv[1:], "c:d:n:q:r:s:l:S:v:h")
    except getopt.GetoptError:
        logger.error("invalid option")
        _usage_hst_test()
        exit(1)
    for o, a in opts:
        if o == '-h':
            _usage_hst_test()
            exit(0)
        elif o == '-c':
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
        elif o == '-v':
            verbose = a
            if ('v' or 'V') in a:
                HS.detailed_str = True
            if 'V' in a:
                vector.detailed_str = True
        elif o == '-r':
            try:
                vec_range = tuple(map(int, a.split(',')))
            except ValueError:
                print("Invalid format for range. Expected format: -r -1,1")
                exit(2)


def run_query(hst, v: np.ndarray):
    if query_rank:
        dist = hst.get_rank_vec_dist(v, query_rank)
        print(f"rank {query_rank}: {dist:.4f}")
    elif query_nn_dist:
        vecs_nn = hst.get_nn_vecs(v, query_nn_dist)
        for vec_nn in vecs_nn:
            dist_nn = hst.get_dist(v, vec_nn.v)
            print(f"NN vec: vid: {vec_nn.vid}, dist: {dist_nn:.4f}")
    elif query_pn_dist:
        stat = HSTStat()
        vec_pn = hst.get_pn_vec(v, query_pn_dist, stat)
        if vec_pn is None:
            print("PN vector not found")
        else:
            dist_pn = hst.get_dist(v, vec_pn.v)
            rank = hst.get_dist_rank(v, dist_pn)
            print(f"PN vec: vid: {vec_pn.vid}, rank: {rank}, dist: {dist_pn:.4f}")
        print(f"# of walks: {stat.n_walks}, # of backtracks: {stat.n_backtracks}")
        print(f"# of candidates per level: ", end='')
        for n in stat.n_candidates_per_level:
            print(f"{n} ", end='')
        print()


def show_summary(hst: HST):
    n_levels, n_hs, n_vecs = hst.get_summary()
    print(f"HST level: {n_levels}, # of vectors: {n_vecs}, # of HS: {n_hs}")


def run_hst_test():
    if seed:
        np.random.seed(seed)
    if path_load:
        global vec_dim
        hst = HST.load(path_load)
        vec_dim = hst.n_dim
    else:
        hst = HST(vec_dim, n_max_child)
    for i in range(n_vectors):
        v = np.random.uniform(vec_range[0], vec_range[1], vec_dim)
        if query_mode:
            run_query(hst, v)
        else:
            hst.insert(v)
    if 't' in verbose:
        hst.show()
    if 's' in verbose:
        show_summary(hst)
    if path_save:
        hst.save(path_save)


if __name__ == "__main__":
    _parse_args()
    run_hst_test()
