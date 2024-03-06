#!/usr/bin/python3

import sys
import getopt
import logging
import numpy as np

from HST import HST


n_max_child: int = 5
vec_dim: int = 8
vec_range = (-1, 1)
n_vectors = 100


def _usage_hst_test():
    print("""\
Usage: HST_test.py [<options>]
   <options>
   -h: help(this message)
   -m <max child in HS>
   -d <vector dimension>: vector dimension
   -n <# of vectors>: default 100
   -r <vector range format>: eg: -1,1 default (-1, 1)
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


def _parse_args():
    global n_max_child, vec_dim, n_vectors, vec_range

    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:d:n:r:h")
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
        elif o == '-r':
            try:
                vec_range = tuple(map(int, a.split(',')))
            except ValueError:
                print("Invalid format for range. Expected format: -r -1,1")
                exit(2)


def run_hst_test():
    hst = HST(vec_dim, n_max_child)
    for i in range(n_vectors):
        vec = np.random.uniform(vec_range[0], vec_range[1], vec_dim)
        hst.add(vec)
    hst.show()

if __name__ == "__main__":
    _parse_args()
    run_hst_test()
