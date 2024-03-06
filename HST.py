import numpy as np

from HS import HS


class HST:
    def __init__(self, n_dim: int, n_max_child: int):
        self.n_dim = n_dim
        self.n_max_child = n_max_child
        self.RHS = None
        pass

    def add(self, vec: np.ndarray):
        if self.RHS is None:
            self.RHS = HS(self, None, [vec])
        else:
            hs_new = self.RHS.add(vec)

            if hs_new is not None:
                hs_parent = HS(self, None, [self.RHS, hs_new])
                self.RHS = hs_parent

    def show(self):
        self.RHS.show("")

    def __repr__(self):
        return "HST"
