class HSTStat:
    n_walks: int = 0
    n_backtracks: int = 0
    n_candidates_per_level: list = []

    def add_n_candidates_per_level(self, level: int, n_candidates: int):
        while len(self.n_candidates_per_level) < level:
            self.n_candidates_per_level.append(0)
        self.n_candidates_per_level[level - 1] += n_candidates

    def clear(self):
        self.n_walks = 0
        self.n_backtracks = 0
        self.n_candidates_per_level = []
