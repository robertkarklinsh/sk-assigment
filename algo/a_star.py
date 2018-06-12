import numpy as np


class AStar(object):
    """

    A* algorithm for finding shortest path between two endpoints in gridworld.
    """

    def __init__(self,
                 ob_space_dims=(10, 10),
                 timeout=500):
        """

        Args:
            ob_space_dims: tuple(height,width)
            timeout: max_num of inner cycles, at each cycle algorithm expands to next states
        """
        self.ob_space_dims = ob_space_dims
        self.n_vertices = self.ob_space_dims[0] * self.ob_space_dims[1]
        self.cost_g = np.zeros(self.ob_space_dims).flatten() + np.infty
        # Heuristic cost function, implemented as euclidean distance to goal:
        self.cost_h = None
        # Keeps track of current shortest paths to opened elements:
        self.traceback_table = np.zeros((self.n_vertices, self.n_vertices), dtype=bool)
        # Open list from original paper paper (Hart et al., 1968)
        self.opened = []
        self.timeout = timeout
        # Used for reporting
        self.n_iterations = 0
        self.name = "A*"

    def find_shortest_path(self, start, goal, env):
        """

        Args:
            env: instance of GridWorld, iterates underlying grids by calling reset() method
            start: int, C style flat index of starting point
            goal: int, C style flat index of ending point

        Returns:
            tuple:
                shortest_path: list of flat indices
                cost: int, distance along shortest path

        """
        goal_i, goal_j = np.unravel_index(goal, self.ob_space_dims)
        self.cost_h = np.fromfunction(lambda i, j: np.sqrt((goal_i - i) ** 2 + (goal_j - j) ** 2),
                                      self.ob_space_dims).flatten()
        s = start
        self.cost_g[s] = 0
        self.opened.append(s)
        t = 0
        while t < self.timeout:
            t = t + 1
            s = self.opened[(self.cost_g + self.cost_h)[self.opened].argmin()]
            self.opened.remove(s)
            if s == goal:
                shortest_path = list(reversed(self._traceback(s, start)))
                cost = self.cost_g[s]
                n_rollouts = self.n_iterations
                self.reset()

                return shortest_path, cost, n_rollouts
            else:
                edges = env.step(s)
                self.n_iterations += 1
                for e in edges:
                    c = self.cost_g[s] + e[1]
                    if self.cost_g[e[0]] > c:
                        self.cost_g[e[0]] = c
                        self.opened.append(e[0])
                        self.traceback_table[e[0], self.traceback_table[e[0], :]] = 0
                        self.traceback_table[e[0], s] = 1
        self.reset()
        raise Exception

    def _traceback(self, s, start):
        path = []
        path.append(s)
        while s != start:
            s = np.where(self.traceback_table[s, :])[0][0]
            path.append(s)
        return path

    def reset(self):
        self.opened = []
        self.cost_g = np.zeros(self.ob_space_dims).flatten() + np.infty
        self.cost_h = None
        self.traceback_table = np.zeros((self.n_vertices, self.n_vertices), dtype=bool)
        self.n_iterations = 0
