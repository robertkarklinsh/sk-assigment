import numpy as np


class ValIter(object):
    """
    Vanilla value iteration algorithm.
    """

    def __init__(self,
                 grid_dims=(10, 10),
                 timeout=1000,
                 delta=0.01,
                 gamma=0.99):
        """

        Args:
            grid_dims: tuple(grid_height,grid_width)
            timeout: int, max_num of inner cycles, at each cycle algorithm applies Bellman operator to the whole state space
            delta: float, sup_norm between value functions of two successive cycles, determines when algorithm exits
            gamma: int, discount parameter
        """
        self.grid_dims = grid_dims
        self.n_cells = self.grid_dims[0] * self.grid_dims[1]
        self.cost_g = np.zeros(self.grid_dims).flatten()
        # Deterministic policy is represented as boolean matrix: array[cur_state, next_state]
        self.policy = np.zeros((self.n_cells, self.n_cells))
        self.env = None
        self.delta = delta
        # Discounting constant
        self.gamma = gamma
        self.timeout = timeout
        # Used for reporting
        self.n_iterations = 0
        self.name = "Value iteration"

    def find_shortest_path(self, start, goal, env):
        t = 0
        self.env = env
        temp = np.ones(self.grid_dims).flatten()
        while np.amax(np.fabs(temp - self.cost_g)) > self.delta:

            t += 1
            if t > self.timeout:
                raise Exception(self.name + " failed to find shortest path: planning took too long")
            temp = self.cost_g.copy()
            for s in self.env.states:
                if s == goal:
                    self.cost_g[s] = 0
                else:
                    next_states = self.env.G[s, :]
                    if np.count_nonzero(next_states) == 0:
                        continue
                    self.cost_g[s] = np.amin(self.env.W[s, next_states] + self.gamma * self.cost_g[next_states])
            self.n_iterations += 1
        self._init_greedy_policy()
        shortest_path = self._traceforward(start, goal)
        cost = self.cost_g[start]
        n_iterations = self.n_iterations
        self.reset()
        return shortest_path, cost, n_iterations

    def _init_greedy_policy(self):

        """
        Init greedy policy with respect to current cost function.
        """
        self.policy = self.policy * 0
        for s in self.env.states:
            next_states = self.env.G[s, :]
            if np.count_nonzero(next_states) == 0:
                continue
            costs = self.env.W[s, next_states] + self.cost_g[next_states]
            greedy_actions = np.logical_and(self.env.W[s, :] + self.cost_g == np.amin(costs), next_states)
            self.policy[s, greedy_actions] = 1. / np.count_nonzero(greedy_actions)

    def _traceforward(self, s, goal):
        path = []
        path.append(s)
        t = 0
        while s != goal:
            t += 1
            if t > self.timeout:
                raise Exception(
                    self.name + " failed to find shortest path: maybe something is blocking the goal?")
            s = np.argmax(self.policy[s, :])
            path.append(s)
        return path

    def reset(self):
        self.cost_g = np.zeros(self.grid_dims).flatten()
        self.policy = np.zeros((self.n_cells, self.n_cells))
        self.n_iterations = 0
