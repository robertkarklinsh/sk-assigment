import numpy as np


class ValIter(object):
    """
    Vanilla value iteration algorithm.
    """

    def __init__(self,
                 ob_space_dims=(10, 10),
                 timeout=500,
                 delta=0.1):
        """

        Args:
            ob_space_dims: tuple(height,width)
            timeout: max_num of inner cycles, at each cycle algorithm applies Bellman operator to the whole state space
            delta: sup_norm between value functions of two successive cycles, determines when algorithm exits
        """
        self.ob_space_dims = ob_space_dims
        self.n_vertices = self.ob_space_dims[0] * self.ob_space_dims[1]
        self.cost_g = np.zeros(self.ob_space_dims).flatten()
        # Deterministic policy is represented as boolean matrix: array[cur_state, next_state]
        self.policy = np.zeros((self.n_vertices, self.n_vertices))
        self.env = None
        self.delta = delta
        # Discounting constant
        self.sigma = 0.95
        self.timeout = timeout
        # Used for reporting
        self.n_iterations = 0
        self.name = "Value iteration"

    def find_shortest_path(self, start, goal, env):
        t = 0
        self.env = env
        temp = np.ones(self.ob_space_dims).flatten()
        while np.amax(np.fabs(temp - self.cost_g)) > self.delta:

            t += 1
            if t > self.timeout:
                raise Exception
            temp = self.cost_g.copy()
            for s in self.env.states:
                if s == goal:
                    self.cost_g[s] = 0
                else:
                    next_states = self.env.G[s, :]
                    self.cost_g[s] = np.amin(self.env.W[s, next_states] + self.sigma * self.cost_g[next_states])
            self.n_iterations += 1
        self._init_greedy_policy()
        shortest_path = self._traceforward(start, goal)
        cost = self.cost_g[start]
        n_iterations = self.n_iterations
        self.reset()
        return shortest_path, cost, n_iterations

    def _init_greedy_policy(self):

        """
        Init deterministic greedy policy with respect to current cost function.
        """
        self.policy = self.policy * 0
        for s in self.env.states:
            next_states = self.env.G[s, :]
            costs = self.env.W[s, next_states] + self.cost_g[next_states]
            greedy_actions = (self.env.W[s, :] + self.cost_g == np.amin(costs)) * next_states
            self.policy[s, greedy_actions] = 1 / np.count_nonzero(greedy_actions)

    def _traceforward(self, s, goal):
        path = []
        path.append(s)
        while s != goal:
            s = np.argmax(self.policy[s, :])
            path.append(s)
        return path

    def reset(self):
        self.cost_g = np.zeros(self.ob_space_dims).flatten()
        self.policy = np.zeros((self.n_vertices, self.n_vertices))
        self.n_iterations = 0
