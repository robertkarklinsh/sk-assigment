import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


class GridWorld(object):
    """

    Wraps raw grid data with RL-style methods reset() and step() as well as state transition graph.
    """

    def __init__(self, data):
        """

        Args:
            data: iterable(np.array((height, width))): zeros are free space and ones are obstacles
        """
        self.data = data
        self.dims = data[0].shape[0], data[0].shape[1]
        # Iterates underlying grids
        self.data = cycle(self.data)
        self.n_vertices = self.dims[0] * self.dims[1]
        self.states = None
        self.cur_grid = None
        # Transition graph
        self.G = None
        self.W = None
        self.start = None
        self.goal = None

    def reset(self, change_obstacles=True):
        """

        Move to next grid in data, sample random endpoints
        Args:
            change_obstacles: if False, preserve underlying grid and change only endpoints

        Returns:
            tuple(start_state, goal_state)
        """
        if change_obstacles or self.cur_grid is None: self.cur_grid = next(self.data).flatten().astype(bool)
        self._sample_endpoints()
        self.G = np.fromfunction(self._incident, (self.n_vertices, self.n_vertices), dtype=np.int16)
        self.W = np.fromfunction(self._distance, (self.n_vertices, self.n_vertices), dtype=np.int16)
        self.states = np.arange(self.n_vertices)[np.invert(self.cur_grid)]
        return self.start, self.goal

    def show(self):
        bitmap = self.cur_grid.astype(np.float32)
        bitmap[self.start] = 0.2
        bitmap[self.goal] = 0.6
        plt.imshow(np.reshape(bitmap, self.dims), cmap='Greys')
        plt.show()

    def step(self, cur_state):
        """

        Rollout next states with respect to current state.
        Takes current state as argument so the environment is stateless (model-based scenario).

        Args:
            cur_state: flat index of current state

        Returns:
            list of tuple(next_state_flat_index, transition_cost)
        """
        if cur_state == self.goal:
            return cur_state, 0
        if self.cur_grid[cur_state]:
            return None, 0
        else:
            next_states = np.where(self.G[cur_state, :])[0]
            costs = self.W[cur_state, next_states]
            return list(zip(next_states, costs))

    def _sample_endpoints(self):
        while True:
            self.start = np.random.randint(0, self.n_vertices)
            self.goal = np.random.randint(0, self.n_vertices)
            is_obstacle = self.cur_grid[self.start]
            if not is_obstacle and not self.cur_grid[self.goal] and self.start != self.goal: break

    def _incident(self, v1, v2):
        dist = self._distance(v1, v2)
        is_incident = (dist < 1.5) * (dist > 0)
        v1_is_free = np.invert(self.cur_grid[v1])
        v2_is_free = np.invert(self.cur_grid[v2])
        return is_incident * v1_is_free * v2_is_free

    def _distance(self, v1, v2):
        v1_i = v1 // self.dims[1]
        v1_j = v1 % self.dims[1]
        v2_i = v2 // self.dims[1]
        v2_j = v2 % self.dims[1]
        return np.sqrt((v1_i - v2_i) ** 2 + (v1_j - v2_j) ** 2)
