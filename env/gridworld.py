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
        self.n_cells = self.dims[0] * self.dims[1]
        # Free cells
        self.states = None
        self.cur_grid = None
        # Transition graph
        self.G = None
        self.W = None
        self._start = None
        self._goal = None

    def reset(self, change_obstacles=True, sample_goal=True):
        """

        Move to next grid in data, sample random endpoints
        Args:
            change_obstacles: if False, preserve underlying grid and change only endpoints

        Returns:
            tuple(start_state, goal_state)
        """
        if change_obstacles or self.cur_grid is None: self.cur_grid = next(self.data).flatten().astype(bool)
        self._sample_endpoints()
        self.G = np.fromfunction(self._incident, (self.n_cells, self.n_cells), dtype=np.int16)
        self.W = np.fromfunction(self._distance, (self.n_cells, self.n_cells), dtype=np.int16)
        self.states = np.arange(self.n_cells)[np.invert(self.cur_grid)]
        return self._start, self._goal

    def show(self, block=True):
        bitmap = self.cur_grid.astype(np.float32)
        if self._start is not None:
            bitmap[self._start] = 0.2
        if self._goal is not None:
            bitmap[self._goal] = 0.6
        plt.imshow(np.reshape(bitmap, self.dims), cmap='Greys')
        if block:
            plt.show()
        else:
            plt.show(block=False)

    def step(self, cur_state):
        """

        Expand next states with respect to current state.
        Takes current state as argument so the environment is stateless (model-based scenario).

        Args:
            cur_state: int, index of current state

        Returns:
            list(tuple(next_state_index, transition_cost))
        """
        if self.cur_grid[cur_state]:
            return list()
        else:
            next_states = np.where(self.G[cur_state, :])[0]
            if next_states.size == 0:
                return list()
            costs = self.W[cur_state, next_states]
            return list(zip(next_states, costs))

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, val):
        if val is None:
            self._goal = None
            return
        if not 0 <= val <= self.n_cells:
            raise Exception
        if self._start is None or val == self._start or self.cur_grid[val]:
            raise Exception
        self._goal = val

    def _sample_endpoints(self):
        while True:
            self._start = np.random.randint(0, self.n_cells)
            self._goal = np.random.randint(0, self.n_cells)
            if not self.cur_grid[self._start] and not self.cur_grid[self._goal] and self._start != self._goal: break

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
