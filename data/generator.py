import numpy as np


class ObstacleMapGenerator(object):
    """
    Returns grids filled with random obstacles.
    """

    def __init__(self,
                 grid_dims=(10, 10),
                 n_grids=100,
                 n_obs=10,
                 max_size=None):
        """

        Args:
            grid_dims: tuple(grid_height, grid_width)
            n_grids: int, number of grids to create
            n_obs: int, how many random obstacles to fill in each grid, currently all obstacles are rectangles
            max_size: int, maximum length of single rectangle along any dimension
        """
        self.data = None
        self.grid_dims = grid_dims
        self.n_grids = n_grids
        self.n_obs = n_obs
        self.max_size = max_size or np.ceil(np.min(self.grid_dims) / 5)

    def generate_data(self):
        self.data = []
        for i in range(self.n_grids):
            grid = self._insert_rnd_obstacles(np.zeros(self.grid_dims))
            self.data.append(grid)
        return self.data

    def _insert_rnd_obstacles(self, grid):
        for _ in range(self.n_obs):
            rand_y = int(np.ceil(np.random.rand() * grid.shape[0] - 1))
            rand_x = int(np.ceil(np.random.rand() * grid.shape[1] - 1))
            rand_width = int(np.ceil(np.random.rand() * self.max_size))
            rand_height = int(np.ceil(np.random.rand() * self.max_size))
            grid = self._insert_rectangle(grid, rand_x, rand_y, rand_width, rand_height)
        return grid

    def _insert_rectangle(self, grid, x, y, width, height):
        grid[y:y + height, x:x + width] = 1
        return grid
