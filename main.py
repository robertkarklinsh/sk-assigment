from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import argparse
import logging
import sys
import numpy as np

from data.generator import ObstacleMapGenerator
from env.gridworld import GridWorld
from runner import Runner
from algo.a_star import AStar
from algo.val_iter import ValIter

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

if __name__ == '__main__':

    modpath = os.path.dirname(os.path.abspath(sys.argv[0]))
    datapath = os.path.join(modpath, 'data/grid_%sx%s.npz')

    parser = argparse.ArgumentParser()

    parser.add_argument('-g', '--gridsize',
                        nargs='+',
                        default=(10, 10),
                        type=int)
    parser.add_argument('-d', '--datafile',
                        type=str,
                        default=datapath,
                        help='Path to data file. '
                             'By default tries to find existing file, otherwise generates and saves new batch of grids.')
    parser.add_argument('-a', '--alg',
                        type=int,
                        default=2,
                        choices=[0, 1, 2],
                        help='0 for VI, 1 for A*, 2 for both (default)')
    args = parser.parse_args()
    filename = args.datafile % (args.gridsize[0], args.gridsize[1])
    try:
        with np.load(filename) as f:
            grids = f['arr_0']
    except Exception as e:
        logger.info('Generating grids with random obstacles...')
        generator = ObstacleMapGenerator(args.gridsize)
        grids = generator.generate_data()
        np.savez_compressed(filename, grids)

    env = GridWorld(grids)
    if args.alg == 0:
        algs = [ValIter(args.gridsize)]
    elif args.alg == 1:
        algs = [AStar(args.gridsize)]
    else:
        algs = [ValIter(args.gridsize), AStar(args.gridsize)]
    try:
        Runner(algs, env).run()
    except KeyboardInterrupt:
        sys.exit(0)
