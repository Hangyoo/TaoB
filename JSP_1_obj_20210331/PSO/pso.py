from tqdm import tqdm
import random
import pickle
from copy import deepcopy
import numpy as np
import pprint
from JSP_1_obj_20210331.PSO.utils import *
from math import *
from JSP_1_obj_20210331 import config

class PSO:

    def __init__(self, n, m, times, schedule):
        self.args = {
            'epoch': config.PSO_epoch,
            'particle_num': config.particle_num,
            'n': n,
            'm': m,
            'start_pairs': 5,
            'alpha': config.alpha,
            'max_length_pairs': 10,
            'seed': 100,
        }
        self.reset(times, schedule)

        # 保存每代最优目标函数
        self.bestList = []

    def show_parameters(self):
        pprint.pprint(self.args)

    def reset(self, times, schedule):
        # pbest and gbest
        self.pbest = [inf] * self.args['particle_num']
        self.gbest = inf
        # init the particle
        self.particles = init(
                self.args['n'], self.args['m'], self.args['particle_num'])
        # init the vec of each particle (pairs)
        self.vecs = [init_pair(self.args['n'] * self.args['m'], self.args['start_pairs']) \
                for _ in range(self.args['particle_num'])]
        # pbest and gbest solution
        self.pbest_solution = deepcopy(self.particles)
        self.gbest_solution = None
        # times and schedule, which are red from the `test.txt`
        self.times = times
        self.schedule = schedule
        print(f'[!] reset and init the PSO solver agent')

    def train(self):
        # set the seed
        random.seed(self.args['seed'])
        np.random.seed(self.args['seed'])
        # training
        pbar = tqdm(range(self.args['epoch']))
        for e in pbar:
            for idx, p in enumerate(self.particles):
                f = calculate_time(
                        self.args['n'], self.args['m'], p,
                        self.times, self.schedule)
                if f < self.pbest[idx]:
                    self.pbest[idx] = f
                    self.pbest_solution[idx] = deepcopy(p)
            # find gbest and corresponding solution
            gbest_idx, gbest_rest = 0, inf
            for idx in range(len(self.particles)):
                if self.pbest[idx] < gbest_rest:
                    gbest_idx = idx
                    gbest_rest = self.pbest[idx]
            self.gbest = gbest_rest
            # 每代的最优目标函数保存到列表中
            self.bestList.append(gbest_rest)
            self.gbest_solution = deepcopy(self.pbest_solution[gbest_idx])
            # move
            for idx in range(len(self.particles)):
                p = self.particles[idx]
                gbest_delta = find_switch_pairs(self.gbest_solution, p, self.args['n'])
                pbest_delta = find_switch_pairs(self.pbest_solution[idx], p, self.args['n'])
                pairs = []
                if random.random() < self.args['alpha']:
                    pairs = pbest_delta
                else:
                    pairs = gbest_delta
                vec = self.vecs[idx] + pairs
                if len(vec) > self.args['max_length_pairs']:
                    vec = random.sample(vec, self.args['max_length_pairs'])
                    self.vecs[idx] = vec
                apply_pairs(self.particles[idx], vec)
            pbar.set_description(f'[!] best result {round(self.gbest, 4)} at epoch {e}')
        # report the result
        print(f'[!] best result: {self.gbest}')

    def save_rest(self, path):
        '''
        save the gbest and the corresponding particles
        '''
        with open(path, 'wb') as f:
            #pickle.dump((self.gbest, self.gbest_solution), f)
            pickle.dump(self.bestList, f)

