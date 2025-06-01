import random
import numpy as np
import matplotlib.pyplot as plt
from typing import *

from graphics.Renderer import Renderer
from utils import CyclicCounter

class Game:
    def __init__(self, field, teams, config, max_iter=1000, record=False, renderer : Renderer = None):
        self.field = field
        self.teams = teams
        self.warrior_pos = set()
        for w in self.get_warriors():
            assert w.pos not in self.warrior_pos, RuntimeError("dublicated position")
            self.warrior_pos.add(w.pos)
        self.max_iter = max_iter
        self.record = record
        self.renderer = renderer
        if self.record:
            print("snap!")
            self.renderer.render(self)
        self.config = config

        self.team_counter = CyclicCounter(len(teams))
        self.n_iter = 0

    def step(self, warriors : Optional[Iterable] = None):
        '''
            randomly chose 1 warrior from current team or from any passed sequence of warriors
            ask warrior for action till it's now enough
        '''
        if self.n_iter >= self.max_iter:
            raise RuntimeError("n_iter is >= max_iter! reset env to continue.")
        if warriors is None:
            warriors = self.teams[self.team_counter.get()]
            self.team_counter.inc()
        self.n_iter += 1
        
        game_stopped = False
        warrior = random.choice(warriors)
        stop = False
        actions = {'move' : [], 'attack' : []}
        while not stop:
            data = self.collect_data()
            action = warrior.get_action(data)
            if action['type'] == 'attack' and len(actions['attack']):
                stop = True
                break
            print(action)
            if action is None:
                stop = True
            else:
                stop, info = self.process_action(warrior, action, actions)
                actions[action['type']].append({'params' : action['params'], 'info' : info})
                
                if self.record:
                    print("snap!")
                    self.renderer.render(self)
            if len(actions['attack']) > 0 and len(actions['move']) > 0 and actions['move'][-1]['info']['accum_sum'] >= self.config.data["game"]['max_dist']['other']:
                stop = True
        
        if self.n_iter == self.max_iter:
            game_stopped = True
        return game_stopped

    def process_action(self, warrior, action, actions):
        max_dist = self.config.data["game"]['max_dist']['other']
        match action['type']:
            case 'move':
                acc_sum = 0 if len(actions['move'])==0 else actions['move'][-1]['info']['accum_sum']
                print(f"{acc_sum=}")
                new_pos = action['params']['pos']
                print(f"{new_pos=}")
                final_pos, d = self.calc_path(warrior.pos, new_pos, max_dist=(max_dist-acc_sum))

                print("move: ", final_pos, d)
                info = {'accum_sum' : acc_sum + d}
                if final_pos == warrior.pos:
                    return True, info
                
                # перемещаем игрока
                self.warrior_pos.remove(warrior.pos)
                self.warrior_pos.add(final_pos)
                warrior.pos = final_pos

                # смотрим, исчерпан ли лимит ходьбы
                if acc_sum+d >= self.config.data["game"]['max_dist']['other']:
                    return True, info
            case 'attack':
                enemy_ind = action['params']['enemy_ind']
                enemy = self.teams[self.team_counter.get()][enemy_ind]
                enemy.health -= action['params']['damage']
                info = {}
            case _:
                raise "wrong action type"
        return False, info

    def find_path(self, start, end):
        stack = [(start, [start])]
        visited = set()
        best_path = [start]
        best_dist = self.field.hex_field.dist(start, end)

        while stack:
            current_position, path = stack.pop()
            if current_position == end:
                return best_path, best_dist
            if current_position in visited:
                continue
            visited.add(current_position)

            N = self.field.hex_field.get_neighbours_in_order_to(current_position, end)
            for neighbour in reversed(N):
                if neighbour.is_passable() and neighbour.pos not in self.warrior_pos and (neighbour.pos not in visited):
                    new_path = path + [neighbour.pos]
                    stack.append((neighbour.pos, new_path))

                    d = self.field.hex_field.dist(new_path[-1], end)
                    print(neighbour.pos, d)
                    if d < best_dist:
                        best_dist = d
                        best_path = new_path.copy()
            print(best_dist, best_path)

        return best_path, float(best_dist)

    def calc_path(self, pos, new_pos, max_dist=None):
        print("finding path...")
        path, dist = self.find_path(pos, new_pos)
        print(path)
        n = min(max_dist, len(path))
        path = path[:n]
        last_pos = path[-1]
        print(last_pos)
        return tuple(map(float, last_pos)), n-1

    def collect_data(self):
        data = {'position' : [], 'class' : [], 'health' : []}
        for team in self.teams:
            for w in team:
                info = w.get_info()
                data['position'].append(info['position'])
                data['class'].append(info['class'])
                data['health'].append(info['health'])
        data['field'] = {'x_max' : self.field.hex_field.n_width, 'y_max' : self.field.hex_field.n_height}
        data['n_enemy'] = len(self.teams[self.team_counter.get()])
        return data

    def get_warriors(self):
        return [w for team in self.teams for w in team]
    
