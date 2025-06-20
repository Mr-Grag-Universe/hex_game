import random
import numpy as np
import matplotlib.pyplot as plt
from typing import *
import logging
import torch
import torch.nn.functional as F
from collections import deque
from functools import lru_cache
from copy import deepcopy

logging.basicConfig(
    level=logging.DEBUG,  # Уровень логирования (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Формат сообщения
    handlers=[
        logging.FileHandler('app.log'),  # Запись логов в файл
        # logging.StreamHandler()  # Вывод логов на консоль
    ]
)

from graphics.Renderer import Renderer
from utils import CyclicCounter, dist

class Game:
    def __init__(self, teams, field, config, logging=True, debug=True):
        self.field = field
        self.config = config
        self.logging = logging
        self.debug = debug
        self.teams_start = deepcopy(teams)
    
    def reset(self, max_iter=1000, record=False, renderer : Renderer = None):
        self.teams = deepcopy(self.teams_start)
        self.warrior_pos = set()
        for w in self.get_warriors():
            assert w.pos not in self.warrior_pos, RuntimeError("dublicated position")
            self.warrior_pos.add(w.pos)
        self.warrior = random.choice(self.teams[0])
        self.actions = {'move' : [], 'attack' : [], 'type_log' : []}
        
        # запись
        self.record = record
        self.renderer = renderer
        if self.record:
            print("snap!")
            self.renderer.render(self)

        self.team_counter = CyclicCounter(len(self.teams))
        self.max_iter = max_iter
        self.n_iter = 0
        self.reward = 0
        return self.observe()

    def action_is_correct(self, action) -> bool:
        # если не словарь
        if not isinstance(action, dict):
            return False
        # если уже была атака
        if action['type'] == 'attack':
            if len(self.actions['attack']):
                return False
        return True

    def step_is_over(self):
        attacked = len(self.actions['attack']) > 0
        moved_max = len(self.actions['move']) > 0 and self.actions['move'][-1]['info']['accum_sum'] >= self.config.data["game"]['max_dist']['other']
        return attacked and moved_max

    def game_is_over(self):
        killed_all = (len(self.teams[0]) * len(self.teams[1]) == 0)
        return killed_all

    def next_team_warrior_prepare(self):
        self.team_counter.inc()
        self.actions = {'move' : [], 'attack' : [], 'type_log' : []}
        # print(self.teams_start, self.teams[self.team_counter.get()])
        self.warrior = random.choice(self.teams[self.team_counter.get()])

    def step(self, action):
        '''
            randomly chose 1 warrior from current team or from any passed sequence of warriors
            returns: -> reward, observation, stop_game
        '''
        if self.n_iter >= self.max_iter or self.game_is_over():
            # raise RuntimeError("n_iter is >= max_iter! reset env to continue.")
            return 0, self.reset(), True
        self.n_iter += 1
        
        game_stopped = (self.n_iter == self.max_iter)

        r_curr = 0
        # штраф за урон от соперника
        if len(self.actions['move']) == 0 and len(self.actions['attack']) == 0:
            r_curr += self.reward
            self.reward = 0
        
        game_stopped = game_stopped or self.game_is_over()
        if not self.action_is_correct(action) and not game_stopped:
            self.next_team_warrior_prepare()
            return -1. + r_curr, self.observe(), game_stopped
        
        if action['type'] == 'none':
            self.next_team_warrior_prepare()
            return r_curr, self.observe(), game_stopped

        step_stop, info, r_curr_new, r_prev, cancelled = self.process_action(action)
        r_curr += r_curr_new
        self.reward += r_prev
        if not cancelled:
            self.actions[action['type']].append({'params' : action['params'], 'info' : info})
            self.actions['type_log'].append(action['type'])

        if self.record and not cancelled:
            self.renderer.render(self)

        data = self.collect_data()
        if self.logging and not cancelled:
            logging.info(f"warrior < {self.warrior.get_info()} > do < {action['type']} > with params < {action['params']} > and info < {info} >")
        if self.game_is_over():
            game_stopped = True
            r_curr += 50
        
        step_stop = step_stop or self.step_is_over()
        if step_stop and not game_stopped:
            self.next_team_warrior_prepare()
        
        if self.n_iter == self.max_iter:
            game_stopped = True

        if game_stopped:
            logging.info(f"Game Over!")
        return r_curr, self.observe(), game_stopped

    def process_action(self, action):
        # -> step_stop, info, r_curr, r_prev, cancelled
        w_info = self.warrior.get_info()
        max_dist = self.config.data["game"]['max_dist']['other']

        r_curr, r_prev = 0, 0
        match action['type']:
            case 'move':
                acc_sum = 0 if len(self.actions['move'])==0 else self.actions['move'][-1]['info']['accum_sum']
                # print(f"{acc_sum=}")
                new_pos = action['params']['pos']
                # print(f"{new_pos=}")
                final_pos, d = self.calc_path(self.warrior.pos, new_pos, max_dist=(max_dist-acc_sum))

                # print("move: ", final_pos, d)
                info = {'accum_sum' : acc_sum + d}
                if final_pos == self.warrior.pos:
                    r_curr -= 1
                    return True, info, r_curr, r_prev, True
                
                # перемещаем игрока
                # self.warrior_pos.remove(self.warrior.pos)
                self.warrior_pos.discard(self.warrior.pos)
                self.warrior_pos.add(final_pos)
                self.warrior.pos = final_pos

                # смотрим, исчерпан ли лимит ходьбы
                if acc_sum+d >= self.config.data["game"]['max_dist']['other']:
                    return True, info, r_curr, r_prev, False
            case 'attack':
                enemy_pos = action['params']['pos']
                t = (self.team_counter.get()+1) % 2
                enemies = self.teams[t]
                enemy_nearest = enemies[0]
                d = dist(enemies[0].pos, enemy_pos)
                for enemy in enemies:
                    d_new = dist(enemy.pos, enemy_pos)
                    if d_new < d:
                        d = d_new
                        enemy_nearest = enemy
                enemy = enemy_nearest

                if w_info['class'] == 'knight':
                    if enemy.pos in self.field.hex_field.get_neighbours(self.warrior.pos, mode='pos'):
                        enemy.health -= w_info['damage']
                        if enemy.health <= 1e-5:
                            self.warrior_pos.remove(enemy.pos)
                            t = (self.team_counter.get()+1) % 2
                            self.teams[t].remove(enemy)
                            r_curr += 5
                            r_prev -= 5
                    else:
                        r_curr -= 1
                        return True, None, r_curr, 0., True
                else:
                    enemy.health -= w_info['damage']
                    if enemy.health <= 0:
                        self.warrior_pos.remove(enemy.pos)
                        t = (self.team_counter.get()+1) % 2
                        self.teams[t].remove(enemy)
                        r_curr += 5
                        r_prev -= 5
                info = {'enemy_pos' : enemy.pos, 'self_pos' : self.warrior.pos, 'enemy_type' : enemy.get_info()['class']}
            case _:
                raise RuntimeError("wrong action type")
        return False, info, r_curr, r_prev, False

    def find_path(self, start, end):
        stack = [(start, [start])]
        visited = set()
        best_path = [start]
        best_dist = dist(start, end)

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

                    d = dist(new_path[-1], end)
                    # print(neighbour.pos, d)
                    if d < best_dist:
                        best_dist = d
                        best_path = new_path.copy()
            # print(best_dist, best_path)

        return best_path, float(best_dist)

    def get_available_area(self, start, max_dist):
        queue = deque([(start, 0)])
        visited = set()

        while queue:
            current_position, d = queue.popleft()
            if d > max_dist:
                break
            visited.add(current_position)
            # if current_position in visited:
            #     continue
            
            N = self.field.hex_field.get_neighbours(current_position)
            for neighbour in N:
                if neighbour.is_passable() and (neighbour.pos not in visited) and neighbour.pos not in self.warrior_pos:
                    queue.append((neighbour.pos, d+1))

        return visited


    def calc_path(self, pos, new_pos, max_dist=None):
        # print("finding path...")
        path, dist = self.find_path(pos, new_pos)
        # print(path)
        n = min(max_dist, len(path))
        path = path[:n]
        last_pos = path[-1]
        # print(last_pos)
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
        data['self'] = self.warrior.get_info()
        return data


    def observe(self):
        # всего 
        layers = [
            self.get_self_pos_layer(),
            *self.get_warriors_layers(),
            *self.get_click_layers(),
            *self.get_surface_layers(),

            self.get_teams_layer(),
        ]
        return torch.stack(layers).to(dtype=torch.float32)

    def xy2ij_obs(self, x, y):
        return int((y-0.5)*2)-1, int(x)
    def ij2xy_obs(self, i, j):
        return int(j), (i+1)/2+0.5

    def get_click_layers(self):
        h, w = self.field.hex_field.get_size()
        w, h = int(w), int((h-0.5)*2)+1
        info = self.warrior.get_info()
        max_dist = self.config.data['game']['max_dist'][info['class']]

        move_layer = torch.zeros(h, w, dtype=torch.uint8)
        attack_layer = torch.zeros(h, w, dtype=torch.uint8)
        
        # fill move layer
        aviable_pos = self.get_available_area(self.warrior.pos, max_dist)
        for pos in aviable_pos:
            i, j = self.xy2ij_obs(*pos)
            move_layer[i, j] = 1
        pos = self.warrior.pos
        i, j = self.xy2ij_obs(*pos)
        move_layer[i, j] = 0

        # fill attack layer
        pos = self.warrior.pos
        for enemy in self.teams[(self.team_counter.get()+1) % 2]:
            match info['class']:
                case 'knight':
                    if enemy.pos in self.field.hex_field.get_neighbours(pos, mode='pos'):
                        i, j = self.xy2ij_obs(*enemy.pos)
                        attack_layer[i, j] = 1
                case 'archer':
                    i, j = self.xy2ij_obs(*enemy.pos)
                    attack_layer[i, j] = 1
                case _:
                    raise RuntimeError("wrong warrior type")

        return move_layer, attack_layer

    @lru_cache(maxsize=None)
    def get_surface_layers(self):
        h, w = self.field.hex_field.get_size()
        w, h = int(w), int((h-0.5)*2)+1
        surface_layers = [torch.zeros(h, w, dtype=torch.float16) for _ in range(1)]
        for cell in self.field.hex_field.get_cells(mode='flatten'):
            i, j = self.xy2ij_obs(*cell.pos)
            surface_layers[0][i,j] = cell.surf.speed
        
        return surface_layers

    def get_self_pos_layer(self):
        h, w = self.field.hex_field.get_size()
        w, h = int(w), int((h-0.5)*2)+1
        layer = torch.zeros(h, w, dtype=torch.uint8)
        i, j = self.xy2ij_obs(*self.warrior.pos)
        layer[i, j] = 1
        return layer

    def get_teams_layer(self):
        h, w = self.field.hex_field.get_size()
        w, h = int(w), int((h-0.5)*2)+1
        layer = torch.zeros(h, w, dtype=torch.uint8)
        for w in self.teams[self.team_counter.get()]:
            layer[self.xy2ij_obs(*w.pos)] = 1
        for w in self.teams[1-self.team_counter.get()]:
            layer[self.xy2ij_obs(*w.pos)] = 2
        return layer

    def get_warriors(self):
        return [w for team in self.teams for w in team]
    
    def get_warriors_layers(self):
        h, w = self.field.hex_field.get_size()
        w, h = int(w), int((h-0.5)*2)+1
        layer_class  = torch.zeros(h, w, dtype=torch.uint8)
        layer_health = torch.zeros(h, w, dtype=torch.float16)
        layer_damage = torch.zeros(h, w, dtype=torch.float16)
        class_map = {'knight' : 1, 'archer' : 2}
        for w in self.get_warriors():
            i, j = self.xy2ij_obs(*w.pos)
            info = w.get_info()
            layer_class[i, j] = class_map[info['class']]
            layer_health[i, j] = info['health']
            layer_damage[i, j] = info['damage']
        return [layer_class, layer_health, layer_damage]