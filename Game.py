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
        self.max_iter = max_iter
        self.record = record
        self.renderer = renderer
        self.config = config

        self.team_counter = CyclicCounter(len(teams))

    def step(self, warriors : Optional[Iterable] = None):
        '''
            randomly chose 1 warrior from current team or from any passed sequence of warriors
            ask warrior for action till it's now enough
        '''
        if warriors is None:
            warriors = self.teams[self.team_counter.get()]
            self.team_counter.inc()
        
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

    def process_action(self, warrior, action, actions):
        match action['type']:
            case 'move':
                acc_sum = 0 if len(actions['move'])==0 else actions['move'][-1]['info']['accum_dist']
                print(f"{acc_sum=}")
                d = action['params']['dist']
                info = {'accum_dist' : acc_sum + d}
                
                # перемещаем игрока
                x, y = warrior.pos
                warrior.pos = x+d, y

                # смотрим, исчерпан ли лимит ходьбы
                if acc_sum+d >= self.config.data["game"]['max_dist']['other']:
                    return True, info
            case 'attack':
                info = {}
            case _:
                raise "wrong action type"
        return False, info

    def collect_data(self):
        data = {'position' : [], 'class' : [], 'health' : []}
        for team in self.teams:
            for w in team:
                info = w.get_info()
                data['position'].append(info['position'])
                data['class'].append(info['class'])
                data['health'].append(info['health'])
        return data

    def get_warriors(self):
        return [w for team in self.teams for w in team]
    
