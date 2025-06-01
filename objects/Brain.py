import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

class Brain:
    def __init__(self, nn, emb_class):
        self.nn = nn
        self.emb_class = emb_class
        self.class2token = {
            'none' : -1,
            'knight' : 0,
            'archer' : 1
        }
    
    def think(self, data):
        '''
            data имеет формат словаря вида: {position : [...], class: [...], 'health' : [...]}
            переводит в формат: [padded(position), pad(emb_classes(class)), health]
        '''
        positions = torch.from_numpy(np.array(data['position']).flatten())
        classes_tokens = [self.class2token[c] for c in data['class']]
        classes = self.emb_class(torch.tensor(classes_tokens)).flatten()
        healthes = torch.tensor(data['health'])

        positions_p = F.pad(positions, (0, 8*2 - positions.size(0)), value=0)
        classes_p   = F.pad(classes, (0, 8*128 - classes.size(0)), value=0)
        healthes_p  = F.pad(healthes, (0, 8 - healthes.size(0)), value=0)

        # print(positions_p.shape, classes_p.shape, healthes_p.shape)
        input = torch.cat([positions_p, healthes_p, classes_p]).to(dtype=torch.float)
        # print(input.shape)

        return self.nn(input)

class PreNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1048, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)

        self.state_proj = nn.Linear(1048, 128)

    def forward(self, x):
        # предполагаем, что x - западдили до 128
        p = self.state_proj(x)

        out = torch.selu(self.fc1(x)*p + p)
        out = torch.selu(self.fc2(out)*p + p)
        out = self.fc2(out)*p + torch.relu(p)

        return out

# class ArcherNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.actions = ['move', 'attack']
#         self.fc = nn.Linear(128, len(self.actions))
    
#     def forward(self, x):
#         return self.fc(x)

# class KnightNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.actions = ['move', 'attack']
#         self.fc = nn.Linear(128, len(self.actions))
    
#     def forward(self, x):
#         return self.fc(x)

class ArcherBrain(Brain):
    def __init__(self):
        self.actions = ['move', 'attack']
        self.chose_action_head = nn.Linear(128, len(self.actions))
        self.actions_heads = {
            'move' : nn.Linear(128, 2),
            'attack' : nn.Linear(128, 1)
        }
        self.prenet = PreNet()
        super().__init__(self.prenet, nn.Embedding(3, 128))
    
    def think(self, data):
        pred = super().think(data)

        action = self.chose_action_head(pred)
        action = self.actions[torch.argmax(action)]
        
        match action:
            case 'move':
                pred = self.actions_heads['move'](pred)
                x, y = tuple(pred)
                # проецируем, если вышли за границы поля
                x = torch.clip(x, 0, data['field']['x_max'])
                y = torch.clip(y, 1, data['field']['y_max'])
                # x до ближайшей оси x
                x = round(float(x))
                # y до ближайшей оси y
                y = round(float(y) * 2) / 2
                params = {'pos' : (x, y)}
            case 'attack':
                params = None
        
        return {'type' : action, 'params' : params}

class KnightBrain(Brain):
    def __init__(self):
        self.actions = ['move', 'attack']
        self.chose_action_head = nn.Linear(128, len(self.actions))
        self.actions_heads = {
            'move' : nn.Linear(128, 2),
            'attack' : nn.Linear(128, 1)
        }
        self.prenet = PreNet()
        super().__init__(self.prenet, nn.Embedding(3, 128))
    
    def think(self, data):
        pred = super().think(data)
        action = self.chose_action_head(pred)
        action = self.actions[torch.argmax(action)]
        
        match action:
            case 'move':
                pred = self.actions_heads['move'](pred)
                x, y = tuple(pred)
                # проецируем, если вышли за границы поля
                x = torch.clip(x, 0, data['field']['x_max'])
                y = torch.clip(y, 1, data['field']['y_max'])
                # x до ближайшей оси x
                x = round(float(x))
                # y до ближайшей оси y
                y = round(float(y) * 2) / 2
                params = {'pos' : (x, y)}
            case 'attack':
                params = None
        
        return {'type' : action, 'params' : params}

if __name__ == '__main__':
    brain = ArcherBrain()
    action = brain.think({'position' : [(1, 1.), (2, 2.5)], 'class' : ['archer', 'knight'], 'health' : [0.9, 0.4]})
    print(action)
    