import os
import json
from typing import *
import numpy as np

from .HexField import HexField

def map2serializable(m):
    m_json = {'heights' : None, 'surface' : None}
    
    p = np.array(list(set(m['heights'].keys())))
    x_min, y_min = p.min(0)
    x_max, y_max = p.max(0)
    M_1 = np.full((int(y_max-y_min)+1, int(x_max-x_min)+1), {}, dtype='object')
    M_2 = np.full((int(y_max-y_min)+1, int(x_max-x_min)+1), {}, dtype='object')
    for (x, y), h in m['heights'].items():
        i, j = int(y)-1, x
        M_1[i,j] = h
        M_2[i,j] = m['surface'][(x,y)]
    m_json['heights'] = M_1.tolist()
    m_json['surface'] = M_2.tolist()
    
    return m_json

def serializable2map(m_json):
    m = {'heights': {}, 'surface': {}}
    
    # Преобразуем списки обратно в массивы NumPy
    M_1 = np.array(m_json['heights'], dtype=object)
    M_2 = np.array(m_json['surface'], dtype=object)
    
    y_max, x_max = M_1.shape
    for i in range(y_max):
        for j in range(x_max):
            h = M_1[i, j]
            s = M_2[i, j]
            if h is not None:
                x = j
                y = i + 1
                m['heights'][(x, y)] = h
                m['surface'][(x, y)] = s
    return m

class Map:
    def __init__(self, src, debug=True):
        '''
        из src получаем data
        data = {'heights' : {...}, 'surface' : {...}, 'растительность' : {...}}
        '''
        if isinstance(src, dict):
            # Если src - это словарь, просто сохраняем его
            self.data = src
        elif isinstance(src, str):
            # Если src - это строка, проверяем, является ли она путем к файлу
            if os.path.isfile(src):
                # Если это файл, читаем его как JSON
                with open(src, 'r', encoding='utf-8') as file:
                    self.data = json.load(file)
            else:
                # Если это строка, пытаемся преобразовать ее в dict
                self.data = json.loads(src)
            self.data = serializable2map(self.data)
        else:
            raise ValueError("Unsupported type for src. Must be dict, str, or a valid file path.")

        points = np.asarray(list(set(self.data['heights'].keys())), dtype=float)
        x_max, y_max = points.max(axis=0)
        x_min, y_min = points.min(axis=0)
        if debug:
            assert set(self.data['heights'].keys()) == set(self.data['surface'].keys())
        
        self.hex_field = HexField(n_width=int(x_max-x_min+1), n_height=float(y_max-y_min+1), debug=True)
        self.hex_field.set_heights(self.data['heights'])
        self.hex_field.set_surface(self.data['surface'])
    
    def get_cells(self, mode : Literal['matrix', 'flatten'] = 'matrix'):
        return self.hex_field.get_cells(mode=mode)