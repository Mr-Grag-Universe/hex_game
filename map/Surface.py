from typing import *

class Surface:
    def __init__(self, s_type : Literal['ground', 'sand', 'swamp', 'rock', 'none'], speed : float = 1.):
        self.s_type = s_type
        self.speed = speed
