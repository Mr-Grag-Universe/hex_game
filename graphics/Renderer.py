from typing import *

class Game:
    pass

class Renderer:
    def __init__(self, canvas):
        self.canvas = canvas

    def render(self, game : Game):
        pass

    def save(self, path : str = 'game_record.gif'):
        pass