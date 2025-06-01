from celluloid import Camera
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import *

from .Renderer import Renderer

class Game:
    pass

class Map:
    pass

class MatplotlibRenderer(Renderer):
    def __init__(self, assets):
        fig, ax = plt.subplots()
        super().__init__(canvas=ax)
        
        self.ax = ax
        self.fig = fig
        self.camera = Camera(fig)

        self.dx = 3/np.sqrt(3)

        self.assets = assets

    def render(self, game : Game):
        self.render_map(game.field)
        # self.render_obstacle(game)
        self.render_warriors(game.get_warriors())
        self.camera.snap()

    def render_map(self, game_map : Map):
        for cell in game_map.get_cells(mode='flatten'):
            x, y = cell.x, cell.y
            x_real, y_real = x*self.dx, y*2
            self.draw_hexagon(self.ax, (x_real, y_real), self.assets['surface'][cell.surf.s_type])

    def draw_hexagon(self, ax, center, assets):
        size=1
        angles = np.linspace(0, 2 * np.pi, 7)
        r = 2*size / np.sqrt(3)
        x_hexagon = center[0] + r * np.cos(angles)
        y_hexagon = center[1] + r * np.sin(angles)
        
        ax.fill(x_hexagon, y_hexagon, color=assets['color'], alpha=0.5)
        ax.plot(x_hexagon, y_hexagon, color='gold')
        # ax.plot(center[0], center[1], 'ro')
    

    def render_warriors(self, warriors : Iterable):
        x, y, c, s = [], [], [], []
        size = 200
        p = 0.5**2
        for w in warriors:
            x.append(w.pos[0]*self.dx)
            y.append(w.pos[1]*2)
            c.append(self.assets['warriors'][w.get_info()['class']])
            s.append(w.health*size*p)
        
        df = pd.DataFrame({'x' : x, 'y' : y, 'c' : c, 's' : s})
        # print(df)
        # self.ax = sns.scatterplot(df, x='x', y='y', hue='c', ax=self.ax, s=size)
        # sns.scatterplot(df, x='x', y='y', color='white', ax=self.ax, s=s)
        self.ax.scatter(x, y, c=c, alpha=1., s=size) #, edgecolors='black')
        self.ax.scatter(x, y, c='white', alpha=1., s=s)
        # plt.legend(loc='none')

    def render_warrior(self, warrior):
        pass

    def save(self, path : str = 'game_record.gif'):
        animation = self.camera.animate()
        animation.save(path, writer = 'imagemagick')
