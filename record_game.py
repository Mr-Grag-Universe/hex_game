from Game import Game
from map.Map import Map
from objects.Warriors import Knight, Archer
from Config import Config

# class GameEnv:
#     def __init__(self):
#         self.map = Map('static/map.json')
#         self.teams = [
#                         [Knight(pos=(0, 1.5)), Knight(pos=(0, 2.5)), Archer(pos=(0, 3.5))], 
#                         [Knight(pos=(8, 1.5)), Knight(pos=(8, 2.5)), Archer(pos=(8, 3.5))], 
#                      ]
#         self.config = Config('static/config.json')
#         self.game = Game(self.map, self.teams, self.config)

#     def 

def record_game(game : Game, max_iter : int = 100):
    for _ in range(max_iter):
        game.step()
    