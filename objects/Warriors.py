from .GameObject import GameObject

class Warrior(GameObject):
    def __init__(self, pos):
        super().__init__(pos)
        print("warrior: ", self.pos)
        self.health = 1.0

    def get_action(self, data):
        '''
            None if skip step
            {'type' : ['move'], 'params' : ...}
        '''
        return self.brain.think(data)

    def get_info(self):
        pass

class Knight(Warrior):
    def __init__(self, pos):
        super().__init__(pos)
        print("knight: ", self.pos)
    
    def get_info(self):
        return {
                    'position' : self.pos, 
                    'class' : 'knight', 
                    'health' : self.health,
                    'damage' : 0.5
               }

class Archer(Warrior):
    def __init__(self, pos):
        super().__init__(pos)
        print("archer: ", self.pos)
    
    def get_info(self):
        return {
                    'position' : self.pos, 
                    'class' : 'archer', 
                    'health' : self.health,
                    'damage' : 0.3
               }
