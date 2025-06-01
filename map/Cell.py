from .Surface import Surface

class Cell:
    def __init__(self, x=0, y=0, surf : Surface = Surface('none'), height : float = 0):
        self.x = x
        self.y = y
        self.surf = surf
        self.height = height

        self.pos = (x, y)

    def set_surface(self, surf):
        self.surf = surf
    
    def __repr__(self):
        return f"[x={self.x}, y={self.y}]"
    
    def is_passable(self):
        return self.surf.speed != 0