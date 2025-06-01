from .Surface import Surface

class Cell:
    def __init__(self, x=0, y=0, surf : Surface = Surface('none'), height : float = 0):
        self.x = x
        self.y = y
        self.surf = surf
        self.height = height

    def set_surface(self, surf):
        self.surf = surf
    
    def __repr__(self):
        return f"[x={self.x}, y={self.y}]"