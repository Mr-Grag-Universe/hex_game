import numpy as np
from typing import *

from .Cell import Cell
from .Surface import Surface


class HexField:
    def __init__(self, n_width=1, n_height=1, debug=True):
        '''
            height может быть нецелым.
            если height целое, то поле "выпуклое" (края ниже максимальное высоты) 

            example:
                height, width = (3, 3)

                           *-----*
                          /       \
                   *-----* x=1;y=1 *-----*
                  /       \       /       \
                 *   x=0   *-----* i=0;j=2 *
                  \ y=1.5 /       \       /
                   *-----* x=1;y=2 *-----*
                  /       \       /       \
                 * i=1;j=0 *-----*         *
                  \       /       \       /
                   *-----*         *-----*
                          \       /
                           *-----*

                height, width = (3.5, 2)

                           *-----*
                          /       \
                   *-----*    0    *-----*
                  /       \       /       \
                 *    5    *-----*    1    *
                  \       /       \       /
                   *-----*    @    *-----*
                  /       \       /       \
                 *    4    *-----*    2    *
                  \       /       \       /
                   *-----*    3    *-----*
                  /       \       /       \
                 *         *-----*         *
                  \       /       \       /
                   *-----*         *-----*
                тут отображена нумерация соседей
        '''
        assert isinstance(n_width, int), RuntimeError("n_width must be int")
        assert n_width > 0, RuntimeError("n_width must be > 0")
        assert isinstance(n_height, int) or (isinstance(n_height, float) and int(2*n_height) == 2*n_height), \
        RuntimeError("n_height or 2*height must be int")
        assert n_height >= 1.5, RuntimeError("n_height must be >= 1")
        if n_height == 1:
            assert n_width == 1, RuntimeError("if n_height=1 then n_width must be =1")

        self.debug = debug
        
        self.n_width = n_width
        self.n_height = n_height

        self.field = self.generate_field(n_height, n_width)

    def xy_on_field(self, x, y):
        if x != int(x) or y*2 != int(y*2):
            return False
        if x < 0 or x >= self.n_width:
            return False
        if y < 1 or y > self.n_height:
            return False
        
        x_min = 0
        x_max = self.n_width-1

        y_min = 1 + int(x%2 == 0) / 2
        y_max = self.n_height
        if self.n_height == int(self.n_height):
            # если целое и фигура "выпуклая", т.е. 
            '''
                 0   1 ...
                    [ ] ...
                [ ]     ...
                    [ ] ...
                [ ]     ...
                    [ ] ...
            '''
            y_max -= int(x%2 == 0) / 2
        else:
            y_max -= int(x%2) / 2

        return x_min <= x and x <= x_max and y_min <= y and y <= y_max
        
    def ij_on_field(self, i, j):
        return self.xy_on_field(*self.ij2xy(i, j))
    
    def generate_field(self, n_height, n_width):
        M = [[Cell(*self.ij2xy(i, j)) for j in range(int(n_width))] for i in range(int(n_height))]
        return M

    def xy2ij(self, x, y):
        return int(y)-1, x

    def ij2xy(self, i, j):
        return j, i+1 + int(j%2 == 0) / 2
    
    def get_neighbour_x_y(self, x, y, direction : Literal[0, 1, 2, 3, 4, 5] = 0):
        # соседи считаются по часовой с нуля=12:00
        if self.debug:
            assert self.xy_on_field(x, y), RuntimeError("This point is not on the field")
        
        match direction:
            case 0:
                if self.debug:
                    assert y-1 >= 1, RuntimeError("there is not cell in <0> direction")
                i, j = self.xy2ij(x, y-1)
                res = self.field[i][j]
            case 1:
                if self.debug:
                    assert y-0.5 >= 1, RuntimeError("there is not cell in <1> direction")
                    assert x+1 < self.n_width, RuntimeError("there is not cell in <1> direction")
                i, j = self.xy2ij(x+1, y-0.5)
                res = self.field[i][j]
            case 2:
                if self.debug:
                    assert y+0.5 < self.n_height, RuntimeError("there is not cell in <2> direction")
                    assert x+1 < self.n_width, RuntimeError("there is not cell in <2> direction")
                i, j = self.xy2ij(x+1, y+0.5)
                res = self.field[i][j]
            case 3:
                if self.debug:
                    assert y+1 < self.n_height, RuntimeError("there is not cell in <3> direction")
                i, j = self.xy2ij(x, y+1)
                res = self.field[i][j]
            case 4:
                if self.debug:
                    assert y+0.5 < self.n_height, RuntimeError("there is not cell in <4> direction")
                    assert x-1 >= 0, RuntimeError("there is not cell in <4> direction")
                i, j = self.xy2ij(x-1, y+0.5)
                res = self.field[i][j]
            case 5:
                if self.debug:
                    assert y-0.5 >= 1, RuntimeError("there is not cell in <5> direction")
                    assert x-1 >= 0, RuntimeError("there is not cell in <5> direction")
                i, j = self.xy2ij(x-1, y-0.5)
                res = self.field[i][j]
            case _:
                # можно заменить на warning и циклический вызов
                if self.debug:
                    raise RuntimeError("there is not such direction")
                res = self.get_neighbour_x_y(x, y, direction=direction%6)

        if self.debug:
            assert self.ij_on_field(i, j), RuntimeError(f"there is not cell in <{direction}> direction (2 check)")
        return res

    def get_neighbour_i_j(self, i, j, direction : Literal[0, 1, 2, 3, 4, 5] = 0):
        return self.get_neighbour_x_y(self, *self.ij2xy(i, j), direction=direction)
    
    def dist(self, pos_1, pos_2):
        return np.sqrt((pos_1[0]-pos_2[0])**2 + (pos_1[1]-pos_2[1])**2)

    def get_neighbours(self, pos):
        N = []
        for d in range(6):
            try:
                N.append(self.get_neighbour_x_y(*pos, direction=d))
            except:
                pass
        return N

    def get_neighbours_in_order_to(self, pos):
        neighbours = self.get_neighbours(pos)
        neighbours.sort(key = lambda n : self.dist(n.pos, pos))
        return neighbours

    def set_heights(self, heights):
        for (x, y), h in heights.items():
            i, j = self.xy2ij(x, y)
            self.field[i][j].height = h

    def set_surface(self, surf):
        for (x, y), s in surf.items():
            i, j = self.xy2ij(x, y)
            self.field[i][j].surf = Surface(**s)

    def get_cells(self, mode : Literal['matrix', 'flatten'] = 'matrix'):
        match mode:
            case 'matrix':
                return self.field
            case 'flatten':
                # return [*raw for raw in self.field]
                return [cell for row in self.field for cell in row]
            case _:
                raise RuntimeError("there is not such get_cells mode!")