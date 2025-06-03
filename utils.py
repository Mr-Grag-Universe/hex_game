import numpy as np

dx = 3/np.sqrt(3)

class CyclicCounter:
    def __init__(self, n, start=0):
        self.n = n
        self.current = start % n
    def inc(self):
        self.current = (self.current + 1) % self.n
    def get(self):
        return self.current

def dist(pos_1, pos_2):
    return np.sqrt(((pos_1[0]-pos_2[0]) * dx)**2 + (2*(pos_1[1]-pos_2[1]))**2)

def xy2ij_obs(x, y):
    return int((y-0.5)*2)-1, int(x)
def ij2xy_obs(i, j):
    if (i+j) % 2 == 0:
        if j > 0:
            j -= 1
        else:
            j += 1
    return int(j), (i+1)/2+0.5