class CyclicCounter:
    def __init__(self, n, start=0):
        self.n = n
        self.current = start % n
    def inc(self):
        self.current = (self.current + 1) % self.n
    def get(self):
        return self.current