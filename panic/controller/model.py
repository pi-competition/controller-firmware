import math

class PathNode:
    def __init__(self, x, y, src):
        self.x = x
        self.y = y
        self.src = src
        self.conns = set()
        self.isection_ind = -1
        # self.start = False

    def add_conn(self, pn):
        self.conns.add(pn)
        pn.conns.add(self)

    def dist(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Zone:
    def __init__(self, nodes):
        self.nodes = nodes
        self.conns = set()

    def add_conn(self, zn):
        self.conns.add(zn)
        pn.conns.add(self)
