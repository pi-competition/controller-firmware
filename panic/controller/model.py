import math
import pathfinder

class PathNode:
    def __init__(self, x, y, src):
        self.x = x
        self.y = y
        self.src = src
        self.conns = set()
        self.isection_ind = -1
        self.distances = {}
    
    # TODO: make better
    def __hash__(self):
        return f"{self.x}:{self.y}".__hash__()

    def add_conn(self, pn):
        self.conns.add(pn)
        dst = self.dist(pn)
        self.distances[pn] = dst
        pn.conns.add(self)
        pn.distances[self] = dst

    def dist(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class Car:
    def __init__(ip):
        self.ip = ip
        self.zone = None
        self.dest = None
        # BEN DO WORK HERE

    def __hash__(self):
        return self.ip.__hash__()

class Graph:
    def __init__(self, nodes):
        self.nodes = set(nodes)
        self.zones = set()
        self.dependency_graph = {} # thing, reqires_list_free
        self.place_locks = {} # car, {place, countdown}
        # self.current_paths = {} # car, [places]
        # cached_distances = {}
    
    def countdownPlaceLocks(self, car):
        for k in self.place_locks[car].keys():
            if self.place_locks[car][k] <= 1: del self.place_locks[car][k]
            else: self.place_locks[car][k] -= 1

class Zone:
    def __init__(self, nodes):
        self.nodes = nodes
        self.throughpath, self.throughdist = pathfinder.nodeToNodeSearch(nodes[0], nodes[-1])
        self.conns = set()
        self.car_within = None

    def set_car(self, car):
        self.car_within = car

    def __hash__(self):
        return self.nodes[0].__hash__()

    def add_conn(self, zn):
        self.conns.add(zn)
        pn.conns.add(self)

class Intersection:
    def __init__(self, nodes):
        self.nodes = nodes
        self.dists = {} # (n, n) : int
        for i in range(len(self.nodes)):
        for other in self.nodes[:i] + self.nodes[i+1:]:
                self.dists[(self.nodes[i], other)] = self.nodes[i].dist(other)
        self.conns = set()
        self.is_free = True

    def __hash__(self):
        return self.nodes[0].__hash__()

    def add_conn(self, zn):
        self.conns.add(zn)
        pn.conns.add(self)

