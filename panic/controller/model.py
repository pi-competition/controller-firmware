import math
from controller import pathfinder
import requests 

class PathNode:
    def __init__(self, x, y, src):
        self.x = x
        self.y = y
        self.src = src
        self.conns = set()
        self.isection_ind = -1
        self.distances = {}
        self.zone = None
    
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
    def __init__(self, ip):
        self.ip = ip
        self.node = None
        self.dest = None
        self.immediate_target = None
        self.x = None
        self.y = None
        self.path = None
        self.angle = None
        self.zbounce = 0
        self.is_transient = False
        self.target_zone = None

        self.enabled = True
        self.setEnabled(False)
        # BEN DO WORK HERE

    def summarise(self):
        return {"x":int(self.x), "y":int(self.y), "imm_target":{"x": int(self.immediate_target.x), "y":int(self.immediate_target.y)}, "dest":{"x":int(self.dest.x), "y": int(self.dest.y)}}

    def setEnabled(self, enabled):
        if enabled == self.enabled: return
        self.enabled = enabled
        requests.get("http://" + self.ip + ":5001/api/" + ("enable" if enabled else "disable"))

    def __hash__(self):
        return self.ip.__hash__()

    def updatePos(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle
        print("Posting api/pos on", self.ip)
        print(angle)
        requests.post("http://" + self.ip + ":5001/api/pos", json={"x": int(x), "y": int(y), "angle": float(angle)})

    def updateTarget(self, node):
        self.immediate_target = node
        print("Posting api/target on", self.ip)
        requests.post("http://" + self.ip + ":5001/api/target", json={"x": int(node.x), "y": int(node.y)})


class Graph:
    def __init__(self, nodes, graph):
        self.nodes = set(nodes)
        self.zones = graph#        print("GRAPH", len(self.zones))
        self.dependency_graph = {} # thing, reqires_list_free
        self.place_locks = {} # car, {place, countdown}
        # self.current_paths = {} # car, [places]
        # cached_distances = {}
    
    def countdownPlaceLocks(self, car):
        for k in self.place_locks[car].keys():
            if self.place_locks[car][k] <= 1: del self.place_locks[car][k]
            else: self.place_locks[car][k] -= 1

    def fromPosToClosestNode(self, x, y, thresh):
        out = []
        for node in self.nodes:
            dist = math.sqrt((node.x - x)**2 + (node.y - y)**2)
            if dist < thresh:
                out.append(node)
        
        return out

    def fromPosToClosestNodeSingular(self, x, y):
        mindist = math.inf
        out = None
        for node in self.nodes:
            dist = math.sqrt((node.x - x)**2 + (node.y - y)**2)
            if dist < mindist:
                mindist = dist
                out = node
        
        return out

class Zone:
    def __init__(self, nodes):
        self.nodes = nodes
        for node in nodes:
            node.zone = self

       # self.throughpath, self.throughdist = pathfinder.nodeToNodeSearch(nodes[0], nodes[-1])
        self.throughdist = 10
        self.throughpath = nodes
        # self.conns = set()
        self.car_within = None

        
        self.conns = set()
        # if self.end_other_zone != None:
        # if not other.zone is None: other.zone.conns.add(self)

    def recompute(self):
        self.start_other_zone = None
        self.start_other_node = None
        for i in list(self.throughpath[0].conns):
            if i.zone != self:
                # the one
                self.start_other_zone = i.zone
                self.start_other_node = i
        self.end_other_zone = None
        self.end_other_node = None
        for i in list(self.throughpath[-1].conns):
            if i.zone != self:
                # the one
                self.end_other_zone = i.zone
                self.end_other_node = i
        self.end_other_zone.conns.add(self)
        self.start_other_zone.conns.add(self)
        self.conns.add(self.end_other_zone)
        self.conns.add(self.start_other_zone)


    def set_car(self, car):
        self.car_within = car

    def __hash__(self):
        return self.nodes[0].__hash__()

    # def add_conn(self, zn):
        # self.conns.add(zn)
        # zn.conns.add(self)

    def nodeToNextZone(self, node, next_zone):
        # this isn't too bad
        node_idx = self.throughpath.index(node)
        # special cases first
        # if node_idx == 0 or node_idx == len(self.nodes) - 1:
        #     # one of the connections is to another zone
        #     other_zone = None
        #     other_node = None
        #     for i in list(node.conns):
        #         if i.zone != self:
        #             # the one
        #             other_zone = i.zone
        #             other_node = i
        #     if other_zone == next_zone: return [other_node]
        #     else:
        #         # the entire shenanigans
        #         if node_idx == 0:
        #             return self.throughpath[1:]
        #         else:
        #             return list(reversed(self.throughpath[:-1]))

        # okay general case

        if self.start_other_zone == next_zone:
            # we need to make it go to the start
            return list(reversed(self.throughpath[:node_idx])) + [self.start_other_node]
        else:
            return self.throughpath[node_idx:] + [self.end_other_node]


class Intersection:
    def __init__(self, nodes):
        self.nodes = nodes
        self.dists = {} # (n, n) : int
        self.conns = set()
        for i in range(len(self.nodes)):
            for other in (self.nodes[:i] + self.nodes[i+1:]):
                # if not other.zone is None: other.zone.conns.add(self)
                self.dists[(self.nodes[i], other)] = self.nodes[i].dist(other)
        for node in nodes:
            node.zone = self
        # self.conns = set()
        self.is_free = True

    def __hash__(self):
        return self.nodes[0].__hash__()

    # def add_conn(self, zn):
        # self.conns.add(zn)
        # pn.conns.add(self)

    def nodeToNextZone(self, node, next_zone):
        # this is the ugly bit
        # first, determine the target node
        if next_zone.start_other_zone == self:
            target = next_zone.throughpath[0]
        else:
            target = next_zone.throughpath[-1]

        # now we djikstra
        path, _ = pathfinder.nodeToNodeSearch(node, target)
        if len(path) == 1:
            print("oh no")
            return path
        return path[1:]

