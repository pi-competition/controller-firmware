from model import *
from typing import Dict
import math

def getMinByKey(dct: Dict[PathNode, int]) -> PathNode:
    mini: int = math.inf
    out: [PathNode | None] = None
    for (k, v) in dct:
        if v < mini:
            mini = v
            out = k

    return k

def genericBiDj(start, end, distance_fn, child_fn):
    distances_a = {}
    prevs_a = {}
    to_explore_a = {} # node, dist
    to_explore_a[start] = 0
    distances_a[start] = 0

    distances_b = {}
    prevs_b = {}
    to_explore_b = {} # node, dist
    to_explore_b[end] = 0
    distances_b[end] = 0

    swap = [(distances_a, prevs_a, to_explore_a), (distances_b, prevs_b, to_explore_b)]
    swapcounter = 0

    join = None

    while len(to_explore_a) + len(to_explore_b) != 0:
        distances, prevs, to_explore = swap[swapcounter]
        if len(to_explore) == 0: continue
        node = getMinByKey(to_explore)
        current_dist = distances[node]
        for child in child_fn(node):
            dist_to_next = distance_fn(node, child)
            distances[child] = dist_to_next + current_dist
            to_explore[child] = dist_to_next + current_dist
            prevs[child] = node
            if child in swap[1-swapcounter][0]: # check other distances
                # bingo
                join = child
                break
        swapcounter = 1 - swapcounter

    # distance summation
    total_dist = distances_a[join] + distances_b[join]
    # path unravelling
    path = []
    current = join
    while current != start:
        current = prevs_a[current]
        path.append(current)
    path.reverse()
    current = join
    while current != end:
        path.append(current)
        current = prevs_b[current]
    path.append(end)
    return (path, total_dist)

def nodeToNodeSearch(n1, n2):
    return genericBiDj(n1, n2, lambda a, b: a.dist(b), lambda n: list(n.conns))

def zoneToZoneSearch(z1, z2):
    return genericBiDj(n1. n2, lambda a, b: a.throughdist/2 + b.throughdist/2, lambda n: list(n.conns))

def findClosestNodeToPos(x, y, graph):
    mindist = math.inf
    closest = None
    for node in list(graph.nodes):
        dist = math.sqrt((x - node.x)**2 + (y - node.y)**2)
        if dist < mindist:
            mindist = dist
            closest = node

    return closest
