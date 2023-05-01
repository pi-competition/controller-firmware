from controller.model import *
from controller.pathfinder import *
import math

def carArrivesAtZone(car: Car, zone, graph):
    car.zone.car_within = None
    zone.set_car(car)
    car.zone = zone
    graph.countdownPlaceLocks(car)

def carAsksForZoneConsent(car: Car, zone, graph):
    # check the numbers
    countdown = graph.place_locks[car][zone]
    min_cnt = countdown
    other = None
    for (c, v) in graph.place_locks.items():
        if c != car:
            if zone in v:
                if v[zone] < min_cnt:
                    # uh oh
                    other = c
                    min_cnt = v[zone]

    if other != None:
        return (False, min_cnt)
    return (True, None)

def tellCarWhereItIs(car: Car, x, y, angle):
    car.updatePos(x, y, angle)

def tellCarWhereItsGoing(car: Car, node):
    car.updateTarget(node.x, node.y)

def carSetsDestination(car: Car, dest: Zone, graph):
    car.dest = dest
    car.path = zoneToZoneSearch(graph, car.zone, dest)
    print("Car path:", car.path)
    print(len(car.path))
    print(car.zone, dest)
    graph.place_locks[car] = {}

    car.updateTarget(dest.nodes[math.floor(len(dest.nodes) / 2)])

def tick(graph, cars):
    for k, car in cars.items():
        print("comms ticking")
        if car.x is None:
            print("uninitted car", k, car)
            car.setEnabled(False)
            continue
        if car.zone == car.dest and (not (car.zone is None)):
            # it is arrived
            print("congrations you are arrived")
            car.path = None
            car.dest = None
            car.setEnabled(False)
            continue
        if not car.zone is None:
            
            if car.dest is None:
                print("dest none")
                car.setEnabled(False)
                continue
            if car.path is None:
                print("path none")
                car.setEnabled(False)
                continue
        print("car running something")
        #     car.zbounce = -1 - car.zbounce
        #     carSetsDestination(car, list(graph.zones)[car.zbounce], graph)
        # while car.zone == car.dest or (car.path.index(car.zone) + 1) >= len(car.path):
        #     car.zbounce = -1 - car.zbounce
        #     carSetsDestination(car, list(graph.zones)[car.zbounce], graph)
        #     print("loopin")
        car_nodes = graph.fromPosToClosestNode(car.x, car.y, 128)
        maxidx = -1
        node__ = None
        for node in car_nodes:
            if not node.zone in car.path: continue
            idx = car.path.index(node.zone)
            if idx > maxidx:
                maxidx = idx
                node__ = node

        if node__ is not None:
            print("speedy boi")

        car_nodes = graph.fromPosToClosestNodeSingular(car.x, car.y)
        maxidx = -1
        node_ = node__
        node_ = car_nodes if node_ is None else node_
        # for node in car_nodes:
            # if not node in car.path: continue
            # idx = car.path.index(node)
            # if idx > maxidx:
                # maxidx = idx
                # node_ = node
        if node_ is None:
            print("significant deviation uhhhh")
            continue
        car_node = node_
        car_zone = car_node.zone
        if car.zone is None:
            car.zone = car_zone
            continue
        if car_zone != car.zone:
            print("Car has non-consentually entered a zone")
            car.zone = car_zone
            if car.path is None or not car.zone in car.path:
                if car.dest is None:
                    # TODO: ben do work here
                    car.dest = list(graph.zones.nodes)[0 if car.zone == list(graph.zones.nodes)[-1] else -1]
                carSetsDestination(car, car.dest, graph)


        car.setEnabled(True)
        # we know where the car is
        # do we know where the car is going?
        # yes we do

        # trans checks
        if car.is_transient:
            if car.zone == car.target_zone:
                car.is_transient = False
                car.target_zone = None
            else:
                # its working on it okay
                continue
        
        next_zone = car.path[car.path.index(car.zone) + 1]
        # here comes the awful logic
        # find the next node we need to visit
        node_nodes = car.zone.nodeToNextZone(car_node, next_zone)
        print("Nodes left", len(node_nodes))
        if len(node_nodes) < 4:
            # move on!
            car.target_zone = next_zone
            # find the node of interest
            noi = None
            for node in list(car.target_zone.nodes):
                for other in list(node.conns):
                    if other.zone == car.zone:
                        # whoo
                        noi = node
                        break
                if noi is not None: break

            if noi is None:
                print("this is not good")
                continue
            car.immediate_target = noi
            car.is_transient = True
            car.updateTarget(car.immediate_target)
        next_node = car.zone.nodeToNextZone(car_node, next_zone)[0]
        if car.immediate_target != next_node:
            car.immediate_target = next_node
            car.updateTarget(next_node)

        # now we issue a correction
        # TODO: do this logic car-side
        dx = next_node.x - car.x
        if dx == 0: dx = 0.0001
        dy = next_node.y - car.y
        # pronounced they-ta
        theta = math.atan(dy/dx)
        # now, compensate into a bearing
        if dx < 0:
            # negi, we work off 3/2 pi (270)
            # and then add mafs
            theta = (3/2) * math.pi  - theta
        else:
            theta = (1/2) * math.pi - theta

        
        # we find the next node
        # there are 2 possibilities
        # node_idx = None
        # for i in range(len(car.zone.nodes)):
        #     if car.zone.nodes[i] == car_node:
        #         node_idx = i
        #         break
        # # if not, oh no
        # if node_idx is None:
        #     # oh no
        #     car.zone = car_node.zone
        #     for i in range(len(car.zone.nodes)):
        #         if car.zone.nodes[i] == car_node:
        #             node_idx = i
        #             break


