from model import *
from pathfinder import *

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

def tellCarWhereItIs(car: Car, pos):
    pass

def tellCarWhereItsGoing(car: Car, node):
    pass

def carSetsDestination(car: Car, dest: Zone, graph):
    car.dest = Zone
    car.path = zoneToZoneSearch(car.zone, dest)
    graph.place_locks[car] = {}
    for i in range(1, len(car.path)):
        graph.place_locks[car][car.path[i]] = i

def tick(graph, cars):
    for car in cars:
        car_pos = 
