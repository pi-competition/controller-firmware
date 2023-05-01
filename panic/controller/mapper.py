import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
from collections import deque
from sys import argv
from controller.model import PathNode
from controller.camera import fishOutArucoTags
import controller.shared
import networkx as nx

road_width = 60

# in dots
zone_length = 3

subpltcol = 3
subpltrow = 3

# mtx = None

subpltind = 1
def showimg(img, title):
    global subpltind
    plt.subplot(subpltrow, subpltcol, subpltind)
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    subpltind += 1



params = cv.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 1
params.maxThreshold = 500


# Filter by Area.
params.filterByArea = True
params.minArea = 40
if controller.shared.debug:
    params.maxArea = 100

params.blobColor = -1
params.filterByColor = not controller.shared.debug

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

def addIsectionNodeToEnd(node, nextNode, curr_series, intersection_nodes, labelled):
    # intersection mafs
    # we BACKtrack along orig_m
    # first, if it's already in an intsect, we need not bother
    if labelled[round(node.y), round(node.x)] != 0:
        node.isection_ind = labelled[round(node.y, node.x)]
        intersection_nodes[node.isection_ind].append(node)
        return False
        # find which one
        # for j in range(len(intersections)):
            # if j[series[0].y, series[0].x] != 0:
                # series[0].isection_ind = j
                # intersection_nodes[j].append(series[0])
                # break
        # continue

    x1 = node.x
    y1 = node.y
    dy = node.y - nextNode.y
    dx = node.x - nextNode.x
    if dx == 0: dx = 0.00001
    if dy == 0: dy = -0.00001
    m = dy/dx
    theta = math.atan(abs(m))
    xv = 0
    yv = 0
    l = 1

    precomp_cos = math.cos(theta) * sign(dx) * -1
    precomp_sin = math.sin(theta) * sign(dy) * -1
    # try:
    while labelled[round(y1 - yv), round(x1 - xv)] == 0:
        # TODO: What in the absolute sodding hell does this do
        # and why does it work so well
        xv = l * precomp_cos # * math.cos(theta) * sign(dx) * -1
        yv = l * precomp_sin # * math.sin(theta) * sign(dy) * -1
        l += 1

    # now we have to find which one

    # move in a little bit
    l += road_width/8
    xv = l * math.cos(theta) * sign(dx) * -1
    yv = l * math.sin(theta) * sign(dy) * -1

    inode = PathNode(x1 - round(xv), y1 - round(yv), curr_series)
    inode.isection_ind = labelled[round(y1 - yv), round(x1 - xv)]
    print("added to", inode.isection_ind)
    inode.add_conn(node)
    node.add_conn(inode)
    intersection_nodes[inode.isection_ind].append(inode)
    # inode = None
    # for j in range(len(intersections)):
        # if intersections[j][round(y1 - yv), round(x1 - xv)] != 0:
            # inode.isection_ind = j
            # inode.add_conn(series[0])
            # intersection_nodes[j].append(inode)
            # break

    # if inode != None:
    return inode
    # else:
        # print("aaaa")
        

    


def clipImageEdges(img):
    if controller.shared.notag: return img

    height, width = img.shape[:2]

    corners, ids, centers = fishOutArucoTags(img)

    if len(corners) != 4:
        img2 = cv.aruco.drawDetectedMarkers(img, corners);
        plt.imshow(img2)
        plt.show()
        raise Exception(f"Wrong number of aruco tags detected! - I see {len(corners)} tags. there should be 4. what are you actually doing")


    clone = img.copy()
    idxes = [0, 0, 0, 0]
    offsets = [(0, 0), (width, 0), (width, height), (0, height)]

    for i in range(len(idxes)):
        idx = -1
        mindist = math.inf
        for x, y, idx in centers:
            dist = math.sqrt((x - offsets[i][0])**2 + (y - offsets[i][1])**2)
            if dist < mindist:
                idxes[i] = idx
                mindist = dist

    # neat. now we want the corners furthest from the corner-corners
    bounding_points = [0, 0, 0, 0]
    for i in range(len(bounding_points)):
        maxdist = 0
        idx = -1
        for x, y in corners[idxes[i]][0]:
            cv.line(clone, (round(x), round(y)), offsets[i], (0, 255, 0), 5)
            dist = math.sqrt((x - offsets[i][0])**2 + (y - offsets[i][1])**2)
            if dist > maxdist:
                bounding_points[i] = (round(x), round(y))
                maxdist = dist

    # and we are done
    # bada bing bada boom
    print(bounding_points)
    for i in range(4):
        cv.line(clone, bounding_points[i - 1], bounding_points[i], (255, 0, 0), 5)

    showimg(clone, "bounded")

    mtx = cv.getPerspectiveTransform(np.float32(bounding_points), np.float32(offsets))
    warped = cv.warpPerspective(img, mtx, (width, height), flags=cv.INTER_LINEAR)

    controller.shared.mtx = mtx
    showimg(warped, "warped")
    showimg(img, "orig")

    return warped
    


detector = cv.SimpleBlobDetector_create(params) if not controller.shared.debug else cv.SimpleBlobDetector_create(params)

params.filterByArea = True

detector2 = cv.SimpleBlobDetector_create(params)

def sign(n):
    return round(n/abs(n))
class Conn:
    def __init__(self, p1, p2):
        self.p1 = min(p1, p2)
        self.p2 = max(p1, p2)
    def is_already_in(self, lst):
        for i in lst:
            if i.p1 == self.p1 and i.p2 == self.p2: return True
        return False
# now as a network of nodes
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.conns = set()

    def add_conn(self, pt):
        self.conns.add(pt)
        pt.conns.add(self)

def mapFromFilteredImg(img):
    global subpltind
    img = clipImageEdges(img)

    controller.shared.mapimg = img

    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV) if not controller.shared.debug else img

    plt.imshow(hsv); plt.show()
# BLACK SENS
    sensitivity = 150 if not controller.shared.debug else 50
    element = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    lower_white = np.array([0,0,0])
    upper_white = np.array([255,255,sensitivity])
    thresholded = cv.bitwise_not(cv.inRange(hsv, lower_white, upper_white))
    iters = 8
    if not controller.shared.debug:
        thresholded = cv.erode(cv.dilate(thresholded, element, iterations=iters), element, iterations=iters)
    print("a")
    plt.imshow(thresholded); plt.show()
    print("imshowed")
# charlie thresholds
#    lower_blue = np.array([100,50,50])
#    upper_blue = np.array([120,255,255])
    lower_blue = np.array([0, 0, 220])
    upper_blue = np.array([255, 255, 255])
    yellowed = (cv.inRange(hsv, lower_blue, upper_blue))

    showimg(cv.cvtColor(img, cv.COLOR_BGR2RGB), "src")
    showimg(cv.cvtColor(thresholded, cv.COLOR_GRAY2RGB), "thresholded")

    mask = cv.erode(yellowed, element, iterations = 4)
    mask = cv.dilate(mask, element, iterations = 4)
    yellowed = cv.erode(mask, element)

    num_labels, labels_im = cv.connectedComponents(yellowed)

    print(num_labels)
    split = np.array(labels_im)

    yellowed = np.array(yellowed)
    plt.subplot(subpltrow, subpltcol, subpltind)
    plt.title("intsects")
    plt.imshow(cv.cvtColor(yellowed, cv.COLOR_GRAY2RGB))
    subpltind += 1

    intersections = []
    intersection_nodes = []
    print("beginning detection")
    blobs = detector.detect(thresholded)
    print("ended detection")
    print(blobs)
    print(blobs[0].pt)

    x = []
    y = []
    for blob in blobs:
        _x, _y = blob.pt
        x.append(round(_x))
        y.append(round(_y))

    plt.subplot(subpltrow, subpltcol, subpltind)
    plt.scatter(x, y)
    plt.title("points")
    subpltind += 1

    points = np.column_stack((x, y))

    print(points)

    distance_connection_thresh = 1.5

    line_connections = []




    nodes = {}

# find closest points
    for point_idx in range(len(points)):
        point = list(points[point_idx])
        if (point[0], point[1]) in nodes.keys():
            node = nodes[(point[0], point[1])]
        else:
            node = Node(point[0], point[1])
            nodes[(point[0], point[1])] = node
        mindist = math.inf
        secondmin = math.inf
        pt = 0
        pt2 = 0
        for idx_2 in range(len(points)):
            if idx_2 == point_idx: continue
            p2 = points[idx_2]
            dist = math.sqrt(pow(points[idx_2][0] - point[0], 2) + pow(points[idx_2][1] - point[1], 2))
            if dist < mindist:
                secondmin = mindist
                pt2 = pt
                mindist = dist
                pt = idx_2
            elif dist < secondmin:
                secondmin = dist
                pt2 = idx_2


        # distances = np.linalg.norm((points[:point] + points[point + 1:])-point, axis=1)
        # min_index = np.argmin(distances)
        # min_index_actl = min_index + 1 if min_index >= point_idx else min_index
        print(f"the closest point to {point} is {points[pt]}") # , at a distance of {distances[min_index]}")
        p2 = points[pt]
        if (p2[0], p2[1]) in nodes.keys():
            n2 = nodes[(p2[0], p2[1])]
        else:
            n2 = Node(p2[0], p2[1])
            nodes[(p2[0], p2[1])] = n2
        node.add_conn(n2)
        # plt.plot([point[0], points[pt][0]], [point[1], points[pt][1]], 'gray', linestyle=':', marker='')
        b = Conn(point, list(points[pt]))
        if not b.is_already_in(line_connections): line_connections.append(b)
        if mindist * distance_connection_thresh > secondmin:
            p2 = points[pt2]
            if (p2[0], p2[1]) in nodes.keys():
                n2 = nodes[(p2[0], p2[1])]
            else:
                n2 = Node(p2[0], p2[1])
                nodes[(p2[0], p2[1])] = n2
            node.add_conn(n2)
            # other one is also valid
            # plt.plot([point[0], points[pt2][0]], [point[1], points[pt2][1]], 'gray', linestyle=':', marker='')
            r = Conn(point, list(points[pt2]))
            # print(r, line_connections)
            if not r.is_already_in(line_connections):
                line_connections.append(r)

    for node in nodes.values():
        for o in list(node.conns):
            plt.plot([node.x, o.x], [node.y, o.y], 'gray', linestyle=':')
    count = 0
    for node in nodes.values():
        if len(node.conns) != 1: count += 1
    # print("sjksdkjsdf")
    # print(count)
    # for (k,v) in nodes.items():
        # print(k)
        # print(v)
        # print(v.con9c83c868ns)

    plt.show()

    finished = False
    done = []
    dot_series = [[]]
    while not finished:
        # find a starting point:
        start = None
        for v in nodes.values():
            if v not in done and len(v.conns) == 1:
                start = v
                break
        if start is None: break
        dot_series[-1].append(start)
        last = start
        done.append(start)
        nxt = list(start.conns)[0]
        while True:
            print(nxt)
            dot_series[-1].append(nxt)
            done.append(nxt)
            if len(nxt.conns) == 1: break
            if list(nxt.conns)[0] in done:
                # ugly fix help me
                # if list(nxt.conns)[1] in done: break
                last = nxt
                nxt = list(nxt.conns)[1]
            else:
                last = nxt
                nxt = list(nxt.conns)[0]
        dot_series.append([])

    plt.subplot(subpltrow, subpltcol, subpltind)
    plt.title("connections 4")
    colours = ['red', 'orange', 'green', 'blue', 'purple']
    series_idx = 0
    print(len(dot_series), dot_series)
    path_series = []
    for series in dot_series:
        if len(series) == 0: continue
        path_series.append([])
        for i in range(len(series) - 1):
            plt.plot([series[i].x, series[i+1].x], [series[i].y, series[i+1].y], colours[series_idx % len(colours)], linestyle=':')
        path_series[-1].append(PathNode(series[0].x, series[0].y, series[0]))
        for i in range(1, len(series)):
            path_series[-1].append(PathNode(series[i].x, series[i].y, series[i]))
            path_series[-1][-1].add_conn(path_series[-1][-2])


        series_idx += 1

    subpltind += 1
    # path_series = [[PathNode(i.x, i.y, i) for i in q] for q in dot_series]

# plt.show()


    showimg(yellowed, "yella")
    plt.show()

    intersection_nodes = [[] for i in range(num_labels)]

    isnodes = []

    # now we need to zoning laws

    zones_all = set()


    for series in path_series:
        zones = []
        for i in range(len(series)):
            if i%zone_length == 0:
                zones.append([])
            zones[-1].append(series[i])
        for nodes in zones:
            if len(nodes) == 0: continue
            zone = controller.model.Zone(nodes)
            zones_all.add(zone)

# add inodes
    for series in path_series:
        if len(series) == 0: continue # fix dis

        inode = addIsectionNodeToEnd(series[0], series[1], series, intersection_nodes, labels_im)
        series.insert(0, inode)
        inode = addIsectionNodeToEnd(series[-1], series[-2], series, intersection_nodes, labels_im)
        series.append(inode)

# now we link up the intersections
# go through each
    isections = set()

    for intersection in intersection_nodes:
        if len(intersection) == 0: continue
        for node in intersection:
            for node2 in intersection:
                if node == node2: continue
                node.add_conn(node2)
                node2.add_conn(node)
        isection = controller.model.Intersection(intersection)
        isections.add(isection)

    for zone in list(zones_all):
        zone.recompute()

        
    colours = ['red', 'orange', 'green', 'blue', 'purple']
    series_idx = 0
    for series in path_series:
        if len(series) == 0: continue
        # i = -1
        # if series[i+1].isection_ind != -1:
            # plt.scatter([series[i+1].x], [series[i+1].y], c='orange')
        # else:
            # plt.scatter([series[i+1].x], [series[i+1].y], c='purple')
        for i in range(len(series)):
            # plt.plot([series[i].x, series[i].x], [series[i].y, series[i].y], colours[series_idx % len(colours)], linestyle=':')
            node = series[i]
            for other in list(node.conns):
                plt.plot([node.x, other.x], [node.y, other.y], linestyle=':')
            if series[i].isection_ind != -1:
                plt.scatter([series[i].x], [series[i].y], c='red')
            else:
                plt.scatter([series[i].x], [series[i].y], c='blue')

        # series_idx += 1

    subpltind += 1




    plt.figure()

# plt.subplot(subpltrow, subpltcol, subpltind)
# plt.title("chorus of misery")

    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    all_of_them = set()
    for series in path_series:
        for node in series:
            all_of_them.add(node)
    print(len(all_of_them), "total nodes")
# plot this garbage
    for node in list(all_of_them):
        for other in node.conns:
            plt.plot([node.x, other.x], [node.y, other.y], 'red', linestyle=':')

    print("here")

    
    
    net = nx.Graph()
    net.add_nodes_from(list(all_of_them))
    for node in net.nodes():
        for other in list(node.conns):
            net.add_edge(*(other, node))
    # net.add_nodes_from(list(zones_all))
    # net.add_nodes_from(list(isections))

    # for zone in net.nodes():
        # for other in list(zone.conns):
            # net.add_edge(*(other, zone))
        # i am beyond all comprehension
        # for n in zone.nodes[-1].conns:
            # if n.zone != zone: net.add_edge(*(n.zone, zone))
        # for n in zone.nodes[0].conns:
            # if n.zone != zone: net.add_edge(*(n.zone, zone))

    graph = controller.model.Graph(all_of_them, net) # list(zones_all.union(isections)))
    if not "noplot" in argv: plt.show()

    nx.draw(net)
    plt.show()

    return (graph, all_of_them, zones_all, isections)







# showimg(res, "yellowed")

if __name__ == "__main__":
    mapFromFilteredImg(cv.imread(argv[-1]))
