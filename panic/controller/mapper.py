import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
from collections import deque
from sys import argv
from model import PathNode


road_width = 60

# in dots
zone_length = 3

subpltcol = 3
subpltrow = 3

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
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 15

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01



detector = cv.SimpleBlobDetector_create(params)
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
    # img = cv.imread(argv[1])
# hsv = cv.flip(cv.cvtColor(img, cv.COLOR_BGR2HSV), 0)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# img = Image.fromarray(np.uint8(img))
    thresholded = cv.bitwise_not(cv.inRange(hsv, (0, 0, 100), (255, 30, 255)))
    yellowed = (cv.inRange(hsv, (20, 70, 70), (40, 255, 255)))

    showimg(cv.cvtColor(img, cv.COLOR_BGR2RGB), "src")

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


    for i in range(1, num_labels):
        specific = (split == i).astype(int)
        intersections.append(specific)
        intersection_nodes.append([])


    # def imshow_components(labels):
    #     # Map component labels to hue val
    #     label_hue = np.uint8(179*labels/np.max(labels))
    #     blank_ch = 255*np.ones_like(label_hue)
    #     labeled_img = cv.merge([label_hue, blank_ch, blank_ch])
    #
    #     # cvt to BGR for display
    #     labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)
    #
    #     # set bg label to black
    #     labeled_img[label_hue==0] = 0
    #
    #     cv.imshow('labeled.png', labeled_img)
    #     cv.waitKey()


    blobs = detector.detect(thresholded)
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



    finished = False
    done = []
    dot_series = [[]]
    while not finished:
        # find a starting point:
        start = None
        for v in nodes.values():
            if v not in done and len(v.conns) == 1:
                start = v
        if start is None: break
        dot_series[-1].append(start)
        last = start
        done.append(start)
        nxt = list(start.conns)[0]
        while True:
            dot_series[-1].append(nxt)
            done.append(nxt)
            if len(nxt.conns) == 1: break
            if list(nxt.conns)[0] == last:
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
    for series in dot_series:
        for i in range(len(series) - 1):
            plt.plot([series[i].x, series[i+1].x], [series[i].y, series[i+1].y], colours[series_idx % len(colours)], linestyle=':')

        series_idx += 1

    subpltind += 1

# plt.show()

    plt.subplot(subpltrow, subpltcol, subpltind)
    plt.title("screaming")


    path_series = []


    for series in dot_series:
        s1 = []
        s2 = []
        # hoo boy maths time
        for i in range(len(series)):
            # get the two points
            p1 = series[i-1] if i != 0 else series[0]
            p2 = series[i+1] if (i+1) < len(series) else series[i]
            # get gradient
            dy = p1.y - p2.y
            dx = p1.x - p2.x
            if dx == 0:
                dx = 0.00001
            if dy == 0:
                dy = 0.00001
            orig_m = dy/dx
            
            m = (orig_m**(-1))*-1
            x1 = series[i].x
            y1 = series[i].y
            l = road_width / 2

            # pronounced th-eyyyy-ta
            theta = math.atan(m)
            xv = l * math.cos(theta)
            yv = l * math.sin(theta)
            
            # plt.scatter([x1 + xv, x1 - xv], [y1 + yv, y1 - yv])

            n1 = PathNode(x1 + xv, y1 + yv, series[i])
            # if len(s1) != 1:
                # s1[-1].add_conn(s1[-2])
            n2 = PathNode(x1 - xv, y1 - yv, series[i])

            

            if len(s1) == 0:
                s1.append(n1)
                s2.append(n2)
            elif n1.dist(s1[-1]) + n2.dist(s2[-1]) < n1.dist(s2[-1]) + n2.dist(s1[-1]):
                s1.append(n1)
                s2.append(n2)
            else:
                s1.append(n2)
                s2.append(n1)

            if len(s1) != 1:
                s1[-1].add_conn(s1[-2])
                s2[-1].add_conn(s2[-2])
        path_series.append(s1)
        path_series.append(s2)
    print(yellowed)


    

# add inodes
    for series in path_series:
        if len(series) == 0: continue # fix dis
        # first one
        # intersection mafs
        # we BACKtrack along orig_m
        # first, if it's already in an intsect, we need not bother
        if yellowed[round(series[0].y), round(series[0].x)] != 0:
            # find which one
            for j in range(len(intersections)):
                if j[series[0].y, series[0].x] != 0:
                    series[0].isection_ind = j
                    intersection_nodes[j].append(series[0])
                    break
            continue

        x1 = series[0].x
        y1 = series[0].y
        dy = series[0].y - series[1].y
        dx = series[0].x - series[1].x
        if dx == 0: dx = 0.00001
        if dy == 0: dy = -0.00001
        m = dy/dx
        theta = math.atan(abs(m))
        xv = 0
        yv = 0
        l = 1
        # try:
        while yellowed[round(y1 - yv), round(x1 - xv)] == 0:
            # TODO: What in the absolute sodding hell does this do
            # and why does it work so well
            xv = l * math.cos(theta) * sign(dx) * -1
            yv = l * math.sin(theta) * sign(dy) * -1
            l += 1

        # now we have to find which one

        # move in a little bit
        l += road_width/8
        xv = l * math.cos(theta) * sign(dx) * -1
        yv = l * math.sin(theta) * sign(dy) * -1

        inode = None
        for j in range(len(intersections)):
            if intersections[j][round(y1 - yv), round(x1 - xv)] != 0:
                inode = PathNode(x1 - round(xv), y1 - round(yv), series[i])
                inode.isection_ind = j
                inode.add_conn(series[0])
                intersection_nodes[j].append(inode)
                break

        if inode != None:
            series.insert(0, inode)
        else:
            print("aaaa")
        
        # AGAIN but now last
        # TODO: dedup logic
        if yellowed[round(series[-1].y), round(series[-1].x)] != 0:
            # find which one
            for j in range(len(intersections)):
                if j[series[-1].y, series[-1].x] != 0:
                    series[-1].isection_ind = j
                    intersection_nodes[j].append(series[-1])
                    break
            continue

        x1 = series[-1].x
        y1 = series[-1].y
        dy = series[-1].y - series[-2].y
        dx = series[-1].x - series[-2].x
        # if dx == 0: dx = 0.00001
        # if dy == 0: dy = 0.00001
        m = dy/dx
        theta = math.atan(abs(m))
        xv = 0
        yv = 0
        l = 1
        # try:
        while yellowed[round(y1 + yv), round(x1 + xv)] == 0:
            # TODO: What in the absolute sodding hell does this do
            # and why does it work so well
            xv = l * math.cos(theta) * sign(dx)
            yv = l * math.sin(theta) * sign(dy)
            l += 1
            if abs(round(y1 + yv)) >= yellowed.shape[0] or abs(round(x1 + xv)) >= yellowed.shape[1]: break

        # overshoot just a little, maybe?
        # xv = (l+2) * math.cos(theta) * sign(dx)
        # yv = (l+2) * math.sin(theta) * sign(dy)

        l += road_width/8
        xv = l * math.cos(theta) * sign(dx)
        yv = l * math.sin(theta) * sign(dy)

        # now we have to find which one
        inode = None
        for j in range(len(intersections)):
            if intersections[j][round(y1 + yv), round(x1 + xv)] != 0:
                inode = PathNode(x1 + round(xv), y1 + round(yv), series[i])
                inode.isection_ind = j
                inode.add_conn(series[-1])
                intersection_nodes[j].append(inode)
                break

        if inode != None:
            series.append(inode)
        else:
            print("bbb")
            # guess i guess
            # l = 30
            # xv = l * math.cos(theta) * sign(dx)
            # yv = l * math.sin(theta) * sign(dy)
            # inode = PathNode(x1 + round(xv), y1 + round(yv), series[i])
            # inode.isection_ind = -2
            # series.append(inode)



# plt.show()

    colours = ['red', 'orange', 'green', 'blue', 'purple']
    series_idx = 0
    for series in path_series:
        if len(series) == 0: continue
        i = -1
        if series[i+1].isection_ind != -1:
            plt.scatter([series[i+1].x], [series[i+1].y], c='orange')
        else:
            plt.scatter([series[i+1].x], [series[i+1].y], c='purple')
        for i in range(len(series) - 1):
            plt.plot([series[i].x, series[i+1].x], [series[i].y, series[i+1].y], colours[series_idx % len(colours)], linestyle=':')
            if series[i+1].isection_ind != -1:
                plt.scatter([series[i+1].x], [series[i+1].y], c='red')
            else:
                plt.scatter([series[i+1].x], [series[i+1].y], c='blue')

        series_idx += 1

    subpltind += 1

# now we link up the intersections
# go through each
    for nodes in intersection_nodes:
        if len(nodes) == 0: continue
        # pull the same annoying circular linking trick
        q = deque(nodes[1:])
        nodes_linear = [nodes[0]]
        while len(q) != 0:
            # find the closest that isnt already in the list
            mindist = math.inf
            mn = None
            for node in q:
                dist = math.sqrt((node.x - nodes_linear[-1].x)**2 + (node.y - nodes_linear[-1].y)**2)
                if dist < mindist:
                    mindist = dist
                    mn = node

            # now, connect it to odd ones plz
            q.remove(mn)
            for inverse_idx in range(-1, -len(nodes_linear) - 1, -2):
                nodes_linear[inverse_idx].add_conn(mn)

            nodes_linear.append(mn)

        # for i in range(len(nodes_linear)):
            # nodes_linear[i].add_conn(nodes_linear[(i+1)%len(nodes_linear)])
            # for b in range(i):
                # if i%2 != b%2:
                    # nodes_linear[i].add_conn(nodes_linear[b])

    plt.figure()

# plt.subplot(subpltrow, subpltcol, subpltind)
# plt.title("chorus of misery")

    plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))

    all_of_them = set()
    for series in path_series:
        for node in series:
            all_of_them.add(node)

# plot this garbage
    for node in list(all_of_them):
        for other in node.conns:
            plt.plot([node.x, other.x], [node.y, other.y], 'red', linestyle=':')





# showimg(res, "yellowed")
    plt.show()