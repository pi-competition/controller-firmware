import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
from collections import deque
from sys import argv

road_width = 60

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

img = cv.imread(argv[1])
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

# plt.imshow(cv.cvtColor(yellowed, cv.COLOR_GRAY2RGB))
# plt.show()

for i in range(1, num_labels):
    specific = (split == i).astype(int)
    intersections.append(specific)
    intersection_nodes.append([])
    # plt.subplot(subpltrow, subpltcol, subpltind)
    # plt.imshow(specific) # cv.cvtColor(specific, cv.COLOR_GRAY2RGB))
    # plt.show()
    # subpltind += 1
# for i in intersections:
    # plt.subplot(subpltrow, subpltcol, subpltind)
    # plt.imshow(i)
    # subpltind += 1

def imshow_components(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue==0] = 0

    cv.imshow('labeled.png', labeled_img)
    cv.waitKey()

# imshow_components(labels_im)

# plt.imshow(labels_im)
# plt.show()

blobs = detector.detect(thresholded)
print(blobs[0].pt)

# thresholded = cv.cvtColor(thresholded, cv.COLOR_GRAY2BGR)

# im_with_keypoints = cv.drawKeypoints(thresholded, blobs, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

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


    # cv.rectangle()

# plt.subplot(subpltrow, subpltcol, subpltind)
# # plt.scatter(x, y)
# plt.title("connections 3")
# for node in nodes.values():
#     plt.scatter([node.x], [node.y])
#     for ext in node.conns:
#         plt.plot([node.x, ext.x], [node.y, ext.y], 'red', linestyle=':')
#     
#
# subpltind += 1


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

        # if len(s2) != 1:
            # s2[-1].add_conn(s2[-2])

        """
        

        a_ = (m**2 - 1)
        b_ = 2*m*y1 - 2*x1
        c_ = 2*m*y1*x1 + x1**2 + y1**2 - l**2

        a = a_
        b = (b_ - 2*a_*x1)
        c = (c_ + a_*x1**2 - b_*x1)

        print(a, b, c)
        try:
            sln_1_x = (-b+math.sqrt(b**2 - 4*a*c))/2*a
            sln_2_x = (-b-math.sqrt(b**2 - 4*a*c))/2*a
            sln_1_y = m*(sln_1_x - x1) + y1
            sln_2_y = m*(sln_2_x - x1) + y1

            plt.scatter([sln_1_x, sln_2_x], [sln_1_y, sln_2_y])
        except:
            pass

        """
    path_series.append(s1)
    path_series.append(s2)
print(yellowed)


def sign(n):
    return round(n/abs(n))

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





"""
plt.subplot(subpltrow, subpltcol, subpltind)

road_dot_series = [[]]

from collections import deque

plt.subplot(subpltrow, subpltcol, subpltind)
# plt.scatter(x, y)
plt.title("connections 1")
for curr in line_connections:
    p1 = curr.p1
    p2 = curr.p2
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linestyle=':')

subpltind += 1

plt.subplot(subpltrow, subpltcol, subpltind)
# plt.scatter(x, y)
# cries in O(n^2)
plt.title("connections 2")
conn = deque(line_connections.copy())

road_node_series = [[]]
# add a dummy to keep the counter happy
# conn.append(-1)
currq = deque([conn.popleft()])
# for point in line_connections:
finished = False
while not finished:
    curr = currq.popleft()
    road_dot_series[-1].append(curr)
    p1 = curr.p1
    p2 = curr.p2
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray', linestyle=':')
    done = False
    idx = 0
    while idx < len(conn):
        if conn[idx].p1 in [p1, p2] or conn[idx].p2 in [p1, p2]:
            currq.append(conn[idx])
            del conn[idx]
            done = True
        else:
            idx += 1
    # series done, sadge
    if len(currq) != 0: continue
    if len(conn) == 0: break
    road_dot_series.append([])
    road_node_series.append([])
    currq.append(conn.popleft())


subpltind += 1

plt.subplot(subpltrow, subpltcol, subpltind)
# plt.scatter(x, y)
# cries in O(n^2)
plt.title("connections electric bogaloo")
conn = deque(line_connections.copy())
curr = conn.popleft()
# for point in line_connections:
colours = ['gray', 'red', 'purple', 'blue', 'green']
col_idx = 0
for series in road_dot_series:
    for point in series:
        plt.plot([point.p1[0], point.p2[0]], [point.p1[1], point.p2[1]], colours[(col_idx % len(colours))], linestyle=':')
    col_idx += 1


subpltind += 1

"""

"""
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

## mask of green (36,0,0) ~ (70, 255,255)
mask1 = cv.inRange(hsv, (36, 0, 0), (70, 255,255))

## mask o yellow (15,0,0) ~ (36, 255, 255)
mask2 = cv.inRange(hsv, (15,0,0), (36, 255, 255))

## final mask and masked
mask = cv.bitwise_or(mask1, mask2)
target = cv.bitwise_and(img,img, mask=mask)
"""

# plt.imshow(img, "gray")
# plt.imshow(thresholded, "gray")
# res = cv.cvtColor(yellowed, cv.COLOR_GRAY2BGR) # 255-cv.cvtColor(thresholded, cv.COLOR_HSV2BGR)
# showimg(res, "filtered")
# showimg(res, "yellowed")
plt.show()
