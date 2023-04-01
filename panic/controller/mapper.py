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

params.blobColor = -1
params.filterByColor = True

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01


def clipImageEdges(img):
    # global mtx
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



    # fish_out_these = [3, 0, 3, 3]
    # for i in range(len(bounding_points)):
        # bounding_points[i] = [round(i) for i in list(corners[idxes[i]][0][fish_out_these[i]])]

    # and we are done
    # bada bing bada boom
    print(bounding_points)
    for i in range(4):
        cv.line(clone, bounding_points[i - 1], bounding_points[i], (255, 0, 0), 5)

    showimg(clone, "bounded")
    # plt.imshow(clone)
    # plt.show()
    # print(np.fload32(offsets), np.array(bounding_points))
    mtx = cv.getPerspectiveTransform(np.float32(bounding_points), np.float32(offsets))
    warped = cv.warpPerspective(img, mtx, (width, height), flags=cv.INTER_LINEAR)

    controller.shared.mtx = mtx
    showimg(warped, "warped")
    showimg(img, "orig")

    return warped
    


detector = cv.SimpleBlobDetector_create(params)

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

    # plt.imshow(img)
    # plt.show()
    # img = cv.imread(argv[1])
# hsv = cv.flip(cv.cvtColor(img, cv.COLOR_BGR2HSV), 0)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# img = Image.fromarray(np.uint8(img))
    sensitivity = 120
    element = cv.getStructuringElement(cv.MORPH_RECT, (2,2))
    lower_white = np.array([0,0,0])
    upper_white = np.array([255,255,sensitivity])
    thresholded = cv.bitwise_not(cv.inRange(hsv, lower_white, upper_white))
    iters = 8
    thresholded = cv.erode(cv.dilate(thresholded, element, iterations=iters), element, iterations=iters)
    plt.imshow(thresholded); plt.show()
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
#     intsecblobs = detector2.detect(np.array(labels_im).astype(np.int8))
#     colours = []
# #
#     for i in intsecblobs:
#         colours.append(labels_im[i.pt])
#
#     for i in range(1, num_labels + 1):
#         if i in colours:
#             pass
#         else:
#             labels_im[labels_im == i] = 0
#
#     for i in colours:
#         labels_im[labels_im == i] = colours.index(i)
#
#     num_labels = len(colours)

    print(num_labels)
    split = np.array(labels_im)

    yellowed = np.array(yellowed)
    plt.subplot(subpltrow, subpltcol, subpltind)
    plt.title("intsects")
    plt.imshow(cv.cvtColor(yellowed, cv.COLOR_GRAY2RGB))
    subpltind += 1

    intersections = []
    intersection_nodes = []


    # for i in range(1, num_labels):
    #     specific = (split == i).astype(bool)
    #     intersections.append(specific)
    #     intersection_nodes.append([])
    #

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

    for node in nodes.values():
        for o in list(node.conns):
            plt.plot([node.x, o.x], [node.y, o.y], 'gray', linestyle=':')
    count = 0
    for node in nodes.values():
        if len(node.conns) != 1: count += 1
    print("sjksdkjsdf")
    print(count)
    for (k,v) in nodes.items():
        print(k)
        print(v)
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

    """

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
    """

    showimg(yellowed, "yella")
    plt.show()

    intersection_nodes = [[]]
    intersection_nodes = [[]]

    isnodes = []

# add inodes
    for series in path_series:
        if len(series) == 0: continue # fix dis
        # first one
        # intersection mafs
        # we BACKtrack along orig_m
        # first, if it's already in an intsect, we need not bother
        if yellowed[round(series[0].y), round(series[0].x)] != 0:
            series[0].isection_ind = yellowed[round(series[0].y, series[0].x)]
            intersection_nodes[series[0].isection_ind].append(series[0])
            # find which one
            # for j in range(len(intersections)):
                # if j[series[0].y, series[0].x] != 0:
                    # series[0].isection_ind = j
                    # intersection_nodes[j].append(series[0])
                    # break
            # continue

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
            series[0].isection_ind = yellowed[round(series[0].y, series[0].x)]
            intersection_nodes[series[0].isection_ind].append(series[0])
            # for j in range(len(intersections)):
            #     if j[series[-1].y, series[-1].x] != 0:
            #         series[-1].isection_ind = j
            #         intersection_nodes[j].append(series[-1])
            #         break
            # continue

        x1 = series[-1].x
        y1 = series[-1].y
        dy = series[-1].y - series[-2].y
        dx = series[-1].x - series[-2].x
        if dx == 0: dx = 0.00001
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
        j = yellowed[round(y1 + yv), round(x1 + xv)]
        print(j)
        # for j in range(len(intersections)):
            # if intersections[j][round(y1 + yv), round(x1 + xv)] != 0:
        inode = PathNode(x1 + round(xv), y1 + round(yv), series[i])
        inode.isection_ind = 0
        inode.add_conn(series[-1])
        isnodes.append(inode)
        # break

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
    isections = set()
    for nodes in [isnodes]: # intersection_nodes:
        """
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
        """
        if len(nodes) == 0: continue
        for node in nodes:
            for other in nodes:
                if node != other:
                    node.add_conn(other)
        isection = controller.model.Intersection(nodes)
        isections.add(isection)

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
    print(len(all_of_them))
# plot this garbage
    for node in list(all_of_them):
        for other in node.conns:
            plt.plot([node.x, other.x], [node.y, other.y], 'red', linestyle=':')

    print("here")

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

    for zone in list(zones_all):
        # i am beyond all comprehension
        for n in zone.nodes[-1].conns:
            if n.zone != zone: zone.add_conn(n.zone)
        for n in zone.nodes[0].conns:
            if n.zone != zone: zone.add_conn(n.zone)

    graph = controller.model.Graph(all_of_them, list(zones_all.union(isections)))
    if not "noplot" in argv: plt.show()

    return (graph, all_of_them, zones_all, isections)







# showimg(res, "yellowed")

if __name__ == "__main__":
    mapFromFilteredImg(cv.imread(argv[-1]))
