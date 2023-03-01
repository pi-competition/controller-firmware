import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math

subpltind = 1
def showimg(img, title):
    global subpltind
    plt.subplot(2, 2, subpltind)
    plt.imshow(img)
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    subpltind += 1

params = cv.SimpleBlobDetector_Params()
"""
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

"""

detector = cv.SimpleBlobDetector_create(params)

img = cv.imread("../road.png")
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
# img = Image.fromarray(np.uint8(img))
thresholded = cv.bitwise_not(cv.inRange(hsv, (0, 0, 100), (255, 30, 255)))


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

plt.subplot(2, 2, subpltind)
plt.scatter(x, y)
plt.title("points")
subpltind += 1

points = np.column_stack((x, y))

print(points)

distance_connection_thresh = 1.5

line_connections = []

# find closest points
for point_idx in range(len(points)):
    point = points[point_idx]
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
    # plt.plot([point[0], points[pt][0]], [point[1], points[pt][1]], 'gray', linestyle=':', marker='')
    b = np.sort([point, points[pt]])
    if not np.in1d([b], line_connections).all(): line_connections.append([point, points[pt]])
    if mindist * distance_connection_thresh > secondmin:
        # other one is also valid
        # plt.plot([point[0], points[pt2][0]], [point[1], points[pt2][1]], 'gray', linestyle=':', marker='')
        r = np.sort([point, points[pt2]])
        # print(r, line_connections)
        if not np.in1d([r], line_connections).all():
            line_connections.append([point, points[pt2]])


    # cv.rectangle()

plt.subplot(2, 2, subpltind)
# plt.scatter(x, y)
plt.title("connections")
for (p1, p2) in line_connections:
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray');

subpltind += 1

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
res = cv.cvtColor(thresholded, cv.COLOR_GRAY2BGR) # 255-cv.cvtColor(thresholded, cv.COLOR_HSV2BGR)
showimg(cv.cvtColor(img, cv.COLOR_BGR2RGB), "src")
showimg(res, "filtered")
plt.show()
