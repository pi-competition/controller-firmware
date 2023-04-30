import controller.utils.bufferless as bufferless
import cv2 as cv
import math
import controller.shared
from matplotlib import pyplot as plt


dic = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

v = bufferless.BufferlessVideoCapture(0)
try:
    artagger = cv.aruco.ArucoDetector(dic)
    ardetector = lambda img: artagger.detectMarkers(img)
except:
    ardetector = lambda img: cv.aruco.detectMarkers(img, dic)

def fishOutArucoTags(img):

    corners, ids, _ = ardetector(img)
    centers = []
    idx = 0
    for tag in corners:
        avx = 0
        avy = 0
        for corner_x, corner_y in tag[0]:
            avx += corner_x
            avy += corner_y

        avx /=4
        avy /=4
        centers.append((avx, avy, idx))
        idx += 1
        # plt.scatter([avx],[avy])

    return (corners, ids, centers)

def bearingBetween2Points(s_x, s_y, d_x, d_y):
    dx = d_x - s_x
    dy = d_y - s_y
    if dx == 0: dx = 0.001
    theta = math.atan(abs(dy/dx))
    if dx < 0:
        if dy < 0:
            return (3/2) * math.pi - theta
        else:
            return (3/2) * math.pi + theta
    elif dy < 0:
        return (1/2) * math.pi + theta
    else:
        return (1/2) * math.pi - theta

tick = 0

def updateCamera():
    global tick
    if controller.shared.debug: return

    img = v.read()
    height, width = img.shape[:2]
    img = cv.warpPerspective(img, controller.shared.mtx, (width, height), flags=cv.INTER_LINEAR)

    corners, ids, centers = fishOutArucoTags(img)

    controller.shared.camimg = cv.aruco.drawDetectedMarkers(img, corners, ids)

    for x, y, idx in centers:
        print("THERES A CAR YO")
        print("at", x, y, "and id is", ids[idx])
        print(corners[idx], corners[idx][0])
        tl_x, tl_y = corners[idx][0][0]

        theta = bearingBetween2Points(x, y, tl_x, tl_y)

        if (not controller.shared.headless) and tick % 5 == 0:
            
            fig, ax = plt.subplots()
            ax.imshow(img)
            ax.plot([centers[idx][0]], [centers[idx][1]], color="green", marker="o")
            length = 500
            x_ = centers[idx][0] + length * math.sin(theta)
            y_ = centers[idx][1] + length * math.cos(theta)

            ax.plot([x_], [y_], color="blue", marker="o")

        
            if ids[idx][0] in controller.shared.cars.keys() and not controller.shared.cars[ids[idx][0]].immediate_target is None:
                ax.plot([controller.shared.cars[ids[idx][0]].immediate_target.x], [controller.shared.cars[ids[idx][0]].immediate_target.y], color="red", marker="o")

            print(theta)

            plt.show()

        tick += 1
        


        # we have ongle! and pos!
        if ids[idx][0] in controller.shared.cars:

            controller.shared.cars[ids[idx][0]].updatePos(centers[idx][0], centers[idx][1], theta)
        else:
            print("its not connected :(")


