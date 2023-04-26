# from controller.mapper import mtx, fishOutArucoTags
import controller.utils.bufferless as bufferless
import cv2 as cv
import math
# import main
import controller.shared
from matplotlib import pyplot as plt
# mtx = None

# from picamera2 import Picamera2, Preview

# cam = Picamera2()
# sc = cam.create_still_configuration()
# cam.configure(sc)
# cam.configure(cam.create_preview_configuration())
# cam.start()

dic = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50)

v = bufferless.BufferlessVideoCapture(0)
try:
    artagger = cv.aruco.ArucoDetector(dic)
    ardetector = lambda img: artagger.detectMarkers(img)
except:
    ardetector = lambda img: cv.aruco.detectMarkers(img, dic)

# def previewToTakePicSetup():
    # cam.start_preview(Preview.QTGL)
    # cam.start()

# def swapAndGetImage():
    # cam.stop_preview()
    # cam.stop()
    # cam.configure(sc)
    # cam.start()
    # q = cam.capture_array()
    # return cv.cvtColor(q, cv.COLOR_RGB2BGR)

def getImage():
    return v.read()

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

# camera = bufferless.BufferlessVideoCapture(0)

def bearingBetween2Points(s_x, s_y, d_x, d_y):
    dx = d_x - s_x
    dy = d_y - s_y
    theta = math.atan(abs(dy/dx))
    if dx < 0:
        if dy < 0:
            theta = (3/2) * math.pi - theta
        else:
            theta = (3/2) * math.pi + theta
    elif dy < 0:
        theta = (1/2) * math.pi + theta
    else:
        theta = (1/2) * math.pi - theta

    return theta


def updateCamera():
    img = v.read()
    # img = cv.cvtColor(cam.capture_array(), cv.COLOR_RGB2BGR)
    # if not q:
        # print("CAMERA BORKED")
        # return False
    height, width = img.shape[:2]
    img = cv.warpPerspective(img, controller.shared.mtx, (width, height), flags=cv.INTER_LINEAR)

    corners, ids, centers = fishOutArucoTags(img)

    controller.shared.camimg = cv.aruco.drawDetectedMarkers(img, corners, ids)

    # please save me from what i have created

    # debug shizz
#    if len(centers) == 0:
#        plt.imshow(img)
#        plt.show()

    for x, y, idx in centers:
        print("THERES A CAR YO")
        print("at", x, y, "and id is", ids[idx])
        print(corners[idx], corners[idx][0])
        tl_x, tl_y = corners[idx][0][0]

        dy = tl_y - y
        dx = tl_x - x
        if dx == 0: dx = 0.0001

        """

        # pronounced they-ta
        theta = math.atan(abs(dy/dx))
        if dx < 0:
            if dy < 0:
                theta = (3/2)*math
        if dx < 0: theta = (3/2)*math.pi - theta
        else: theta = (1/2)*math.pi - theta

        # is angle to top left corner
        # we want angle straight
        # make it straight
        theta += math.pi/4
        theta %= 2*math.pi
"""
        theta = bearingBetween2Points(x, y, tl_x, tl_y)
        #theta += (1/4) * math.pi
        #theta += 2*math.pi
        #theta = theta % 2*math.pi

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.plot([centers[idx][0]], [centers[idx][1]], color="green", marker="o")
        length = 500
        x_ = centers[idx][0] + length * math.sin(theta)
        y_ = centers[idx][1] + length * math.cos(theta)

        ax.plot([x_], [y_], color="blue", marker="o")
        print(theta)

 #       plt.show()
        


        # we have ongle! and pos!
        if ids[idx][0] in controller.shared.cars:
            controller.shared.cars[ids[idx][0]].updatePos(centers[idx][0], centers[idx][1], theta)
        else:
            print("its not connected :(")


