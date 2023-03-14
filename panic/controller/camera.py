# from controller.mapper import mtx, fishOutArucoTags
import controller.utils.bufferless as bufferless
import cv2 as cv
import math
# import main
import controller.shared

# mtx = None


ardetector = cv.aruco.ArucoDetector(cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_50))

def fishOutArucoTags(img):

    corners, ids, _ = ardetector.detectMarkers(img)
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

camera = bufferless.BufferlessVideoCapture(0)

def updateCamera():
    img = camera.read()
    # if not q:
        # print("CAMERA BORKED")
        # return False
    height, width = img.shape[:2]
    img = cv.warpPerspective(img, controller.shared.mtx, (width, height), flags=cv.INTER_LINEAR)

    corners, ids, centers = fishOutArucoTags(img)

    # please save me from what i have created

    for x, y, idx in centers:
        tl_x, tl_y = corners[idx][0]

        dy = tl_y - y
        dx = tl_x - x
        if dx == 0: dx = 0.0001

        # pronounced they-ta
        theta = math.atan(dy/dx)
        if dx < 0: theta = (3/2)*math.pi - theta
        else: theta = (1/2)*math.pi - theta

        # is angle to top left corner
        # we want angle straight
        # make it straight
        theta += math.pi/4
        theta %= 2*math.pi

        # we have ongle! and pos!
        shared.cars[ids[idx]].updatePos(centers[idx], theta)

