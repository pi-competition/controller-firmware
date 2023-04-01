import flask 
from flask import request, jsonify
from sys import argv
from flask_restful import Resource
from server.utils.response import success, error
from api import PANICAPI
import threading
import urllib
import json
import controller
import controller.model
import controller.camera
import controller.comms
import controller.mapper
from controller import shared
from controller import model
app = flask.Flask(__name__)
app.config["DEBUG"] = True
api = PANICAPI(app)
app.debug = False

# cars = {}

shared.cars = {}

DEVICES = {}
CONFIG = {}
app.config["DEVICES"] = DEVICES

"""
with open("config.json") as f:
    CONFIG = json.loads(f.read())
    app.config["CONFIG"] = CONFIG
"""

@app.errorhandler(404)
def page_not_found(e):
    return error("The requested resource could not be located", 404)

@app.route('/*', methods=['OPTIONS'])
def options():
    allowed_origins = ["staging.teampanic.eu.org", "teampanic.eu.org", "localhost"]
    if request.headers.get('Origin') in allowed_origins:
        return Response(status=204, headers={
            'Access-Control-Allow-Origin': request.headers.get('Origin'),
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'})
    else:
        return Response(status=204, headers={
            'Access-Control-Allow-Origin': 'teampanic.eu.org',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type, Authorization'})

from server.routes.api.ping import Ping
api.add_resource(Ping, '/api/ping')
from server.routes.api.refresh_camera import RefreshCamera
api.add_resource(RefreshCamera, '/api/refresh_camera')
from server.routes.api.verify import Verify
api.add_resource(Verify, '/api/verify')
from server.routes.api.server.restart import Restart
api.add_resource(Restart, '/api/server/restart')
from server.routes.api.cars.status import Status
api.add_resource(Status, '/api/cars/status')
from server.routes.api.cars.reset_bulk import ResetBulk
api.add_resource(ResetBulk, '/api/cars/reset-bulk')
from server.routes.ext.mapimg import MapImg
api.add_resource(MapImg, '/ext/map/img')
from server.routes.ext.maprefresh import MapRefresh
api.add_resource(MapRefresh, '/ext/map/refresh')
from server.routes.ext.mapdata import MapData
api.add_resource(MapData, '/ext/map/data')
from server.routes.ext.tps import Tps
api.add_resource(Tps, '/ext/tps')
from server.routes.ext.camimg import CamImg
api.add_resource(CamImg, '/ext/cam/img')
from server.routes.ext.cars.summary import CarSummary
api.add_resource(CarSummary, '/ext/cars/summary')

def get_devices():
    threading.Timer(60, get_devices).start()
    print("RUNNING")
    global DEVICES
    try:
        res = urllib.request.urlopen("http://127.0.0.1:5000/devices").read()
    except:
        return print("RADAR IS OFFLINE")
    DEVICES = json.loads(res.decode("utf-8"))["data"]["devices"]

    for device in DEVICES:
        if device["ip"] != None:
            try:
                res = urllib.request.urlopen(f"http://{device['ip']}:5001/api/ping").read()
                # j = json.loads(res.decode("utf-8"))
                device["info"] = json.loads(res.decode("utf-8"))["data"]
                device["status"] = device["info"]["status"]
            except:
                device["status"] = "offline"
                device["info"] = {"name": device["name"], "type": device["type"], "status": "offline"}
        else:
            device["status"] = "offline"
            device["info"] = {"name": device["name"], "type": device["type"], "status": "offline"}
    print(DEVICES)
    app.config["DEVICES"] = DEVICES
    
    for device in DEVICES:
        print(device)
        if device["status"] != "offline":
            if device["info"]["id"] not in shared.cars.keys():
                # new car!
                print("New car found!")
                car = model.Car(device["ip"])
                shared.cars[device["info"]["id"]] = car
    ips = [(device["ip"] if device["status"] != "offline" else None) for device in DEVICES]
    todel = []
    for k, car in shared.cars.items():
        if car.ip not in ips:
            # uh no
            print("lost car")
            todel.append(k)
    for i in todel:
        del shared.cars[i]
        # TODO: beter cleanup, !IMPORTANT






get_devices()
# threading.Timer(60, get_devices).start()
def run():
    app.run(port=5001, host="0.0.0.0", threaded=False)

def dev():
    app.run(port=5001, host="0.0.0.0", debug=True)

#dev()
comms_thread = threading.Thread(target=run)
comms_thread.start()

print("does this work")

import cv2
# temp_img = cv2.imread("tagged2.png")

from controller import camera
# camera.previewToTakePicSetup()
while True:
    input("Press enter to take map image")
# mapimg = temp_img
    temp_img = camera.getImage()
    from matplotlib import pyplot as plt
    plt.imshow(temp_img)
    plt.show()
# if not "noplot" in argv: plt.show()
    if "y" == input("is this your picture?"): break
graph, nodes, zones, isections = controller.mapper.mapFromFilteredImg(temp_img)
shared.graph = graph

print(type(controller.shared.graph))
tps = 0
ticks_per_sec = 0
def sync_tps():
    # this is probably a bad idea
    threading.Timer(5.0, sync_tps).start()
    global ticks_per_sec
    global tps
    tps = ticks_per_sec/5
    controller.shared.tps = tps
    ticks_per_sec = 0
    print("ticks per sec:", tps)

tps_sync = threading.Timer(5.0, sync_tps)
tps_sync.start()

while True:
    # print("ticking")
    controller.camera.updateCamera()
    controller.comms.tick(shared.graph, shared.cars)
    ticks_per_sec += 1
