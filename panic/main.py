import flask 
from flask import request, jsonify
from flask_restful import Resource, Api
import threading
import urllib
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True
api = Api(app)

DEVICES = {}

from server.routes.api.ping import Ping
api.add_resource(Ping, '/api/ping')
from server.routes.api.refresh_camera import RefreshCamera
api.add_resource(RefreshCamera, '/api/refresh_camera')


def get_devices():
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
                res = urllib.request.urlopen(f"http://{device['ip']}:5000/api/ping").read()
                j = json.loads(res.decode("utf-8"))
                device["info"] = json.loads(res.decode("utf-8"))
            except:
                device["status"] = "offline"
                device["info"] = None
        else:
            device["status"] = "offline"
            device["info"] = None
    print(DEVICES)






get_devices()
threading.Timer(60, get_devices).start()

app.run(port=5001, host="0.0.0.0")

print("does this work")

# create a new thread to GET http://127.0.0.1:5000/devices every 1 minute


