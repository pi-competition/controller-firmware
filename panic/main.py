import flask 
from flask import request, jsonify
from flask_restful import Resource
from server.utils.response import success, error
from api import PANICAPI
import threading
import urllib
import json

app = flask.Flask(__name__)
app.config["DEBUG"] = True
api = PANICAPI(app)
app.debug = False

DEVICES = {}
CONFIG = {}
app.config["DEVICES"] = DEVICES
with open("config.json") as f:
    CONFIG = json.loads(f.read())
    app.config["CONFIG"] = CONFIG

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
                # j = json.loads(res.decode("utf-8"))
                device["info"] = json.loads(res.decode("utf-8"))
            except:
                device["status"] = "offline"
                device["info"] = {"name": device["name"], "type": device["type"], "status": "offline"}
        else:
            device["status"] = "offline"
            device["info"] = {"name": device["name"], "type": device["type"], "status": "offline"}
    print(DEVICES)
    app.config["DEVICES"] = DEVICES





get_devices()
threading.Timer(60, get_devices).start()
def run():
    app.run(port=5001, host="0.0.0.0", threaded=False)

def dev():
    app.run(port=5001, host="0.0.0.0", debug=True)

#dev()
comms_thread = threading.Thread(target=run)
comms_thread.start()

print("does this work")



