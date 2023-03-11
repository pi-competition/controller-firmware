# ping flask endpoint

from flask import Flask, Response, jsonify
from flask_restful import Resource, Api
import os
from server.utils.response import success

DEVICE_INFO = {
    "name": "CONTROL-1",
    "type": "CONTROL",
    "hostname": os.environ["HOSTNAME"] if "HOSTNAME" in os.environ else "unknown",
    "status": "online",
}

class Ping(Resource):
    def get(self):
        print("ok")
        return success(DEVICE_INFO)

