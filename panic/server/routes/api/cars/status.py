# ping flask endpoint

from flask import Flask, Response, jsonify, current_app as app
from flask_restful import Resource, Api
import os
from server.utils.response import success, error


class Status(Resource):
    def get(self):
        info = []
        if not app.config["DEVICES"]:
            return error("This request cannot be handled as this time as RADAR is unreachable", 503)
        for device in app.config["DEVICES"]:
            if device["info"]["type"] == "CAR":
                info.append({
                    "id": int(device["info"]["name"].split("-")[1]) - 1,
                    **device["info"]
                })


        return success(info)

