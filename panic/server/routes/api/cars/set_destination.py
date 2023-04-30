# ping flask endpoint

from flask import Flask, Response, jsonify, request, current_app as app
from flask_restful import Resource, Api
import os
from server.utils.response import success, error
import controller.shared
import controller.comms

class SetDestination(Resource):
    def post(self):
        try:
            data = request.get_json()
        except:
            return error("Invalid JSON", 400)

        car = list(controller.shared.cars.values())[0]
        node = controller.shared.graph.fromPosToClosestNodeSingular(int(data["x"]), int(data["y"]), 100)
        zone = node.zone
        controller.comms.carSetsDestination(car, zone, controller.shared.graph)

        return success()
