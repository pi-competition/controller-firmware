from flask import Flask, Response, jsonify
from flask_restful import Resource, Api
import os
from server.utils.response import success
# from controller.shared import graph
import controller

class CarSummary(Resource):
    def get(self):
        # hoo
        cars = {k: v.summarise() for k, v in controller.shared.cars.items()}

        return success(cars)
