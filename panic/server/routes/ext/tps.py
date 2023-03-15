from flask import Flask, Response, jsonify
from flask_restful import Resource, Api
import os
import controller
from server.utils.response import success

class Tps(Resource):
    def get(self):
        return success(controller.shared.tps)
