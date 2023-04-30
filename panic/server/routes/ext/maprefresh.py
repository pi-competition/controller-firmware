from flask import Flask, Response, jsonify
from flask_restful import Resource, Api
import os
from server.utils.response import success, error
# from controller.shared import graph
import controller

class MapRefresh(Resource):
    def get(self):
        try:
            temp_img = controller.camera.getImage()
            graph, nodes, zones, isections = controller.mapper.mapFromFilteredImg(temp_img)
        except e:

            return error("it no worky" + str(e), 503)
        controller.shared.graph = graph



        return success({"zones": zones, "nodes": nodes})
