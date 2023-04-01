from flask import Flask, Response, jsonify
from flask_restful import Resource, Api
import os
from server.utils.response import success
# from controller.shared import graph
import controller

class MapRefresh(Resource):
    def get(self):
        temp_img = controller.camera.getImage()
        graph, nodes, zones, isections = controller.mapper.mapFromFilteredImg(temp_img)
        controller.shared.graph = graph



        return success({"zones": zones, "nodes": nodes})
