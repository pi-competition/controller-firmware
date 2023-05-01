from flask import Flask, Response, jsonify
from flask_restful import Resource, Api
import os
from server.utils.response import success
# from controller.shared import graph
import controller

class MapData(Resource):
    def get(self):
        # hoo
        nodes = []
        for node in controller.shared.graph.nodes:
            new_node = {
                "x": int(node.x),
                "y": int(node.y),
                "is_intersection": node.isection_ind != -1,
                "conns": [{"x":int(n.x), "y":int(n.y)} for n in list(node.conns)]
            }
            nodes.append(new_node)
        zones = []
        # for zone in controller.shared.graph.zones:
            # new_zone = [{"x":int(n.x), "y":int(n.y)} for n in zone.nodes]
            # zones.append(new_zone)



        return success({"nodes": nodes})

