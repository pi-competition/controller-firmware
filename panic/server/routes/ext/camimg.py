import flask
from flask import Flask, Response, jsonify
from flask_restful import Resource, Api
import os
import cv2
from server.utils.response import success
import controller.shared


class CamImg(Resource):
    def get(self):
        _, im_bytes_np = cv2.imencode('.png', controller.shared.camimg)
    
        # Constuct raw bytes string 
        bytes_str = im_bytes_np.tobytes()

        # Create response given the bytes
        response = flask.make_response(bytes_str)
        response.headers.set('Content-Type', 'image/png')
    
        return response
