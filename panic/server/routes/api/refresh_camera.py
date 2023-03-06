# ping flask endpoint

from flask import Flask, Response, jsonify, current_app as app
from flask_restful import Resource, Api
import os
from server.utils.response import success


class RefreshCamera(Resource):
    def post(self):
        # import abbas junk and do something
        return success({})
        
