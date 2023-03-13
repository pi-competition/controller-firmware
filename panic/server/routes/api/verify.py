# ping flask endpoint

from flask import Flask, Response, jsonify, request, current_app as app
from flask_restful import Resource, Api
import os
from server.utils.response import success, error
from server.utils.validate import validate

schema = [
    {
        "name": "password",
        "type": "str",
        "required": True,
        "min": 1,
        "max": 100
    }
]

class Verify(Resource):
    def post(self):
        try:
            data = request.get_json()
        except:
            return error("Invalid JSON", 400)
        valid = validate(schema, data)
        print(valid)
        if valid[0] == False:
            return error(valid[1], 400)
        if data["password"] != app.config["CONFIG"]["API_PASSWORD"]:
            return error("Password is incorrect", 401)
        return success({}, 204)
        
