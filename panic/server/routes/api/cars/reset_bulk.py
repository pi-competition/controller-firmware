# ping flask endpoint

from flask import Flask, Response, jsonify, request, current_app as app
from flask_restful import Resource, Api
import urllib
from server.utils.response import success, error
from server.utils.validate import validate

schema = [
    {
        "name": "ids",
        "type": "list",
        "subtype": "int",
        "required": True,
        "min": 1,
        "max": 4
    }
]

class ResetBulk(Resource):
    def post(self):
        # get auth header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return error("You must provide authentication credentials to access this resource.", 401)
        if auth_header != app.config["CONFIG"]["API_PASSWORD"]:
            return error("You do not have permission to access this resource.", 403)

        try:
            data = request.get_json()
        except:
            return error("Invalid JSON", 400)
        valid = validate(schema, data)
        print(valid)
        if valid[0] == False:
            return error(valid[1], 400)
        print(app.config["DEVICES"])
        reset = []
        for id in data["ids"]:
            item = next((item for item in app.config["DEVICES"] if item["info"] != None and item["type"] == "CAR" and item["info"]["id"] == id), None)
            if not item:
                return error(f"Invalid car ID {id}", 400)
            if item["status"] != "online":
                return error(f"The requested action cannot be performed on car {id} at this time", 400)
            reset.append({
                "id": id,
                "url": f"http://{item['ip']}:5001/api/reset"
            })

        for item in reset:
            try:
                urllib.request.urlopen(item["url"], data=bytes(jsonify({"password": app.config["CONFIG"]["API_PASSWORD"]}), "utf-8"))
            except:
                return error(f"An error occurred while resetting car {item['id']}", 500)

            

        

        # send a 204 response then restart the raspberry pi (delay in a thread)
        return success({}, 204)
        




