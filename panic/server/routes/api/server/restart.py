# ping flask endpoint

from flask import Flask, Response, jsonify, request, current_app as app
from flask_restful import Resource, Api
import os
import threading
import time
from server.utils.response import success, error
from server.utils.validate import validate

class Restart(Resource):
    def post(self):
        # get auth header
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return error("You must provide authentication credentials to access this resource.", 401)
        if auth_header != app.config["CONFIG"]["API_PASSWORD"]:
            return error("You do not have permission to access this resource.", 403)
        # send a 204 response then restart the raspberry pi (delay in a thread)
        #return error("This endpoint has been disabled", 503)
        def restart():
            time.sleep(1)
            print("goodbye")
            os.system("sudo reboot")
        threading.Thread(target=restart).start()
        return success({}, 204)
        



        
