import json
from flask import Response
from http import HTTPStatus
def error(message, status_code=400):
    j = json.dumps(
        {
            "success": False,
            "code": status_code,
            # get the status code message from the status code
            "message": str(HTTPStatus(status_code).phrase),
            "error": message,
        }, allow_nan=False
    )
    
    return Response(j, status=status_code, mimetype="application/json")


def success(data, status_code=200):
    j = json.dumps(
        {
            "success": True,
            "code": status_code,
            # get the status code message from the status code
            "message": str(HTTPStatus(status_code).phrase),
            "data": data,
        }, allow_nan=False
    )
    
    return Response(j, status=status_code, mimetype="application/json")
