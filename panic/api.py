from flask_restful import Api
from server.utils.response import error

class PANICAPI(Api):
    def handle_error(self, e):
        if e.code == 405:
            return error("The requested method is not allowed", 405)
        print(e)
        return error("An unexpected error occurred.", 500)

