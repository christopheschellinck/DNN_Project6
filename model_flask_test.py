# from flask import Flask, request, jsonify
# import os
#
# app = Flask("__name__")
# port = int(os.environ.get("PORT", 5000))
#
# # @app.route('/sample')
# # def running():
# #     return ('Flask is running')
#
# # http://10.0.0.4:5000/static/hello.html
# from flask import request, jsonify, Flask
#
# app = Flask(__name__)
#
# @app.route('/hello',methods=['POST'])
# def hello():
#     message = request.get_json(force=True)
#     name = message['name']
#     response = {
#         'greeting': 'Hello, ' + name + '!'
#     }
#     return jsonify(response)
#
# if __name__=="__main__":
#     app.run(debug=True)
#
#
