import flask
from flask import request
from flask import jsonify, make_response
import pickle
import numpy as np
from flask_cors import CORS, cross_origin

app = flask.Flask(__name__)
app.config["DEBUG"] = True
CORS(app)

@app.route('/getCurrentAndVoltageValues', methods=['GET'])
def compute():
    with open('DecisionTree.pkl', 'rb') as f:
        reg = pickle.load(f)
        prediction = reg.predict(
            np.array([[
                float(request.args.get("Param1")),
                float(request.args.get("Param2")),
                float(request.args.get("Power")),
                float(request.args.get("Gamma Point")),
                float(request.args.get("Harmonic Number")),
                float(request.args.get("A1 Real")),
                float(request.args.get("A1 Imaginary")),
                float(request.args.get("B1 Real")),
                float(request.args.get("B1 Imaginary")),
                float(request.args.get("A2 Real")),
                float(request.args.get("A2 Imaginary")),
                float(request.args.get("B2 Real")),
                float(request.args.get("B2 Imaginary"))
            ]]))

    resp = make_response(jsonify({"I1": prediction[0][0], "I2": prediction[0][1]}))
    resp.headers['Access-Control-Allow-Origin'] = '*'
    return resp

app.run(host="0.0.0.0")
