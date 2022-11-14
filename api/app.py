from flask import Flask
from flask import request
from flask import jsonify
from joblib import load

app = Flask(__name__)
model_path = "svm_gamma=0.001_C=0.5.joblib"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/sum", methods = ['POST'])
def sum():
    print(request.json)
    x = request.json['x']
    y = request.json['y']
    z = x + y
    return jsonify({'sum' : z})

@app.route("/predict", methods = ["POST"])
def predict_class():
    image1 = request.json['image1']
    image2 = request.json['image2']
    print("Loading successful")
    predicted1 = model.predict([image1])
    predicted2 = model.predict([image2])
    if predicted1 == predicted2:
        return "Same Digit"
    else:
        return "Different Digit"


if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)