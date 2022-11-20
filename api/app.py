from flask import Flask
from flask import request
from flask import jsonify
from joblib import load

app = Flask(__name__)
model_path = "models/svm_random_state=84.joblib"
model = load(model_path)

@app.route("/predict", methods = ["POST"])
def predict_class():
    image1 = request.json['image1']
    model_path = request.json['model']
    model = load(model_path)
    predicted1 = model.predict([image1])
    print(predicted1)
    return int(predicted1)

if __name__ == "__main__":
    app.run(host="0.0.0.0",port=5000)