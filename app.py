from flask import Flask, request, jsonify
import pickle
import numpy as np

model = pickle.load(open('models/simple.pkl', 'rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return "Hello all..!!"

@app.route('/predict',methods=['POST'])
def predict():
    hours = request.form.get('hours')

    input_query = np.array([[hours]])
    result = model.predict(input_query)[0]
    return jsonify({"Scores": str(result)})

if __name__ == '__main__':
    app.run(debug=True)
