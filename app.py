from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('lungcancer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('lung.html')


@app.route('/predict', methods = ['GET', 'POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = int(prediction[0])

    if output == 0:
        return render_template('lung.html', prediction_text= 'Yaaaaay!! You are safe from Lung Cancer')
    else:
        return render_template('lung.html', prediction_text= 'Be careful! There is a chance of Lung cancer') 


if __name__ == "__main__":
    app.run(debug=True)
