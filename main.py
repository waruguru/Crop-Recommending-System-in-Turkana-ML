# app.py
from flask import Flask, render_template, request
from Crop_pred import predict, test_model

app = Flask(__name__)


# defining a route
@app.route("/", methods=['GET', 'POST', 'PUT'])  # decorator
def home():  # route handler function
    # returning a response
    # since we sent the data using POST, we'll use request.form
    # we can also request.values
    # print('Moisture: ', request.form['moisture'])
    # we can also request.values
    # print('Humidity: ', request.form['humidity'])
    return render_template("./index.html")


# handling form data
@app.route('/predicted', methods=['POST'])
def handle_data():
    # since we sent the data using POST, we'll use request.form
    print('Temperature: ', request.form['temperature'])
    print('Moisture: ', request.form['moisture'])
    # we can also request.values
    print('Humidity: ', request.form['humidity'])
    l = []
    l.append(request.form['temperature'])
    l.append(request.form['moisture'])
    l.append(request.form['humidity'])
    l.append(request.form['humidity'])

    crop_predicted = predict(l)

    # we can also request.values
    return render_template('./crops.html', crop_predicted=crop_predicted)


app.run(debug=True)
