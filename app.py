from flask import Flask, render_template, request
import numpy as np
import requests
import config
import pickle




#"""creating object of flask"""
app = Flask(__name__)

crop_recommendation_model_path = 'NBClassifier.pkl'
crop_recommendation_model = pickle.load(open(crop_recommendation_model_path, 'rb'))


def weather_fetch(city_name):
    """
    Fetch and returns the temperature and humidity of a city
    :params: city_name
    :return: temperature, humidity
    """
    api_key = config.weather_api_key
    base_url = "http://api.openweathermap.org/data/2.5/weather?"

    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()

    if x["cod"] != "404":
        y = x["main"]

        temperature = round((y["temp"] - 273.15), 2)
        humidity = y["humidity"]
        return temperature, humidity
    else:
        return None



@app.route("/")
def hello():
    return render_template("index.html")

@app.route("/submit", methods = ['POST'])
def submit():
    # HTML -> .py
    if request.method == 'POST':
        N = int(request.form['nitrogen'])
        P = int(request.form['phosphorous'])
        K = int(request.form['pottasium'])
        ph = float(request.form['ph'])
        rainfall = float(request.form['rainfall'])

        #state = request.form.get("stt")
        city = request.form.get("city")

        if weather_fetch(city) != None:
            temperature, humidity = weather_fetch(city)
            data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            my_prediction = crop_recommendation_model.predict(data)
            final_prediction = my_prediction[0]

            return render_template('submit.html', prediction=final_prediction)

        else:

            return render_template('index.html')


    #return render_template('submit.html', prediction=final_prediction)



    # render whatever information we got from .py -> HTML

    #return render_template("", n = name)


if __name__ == "__main__":
    app.run(debug=True)