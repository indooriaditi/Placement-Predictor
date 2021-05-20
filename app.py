from flask import Flask, render_template, request
import numpy as np
import pickle
from dash_application import create_dash_application

app = Flask('DATA_VISUAL')

create_dash_application(app)

# prediction function
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(open("naivebayesmodel.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    return result[0]

def ProbPredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 4)
    loaded_model = pickle.load(open("naivebayesmodel.pkl", "rb"))
    probability = loaded_model.predict_proba(to_predict)[:,-1]
    return probability

@app.route('/result', methods = ['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        result = ValuePredictor(to_predict_list)  
        if int(result)== 1:
            prediction ='Kudos! Your chance of being placed is'
        else:
            prediction ='Need to work harder! Your chance of being placed is'
        probability = np.round(ProbPredictor(to_predict_list),4)
        return render_template("predictorform.html", prediction = prediction ,probability=probability*100)




        

@app.route('/')
def show_predict_stock_form():
    return render_template('predictorform.html')


if __name__ == "__main__":
    app.run(debug=True)        