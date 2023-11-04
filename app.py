from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlProject.pipeline.prediction import PredictionPipeline
from mlProject.config.configuration import ConfigurationManager
from mlProject.components.data_transformation import DataTransformation
app = Flask(__name__) # initializing a flask app
import random

@app.route('/',methods=['GET'])  # route to display the home page
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict',methods=['POST','GET']) # route to show the predictions in a web UI
def index():
    if request.method == 'POST':
        try:
            #  reading the inputs given by the user
            gender =request.form['gender']
            SeniorCitizen =int(request.form['SeniorCitizen'])
            Partner =request.form['Partner']
            Dependents =request.form['Dependents']
            Tenure =float(request.form['Tenure'])
            PhoneService =request.form['PhoneService']
            MultipleLines =request.form['MultipleLines']
            PaperlessBilling =request.form['PaperlessBilling']
            MonthlyCharges =float(request.form['MonthlyCharges'])
            TotalCharges =float(request.form['TotalCharges'])

            dict_data = {'customerID': ['7590-VHVEG'],
             'gender': [gender],
             'SeniorCitizen': [SeniorCitizen],
             'Partner': [Partner],
             'Dependents': [Dependents],
             'tenure': [Tenure],
             'PhoneService': [PhoneService],
             'MultipleLines': [MultipleLines],
             'InternetService': ['DSL'],
             'OnlineSecurity': ['No'],
             'OnlineBackup': ['Yes'],
             'DeviceProtection': ['No'],
             'TechSupport': ['No'],
             'StreamingTV': ['No'],
             'StreamingMovies': ['No'],
             'Contract': ['Month-to-month'],
             'PaperlessBilling': [PaperlessBilling],
             'PaymentMethod': ['Electronic check'],
             'MonthlyCharges': [MonthlyCharges],
             'TotalCharges': [TotalCharges]
                         }

            data = pd.DataFrame(dict_data)

            config = ConfigurationManager()
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            processed_data = data_transformation.preprocessing(data)

            obj = PredictionPipeline()
            predict = obj.predict(processed_data)
            predict = "Yes" if predict==1 else "No"


            return render_template('results.html', prediction = str(predict))

        except Exception as e:
            print('The Exception message is: ',e)
            return 'something is wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	# app.run(host="0.0.0.0", port = 8080, debug=True)
	app.run(host="0.0.0.0", port = 8080)