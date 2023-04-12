from flask import Flask,render_template,request,redirect
from flask_cors import CORS,cross_origin
import pickle
import pandas as pd
import numpy as np


app=Flask(__name__)
cors=CORS(app)
model=pickle.load(open('LinearRegressionModel.pkl','rb'))
car=pd.read_csv('Cleaned_Car_data.csv')
@app.route('/',methods=['GET','POST'])

def index():
    companies=sorted(car['company'].unique())
    car_models=sorted(car['name'].unique())
    year=sorted(car['year'].unique(),reverse=True)
    fuel_type=car['fuel_type'].unique()
    transmission=car['transmission'].unique()
    companies.insert(0,'Select Company')
    return render_template('index.html',companies=companies, car_models=car_models, years=year, fuel_types=fuel_type, transmission=transmission)
@app.route('/predict',methods=['POST'])
@cross_origin()

def predict():

    company=request.form.get('company')

    car_model=request.form.get('car_models')
    year=request.form.get('year')
    fuel_type=request.form.get('fuel_type')
    driven=request.form.get('kilo_driven')
    transmission=request.form.get('transmission')

    prediction=model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type', 'transmission'],
                                          data=np.array([car_model, company, year, driven, fuel_type, transmission]).reshape(1, 6)))
    print(prediction)

    return str(np.round(prediction[0],2))

df = pd.read_csv('C:/Users/Amay/Desktop/Prediction Model For Car Price/templates/Car_data.csv')
df.to_csv('C:/Users/Amay/Desktop/Prediction Model For Car Price/templates/Car_data.csv', index=None)

@app.route('/carrec')
def carrec():
    # converting csv to html
    data = pd.read_csv('C:/Users/Amay/Desktop/Prediction Model For Car Price/templates/Car_data.csv')
    data.to_csv('C:/Users/Amay/Desktop/Prediction Model For Car Price/templates/Car_data.csv', index=None)
    return render_template('carrec.html', tables=[data.to_html()], titles=[''])


if __name__=='__main__':
    app.run()
