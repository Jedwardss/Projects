# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 16:39:01 2020

@author: James Edwards
"""
import pickle
from flask import Flask
from google.cloud import storage
import io
import pandas as pd

test_data_file = 'test.csv'
path = 'svclassifier.pkl'
bucket_name = 'scg-intern-01'
model = None
key = 'key.json'


def prep_data(test_data):
    # Missing Values

    # Filling the missing values in Age with the medians of Sex and Pclass groups
    test_data['Age'] = test_data.groupby(['Sex', 'Pclass'])['Age'].apply(lambda x: x.fillna(x.median()))

    # Filling embarked
    test_data['Embarked'] = test_data['Embarked'].fillna('S')

    # Filling fare
    med_fare = test_data.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
    test_data['Fare'] = test_data['Fare'].fillna(med_fare)

    # Filling cabin

    nanReplaced = test_data.Cabin.fillna("X")
    test_data["cabinProcessed"] = nanReplaced.str.get(0)

    # Title replacement

    test_data['Title'] = test_data.Name.str.extract('([A-Za-z]+)\.')
    test_data['Title'].replace(['Mlle', 'Mme', 'Ms', 'Dr', 'Major', 'Lady', 'Countess', 'Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)

    features = ["Pclass", "Sex", "SibSp", "Parch", "cabinProcessed", "Title", "Age", "Embarked", "Fare"]
    data_final = pd.get_dummies(test_data[features])
    return data_final

def download_pickle(bucket, path, key):
    client = storage.Client.from_service_account_json(key)
    bucket = client.get_bucket(bucket)
    blob = storage.Blob(path, bucket)
    byte_stream = io.BytesIO()
    blob.download_to_file(byte_stream)
    byte_stream.seek(0)
    return byte_stream
 
def download_data(bucket, file, key) :
    client = storage.Client.from_service_account_json(key)
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file)
    data = blob.download_as_string()
    return data
       
def upload_predictions(df, key):
    outputfile = io.StringIO()
    df.to_csv(outputfile, encoding='utf-8', index=False)
    outputfile.seek(0)
    client = storage.Client.from_service_account_json("key.json")
    bucket = client.get_bucket(bucket_name)
    blob = storage.Blob("predicitons.csv", bucket)
    blob.upload_from_file(outputfile)
   
app = Flask(__name__)

@app.route('/')
def home_endpoint():
    return 'Hi welcome to my first web app! Please type /predict after the URL above to receive survival predictions based on data from the titanic.'

@app.route('/predict', methods = ['GET', 'POST'])
def get_prediction():
    
    # Get test data
    
    data = download_data(bucket_name, test_data_file, key)
    test_data = pd.read_csv(io.BytesIO(data), encoding = 'utf-8', sep = ",")
    
    # Prep data
    
    data_final = prep_data(test_data)
    
    # Get model
    
    model = pickle.loads(download_pickle(bucket_name, path, key).read())
    
    # Predict
    
    predictions = model.predict(data_final)
    
    # Upload
    
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    upload_predictions(output, key)
    return 'Predictions uploaded to GCS'
    
    
if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=5000)