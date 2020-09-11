# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 21:07:50 2020

@author: James Edwards
"""

import pickle
from flask import Flask
from google.cloud import storage
import io
import pandas as pd
from sklearn.svm import SVC

path = 'svclassifier.pkl'
train_file_name = 'train.csv'
bucket_name = 'scg-intern-01'
app = Flask(__name__)

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
    test_data['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
    return test_data

def download_data(bucket, file) :
    client = storage.Client.from_service_account_json("key.json")
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(file)
    data = blob.download_as_string()
    return data

def upload_pickle(model, bucket, path):
    pklfile = io.BytesIO()
    pickle.dump(model, pklfile)
    pklfile.seek(0)
    client = storage.Client.from_service_account_json("key.json")
    bucket = client.get_bucket(bucket_name)
    blob = storage.Blob("svclassifier.pkl", bucket)
    blob.upload_from_file(pklfile)
         
@app.route('/')
def home_endpoint():
    return 'Please type /train after the URL above to train and upload the latest version of the SVC classifier.'

@app.route('/train', methods = ['GET', 'POST'])
def train_model():
    
    # Get data
    
    data = download_data(bucket_name, train_file_name)
    train_data = pd.read_csv(io.BytesIO(data), encoding = 'utf-8', sep = ",")
    
    # Clean data
    data_final = prep_data(train_data)
    y = data_final["Survived"]
    features = ["Pclass", "Sex", "SibSp", "Parch", "cabinProcessed", "Title", "Age", "Embarked", "Fare"]
    X = pd.get_dummies(train_data[features])
    
    # Create and fit model
    svclassifier = SVC(kernel='linear')
    svclassifier.fit(X, y)
    
    # Upload model as pickle
    upload_pickle(svclassifier, bucket_name, path)
    return 'New trained model uploaded to GCS'

if __name__ == '__main__':
    app.run(debug = True, host='0.0.0.0', port=5000)