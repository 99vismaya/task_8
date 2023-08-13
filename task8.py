# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 11:41:25 2023

@author: Dell
"""

import pandas as pd
import numpy as np

# Load the dataset (replace 'weather_dataset.csv' with your file)
df = pd.read_csv('C:/Users/Dell/Desktop/515849391ad37fe593997fe0db98afaa-f663366d17b7d05de61a145bbce7b2b961b3b07f/weather.csv')

# Display the first few rows of the dataset
print(df.head())

from sklearn.preprocessing import LabelEncoder
# creating instance of labelencoder
labelencoder = LabelEncoder()

df['temperature']= labelencoder.fit_transform(df['temperature'])
df['humidity'] = labelencoder.fit_transform(df['humidity'])
df['windy'] = labelencoder.fit_transform(df['windy'])

X = df[['temperature', 'humidity', 'windy']]  # Features
y = df['weather_condition'] 

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC

model_linear = SVC(kernel = "poly")
model_linear.fit(X_train, y_train)

pred_test_linear = model_linear.predict(X_test)
spt = np.mean(pred_test_linear == y_test)

pred_train_linear = model_linear.predict(X_train)
spn = np.mean(pred_train_linear == y_train)

import pickle

with open("task8", 'wb') as task8 :
    pickle.dump(model_linear, task8)

task8.close()
