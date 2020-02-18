# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 15:12:51 2020

@author: 138709
"""

# Artificial Neural Network

# Data Preprocessing

# Import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
Y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoderX1 = LabelEncoder()
labelEncoderX2 = LabelEncoder()
X[:,1] = labelEncoderX1.fit_transform(X[:,1])
X[:,2] = labelEncoderX2.fit_transform(X[:,2])


oneHotEncoderX1 = OneHotEncoder(categorical_features = [1])
X = oneHotEncoderX1.fit_transform(X).toarray()
X = X[:, 1:]


# Splitting the dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Making the ANN


# Import Keras
import keras
from keras.models import Sequential
from keras.layers import Dense


# Initializing ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
model.add(Dense(output_dim = 6, init = 'uniform', activation = 'relu'))

# Adding the output layer
model.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling ANN
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the training set
model.fit(X_train, Y_train, batch_size = 10, epochs = 100)

# Predicting the test set results
Y_pred = model.predict(X_test)
Y_pred = (Y_pred > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
