# importing the required libraries
from typing import Any
import json
import pandas as pd

import numpy as np
from sklearn.model_selection import train_test_split
# Importing Decision Tree classifierDecisionTreeRepressor
from sklearn.tree import DecisionTreeRegressor

clf = DecisionTreeRegressor()  # Create Decision Tree classifier object


def train_model():
    # Reading the csv filepip got from UCI
    data = pd.read_csv('cpdata.csv')

    print(data.head(1))  # considers first line of dataset as the header
    # sklearn is a machine learning package which include a lot of ML algorithms.
    # pandas's read_csv method read the CSV data file Used to read and write different files.
    # Creating dummy variable for target i.e label
    label = pd.get_dummies(data.label).iloc[:, 1:]
    data = pd.concat([data, label], axis=1)
    data.drop('label', axis=1, inplace=True)
    print('The data present in one row of the dataset is')
    print(data.head(1))
    train = data.iloc[:, 0:4].values
    test = data.iloc[:, 4:].values

    # Dividing the data into training and test set using sklearn according to ratio provided
    # X has attributes and Y has target variables of the dataset
    X_train, X_test, y_train, y_test = train_test_split(train, test,
                                                        test_size=0.3)  # split ratio of 80:20 .The 20% testing data is
    # represented by the 0.2 at the end

    # normalise the values to improve the model accuracy use standardised() function from  sklearn.
    from sklearn.preprocessing import StandardScaler  # need to process the raw data to boost the performance of models.

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)  # Fit to data, then transform it.
    X_test = sc.transform(X_test)

    # Fitting the classifier into training set, Performing The decision tree analysis using scikit learn
    clf.fit(X_train, y_train)  # Train Decision Tree Classifier
    pred = clf.predict(X_test)  # Predict the response for test dataset

    from sklearn.metrics import accuracy_score

    # Finding the accuracy of the model
    a = accuracy_score(y_test, pred)
    print("The accuracy of this model is: ", a * 100)
    return clf


def test_model():
    # Using firebase to import data to be tested
    clf = test_model()
    df = pd.read_csv('out.csv', usecols=['humidity', 'temperature', 'moisture', 'moisture'])
    average = df.mean(axis=0)
    print(average["humidity"], 'humidity')
    print(average["temperature"], 'temperature')
    print(average["moisture"], 'moisture')
    print(average["moisture"], 'moisture')

    # print(t)

    shum = average['humidity']
    stemp = average['temperature']
    smoist = average['moisture']
    smoist = average['moisture']

    l = []
    l.append(shum)
    l.append(stemp)
    l.append(smoist)
    l.append(smoist)
    predictcrop = [l]

    # Putting the names of crop in a single list
    crops = ['wheat', 'mungbean', 'Tea', 'millet', 'maize', 'lentil', 'jute', 'cofee', 'cotton', 'ground nut', 'peas',
             'rubber', 'sugarcane', 'tobacco', 'kidney beans', 'moth beans', 'coconut', 'blackgram', 'adzuki beans',
             'pigeon peas', 'chick peas', 'banana', 'grapes', 'apple', 'mango', 'muskmelon', 'orange', 'papaya',
             'watermelon', 'pomegranate']
    cr = 'rice'

    # Predicting the crop
    predictions = clf.predict(predictcrop)
    count = 0
    for i in range(0, 30):
        if (predictions[0][i] == 1):
            c = crops[i]
            count = count + 1
            break;
        i = i + 1
    if (count == 0):
        print('The predicted crop is %s' % cr)
    else:
        print('The predicted crop is %s' % c)

    # Sending the predicted crop to database

    with open("crop.csv", "w", encoding='utf-8') as f:
        f.write("predictedcrop \n")

    with open("crop.csv", "a", encoding='utf-8') as f:
        f.write("%s\n" % (c))


def predict(l):
    clf = train_model()
    # test_model()
    crop = ''

    predictcrop = [l]

    # Putting the names of crop in a single list
    crops = ['wheat', 'mungbean', 'Tea', 'millet', 'maize', 'lentil', 'jute', 'cofee', 'cotton', 'ground nut', 'peas',
             'rubber', 'sugarcane', 'tobacco', 'kidney beans', 'moth beans', 'coconut', 'blackgram', 'adzuki beans',
             'pigeon peas', 'chick peas', 'banana', 'grapes', 'apple', 'mango', 'muskmelon', 'orange', 'papaya',
             'watermelon', 'pomegranate']
    cr = 'rice'

    # Predicting the crop
    predictions = clf.predict(predictcrop)
    count = 0
    for i in range(0, 30):
        if (predictions[0][i] == 1):
            c = crops[i]
            count = count + 1
            break;
        i = i + 1
    if (count == 0):
        print('The predicted crop is %s' % cr)
        # Sending the predicted crop to database
        crop = cr

        with open("crop.csv", "w", encoding='utf-8') as f:
            f.write("predictedcrop \n")

        with open("crop.csv", "a", encoding='utf-8') as f:
            f.write("%s\n" % (c))
    else:
        print('The predicted crop is %s' % c)
        crop = c

        # Sending the predicted crop to database

        with open("crop.csv", "w", encoding='utf-8') as f:
            f.write("predictedcrop \n")

        with open("crop.csv", "a", encoding='utf-8') as f:
            f.write("%s\n" % (c))
    return crop
