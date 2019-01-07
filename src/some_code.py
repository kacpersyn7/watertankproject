#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 01:07:05 2018

@author: kacper
"""
import LoadData
import pandas as pd
import numpy as np

experiments = LoadData.LoadData()
all_data = experiments.build_data_frame_without_numbers(['019','018'])
a = all_data.error
all_data.error = ((a==1) & (a.shift(1500)==1) & (a.shift(-1500)))
features = ['h1', 'h2', 'h3', 'error']

e1 = all_data['h1'].rolling(window=1000, center=True)
e2 = all_data['h2'].rolling(window=1000, center=True)
e3 = all_data['h3'].rolling(window=1000, center=True)
# make new features
cor12 = e1.std()#[window/100-0.01:]
cor13 = e2.std()#[window/100-0.01:]
cor23 = e3.std()#[window/100-0.01:]
m1 = e1.mean()
m2 = e2.mean()
m3 = e3.mean()
data_merged = np.array([cor12, cor13, cor23, m1, m2, m3, all_data['error']])
df_merged = pd.DataFrame(data_merged.T, columns=['cor12', 'cor13', 'cor23', 'm1', 'm2', 'm3', 'error'])
df_merged = df_merged.dropna()

import seaborn as sns
import matplotlib.pyplot as plt

randomized = df_merged.sample(frac=0.015)
sns.pairplot(randomized,hue="error")
plt.title("pair plot for variables")
plt.show()

from sklearn.model_selection import train_test_split

train, test = train_test_split(randomized, test_size=.3, random_state=123)

train_X = train[[x for x in train.columns if x not in ["error"]]]
train_Y = train[["error"]]
test_X = test[[x for x in train.columns if x not in ["error"]]]
test_Y = test[["error"]]

# MODEL FUNCTION

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc


def model(algorithm, dtrain_x, dtrain_y, dtest_x, dtest_y, of_type):
    print("*****************************************************************************************")
    print("MODEL - OUTPUT")
    print("*****************************************************************************************")
    algorithm.fit(dtrain_x, dtrain_y)
    predictions = algorithm.predict(dtest_x)

    print(algorithm)
    print("\naccuracy_score :", accuracy_score(dtest_y, predictions))

    print("\nclassification report :\n", (classification_report(dtest_y, predictions)))

    plt.figure(figsize=(13, 10))
    plt.subplot(221)
    sns.heatmap(confusion_matrix(dtest_y, predictions), annot=True, fmt="d", linecolor="k", linewidths=3)
    plt.title("CONFUSION MATRIX", fontsize=20)

    predicting_probabilites = algorithm.predict_proba(dtest_x)[:, 1]
    fpr, tpr, thresholds = roc_curve(dtest_y, predicting_probabilites)
    plt.subplot(222)
    plt.plot(fpr, tpr, label=("Area_under the curve :", auc(fpr, tpr)), color="r")
    plt.plot([1, 0], [1, 0], linestyle="dashed", color="k")
    plt.legend(loc="best")
    plt.title("ROC - CURVE & AREA UNDER CURVE", fontsize=20)

    if of_type == "feat":

        dataframe = pd.DataFrame(algorithm.feature_importances_, dtrain_x.columns).reset_index()
        dataframe = dataframe.rename(columns={"index": "features", 0: "coefficients"})
        dataframe = dataframe.sort_values(by="coefficients", ascending=False)
        plt.subplot(223)
        ax = sns.barplot(x="coefficients", y="features", data=dataframe, palette="husl")
        plt.title("FEATURE IMPORTANCES", fontsize=20)
        for i, j in enumerate(dataframe["coefficients"]):
            ax.text(.011, i, j, weight="bold")

    elif of_type == "coef":

        dataframe = pd.DataFrame(algorithm.coef_.ravel(), dtrain_x.columns).reset_index()
        dataframe = dataframe.rename(columns={"index": "features", 0: "coefficients"})
        dataframe = dataframe.sort_values(by="coefficients", ascending=False)
        plt.subplot(223)
        ax = sns.barplot(x="coefficients", y="features", data=dataframe, palette="husl")
        plt.title("FEATURE IMPORTANCES", fontsize=20)
        for i, j in enumerate(dataframe["coefficients"]):
            ax.text(.011, i, j, weight="bold")

    elif of_type == "none":
        return (algorithm)


from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# rf =DecisionTreeClassifier()
# rf=GaussianProcessClassifier(1.0 * RBF(1.0))

#rf = KNeighborsClassifier(n_jobs=4, n_neighbors=7)
#rf = MLPClassifier(alpha=0.06)
rf = SVC(probability=True)
al = model(rf, train_X, train_Y, test_X, test_Y, 'none')

all_data_num = experiments.build_data_frame()


def check_error(data):
    exp15 = data
    window=1000
    e1 = exp15['h1'].rolling(window=window,center=True)
    e2 = exp15['h2'].rolling(window=window,center=True)
    e3 = exp15['h3'].rolling(window=window,center=True)

    cor12 = e1.std()#[window/100-0.01:]
    cor13 = e2.std()#[window/100-0.01:]
    cor23 = e3.std()#[window/100-0.01:]
    m1 = e1.mean()#[window/100-0.01:]
    m2 = e2.mean()#[window/100-0.01:]
    m3 = e3.mean()#[window/100-0.01:]
    cor12 = cor12.dropna()
    cor13 = cor13.dropna()
    cor23 = cor23.dropna()
    m1 = m1.dropna()
    m2 = m2.dropna()
    m3 = m3.dropna()
    ll = al.predict(np.array([cor12, cor13, cor23, m1, m2, m3]).T)
    plt.plot(ll)

    exp15.plot(subplots=True)
