# -*- coding: utf-8 -*-
import LoadData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
from scipy.stats.stats import pearsonr   
experiments = LoadData.LoadData()
data = experiments.build_data_frame()
window_size = 6000
with open('times.json', 'r') as fp:
    time_intervals_dict = json.load(fp)

column_names = list(set([column[0] for column in data]))
column_names.sort()
#stationary_data.index = (stationary_data.index - min(stationary_data.index))
num_of_features=6
data_list = []
window_size = 18000
for name in time_intervals_dict: 
    for elem in time_intervals_dict[name]:
        three_min = data[name][elem[0]:elem[1]]
        three_min = three_min.fillna(0)
        num_of_error = sum(three_min.error==1)
        if num_of_error > 0.1 * window_size:
            error=1
        else:
            error=0
        
        data_list.append([np.std(three_min.h1), np.std(three_min.h2), np.std(three_min.h3),
                          np.mean(three_min.h1), np.mean(three_min.h2), np.mean(three_min.h3),
                          error])
    
from_list = pd.DataFrame(np.array(data_list), columns=['std_h1', 'std_h2', 'std_h3', 'mean_h1', 'mean_h2', 'mean_h3', 'error'])
sns.pairplot(from_list,hue="error")
plt.title("pair plot for variables")
plt.show()

from sklearn.model_selection import train_test_split

train, test = train_test_split(from_list, test_size=.3, random_state=123)

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

#    predicting_probabilites = algorithm.predict_proba(dtest_x)[:, 1]
#    fpr, tpr, thresholds = roc_curve(dtest_y, predicting_probabilites)
#    plt.subplot(222)
#    plt.plot(fpr, tpr, label=("Area_under the curve :", auc(fpr, tpr)), color="r")
#    plt.plot([1, 0], [1, 0], linestyle="dashed", color="k")
#    plt.legend(loc="best")
#    plt.title("ROC - CURVE & AREA UNDER CURVE", fontsize=20)

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
# from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# rf =DecisionTreeClassifier()
# rf=GaussianProcessClassifier(1.0 * RBF(1.0))

#rf = KNeighborsClassifier(n_jobs=4, n_neighbors=7)
rf = SVC()
al = model(rf, train_X, train_Y, test_X, test_Y, 'none')

all_data_num = experiments.build_data_frame()


def check_error(data):
    exp15 = data
    window=3000
    e1 = exp15['h1'].rolling(window=window)
    e2 = exp15['h2'].rolling(window=window)
    e3 = exp15['h3'].rolling(window=window)

    cor12 = e1.std()
    cor13 = e2.std()
    cor23 = e3.std()

#    cor12 = e1.corr(e2)#[window/100-0.01:]
#    cor13 = e1.corr(e3)#[window/100-0.01:]
#    cor23 = e2.corr(e3)#[window/100-0.01:]
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