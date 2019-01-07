

import LoadData
from BAFFLE import BAFFLE
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import json
import seaborn as sns
from scipy.stats.stats import pearsonr
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc


def verif(dtest_y, predictions):
    print("\naccuracy_score :", accuracy_score(dtest_y, predictions))
    a = classification_report(dtest_y, predictions)
    print("\nclassification report :\n", (a))
    return a

#    plt.figure(figsize=(13, 10))
#    plt.subplot(221)
#    sns.heatmap(confusion_matrix(dtest_y, predictions), annot=True, fmt="d", linecolor="k", linewidths=3)
#    plt.title("CONFUSION MATRIX", fontsize=20)

experiments = LoadData.LoadData()
data = experiments.build_data_frame()
k = 0.9
lag_size = 3000
window_size = 3000
baffle_alg = BAFFLE(lag_size, window_size, k=k, alpha=0.3, explained_variance_ratio=0.95)

with open('time_intervals.json', 'r') as fp:
    time_intervals_dict = json.load(fp)
features = ['h1', 'h2', 'h3']
column_names = list(set([column[0] for column in data]))
column_names.sort()
column_names.insert(0, column_names.pop(column_names.index('014')))
column_names.insert(0, column_names.pop(column_names.index('013')))
column_names.insert(0, column_names.pop(column_names.index('012')))
column_names.insert(0, column_names.pop(column_names.index('011')))
column_names.insert(0, column_names.pop(column_names.index('009')))
column_names.insert(0, column_names.pop(column_names.index('008')))
column_names.insert(0, column_names.pop(column_names.index('006')))
column_names.insert(0, column_names.pop(column_names.index('004')))

# stationary_data.index = (stationary_data.index - min(stationary_data.index))
i = 0
data_list = []
for name in time_intervals_dict:
    for elem in time_intervals_dict[name]:
        i+=1
        three_min = data[name][elem[0]:elem[1]]
        three_min = three_min.fillna(0)
        three_min.index = (three_min.index - min(three_min.index))
        a = three_min.error
        k_er = 6000
        if name in ['004', '006', '008', '009', '011', '012', '013', '014']:
            if elem[0] == 200:
                k_er = 4000
            elif elem[0] == 700:
                k_er = 7000
        elif name == '007':
            if elem[0] == 1060:
                k_er = 1500
            else:
                k_er = 4000
        elif name == '019':
            if elem[0] == 1000:
                k_er = 8000
        elif name == '020':
            if elem[0] == 450:
                k_er = 7000
            elif elem[0] == 1090:
                k_er = 4000
        elif name == '027':
            if elem[0] == 1150:
                k_er = 3000
            elif elem[0] == 1450:
                k_er = 2500
        elif name == '028':
            if elem[0] == 200:
                k_er = 5000
            elif elem[0] == 1500:
                k_er = 2000
        elif name == '029':
            if elem[0] == 200:
                k_er = 3000
            elif elem[0] == 1500:
                k_er = 3500
            elif elem[0] == 1600:
                k_er = 2000
        elif name == '030':
            if elem[0] == 1600:
                k_er = 2000
        three_min.error = ((a==1) & (a.shift(100)==1) | (a.shift(k_er)) | (a.shift(k_er//2)))
        data_list.append(three_min)

pd.concat(data_list).to_csv('data')