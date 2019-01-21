import LoadData
from BAFFLE import BAFFLE
from MahClassifer import MahClassifer
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
path = '../results/'
path_to_data = '../data/'


def create_confusion_from_times(times_real, times_predicted, edz=1000, name=''):
    pn=0
    pp=0
    fp=0
    fn=0
    for prediction, real in zip(times_predicted, times_real):
        if prediction == 0.0 and real == 0.0:
            pn += 1
        elif prediction != 0.0 and real == 0.0:
            fp += 1
        elif prediction == 0.0 and real != 0.0:
            fn += 1
        else:
            if abs(prediction - real) < edz:
                pp += 1
            else:
                fp += 1

    my_confusion_matrix = np.array([[pn, fp],
                                 [fn, pp]])
    acc = (pp+pn)/(pn+pp+fp+fn)
    prec = pp/(pp+fp)
    rec = pp/(pp+fn)
    f1 = (2*pp)/(2*pp + fn + fp)
    results = (' '.join(["{0:.2f}".format(acc), "{0:.2f}".format(prec),
                         "{0:.2f}".format(rec),
                         "{0:.2f}".format(f1)]))
    plt.figure()
    conf_mat=sns.heatmap(my_confusion_matrix, annot=True, fmt="d", linecolor="k", linewidths=3)
    conf_mat = conf_mat.get_figure()
    conf_mat.savefig(path + name + '_macierzpomylekedz.png')
    return results


def verify(dtest_y, predictions, name):
    acc = accuracy_score(dtest_y, predictions)
    print("\naccuracy_score :", acc)
    report = classification_report(dtest_y, predictions)
    print(report)
    results = (' '.join(["{0:.2f}".format(acc), "{0:.2f}".format(precision_score(dtest_y, predictions)),
                         "{0:.2f}".format(recall_score(dtest_y, predictions)),
                         "{0:.2f}".format(f1_score(dtest_y, predictions))]))
    plt.figure()
    plt.subplot(221)
    conf_mat = sns.heatmap(confusion_matrix(dtest_y, predictions), annot=True, fmt="d", linecolor="k", linewidths=3)
    conf_mat = conf_mat.get_figure()
    conf_mat.savefig(path + name + '_macierzpomylek.png')
    return results


params = [{'k': 0.9, 'a': 0.25, 'var': 0.7, 'voting_mode': 'majority', 'pca_mode': 'normal'},
          {'k': 0.9, 'a': 0.5, 'var': 0.7, 'voting_mode': 'majority', 'pca_mode': 'normal'},
          {'k': 0.9, 'a': 0.75, 'var': 0.7, 'voting_mode': 'majority', 'pca_mode': 'normal'},
          {'k': 0.3, 'a': 0.5, 'var': 0.7, 'voting_mode': 'majority', 'pca_mode': 'normal'},
          {'k': 5, 'a': 0.5, 'var': 0.7, 'voting_mode': 'majority', 'pca_mode': 'normal'},
          {'k': 0.9, 'a': 0.5, 'var': 0.95, 'voting_mode': 'majority', 'pca_mode': 'normal'},
          {'k': 10, 'a': 0.25, 'var': 0.7, 'voting_mode': 'or', 'pca_mode': 'normal'},
          {'k': 10, 'a': 0.25, 'var': 0.95, 'voting_mode': 'or', 'pca_mode': 'normal'},
          {'k': 10, 'a': 0.25, 'var': 1, 'voting_mode': 'or', 'pca_mode': 'none'}]

k = 0.25
alpha = 0.25
lag_size = 2000
window_size = 2000
edz = 2000
variance_ratio=0.7
pca_mode = 'normal'
voting_mode = 'or'
features = ['h1', 'h2', 'h3']

data = []
for filename in os.listdir(path_to_data):
    data.append((filename, pd.read_csv(path_to_data + filename)))
# stationary_data.index = (stationary_data.index - min(stationary_data.index))
# i = 0
data_list = []
metrics = []
times_real = []
times_predicted = []
all_test = []
all_results = []
for par in params:
    data_list = []
    metrics = []
    times_real = []
    times_predicted = []
    all_test = []
    all_results = []
    k = par['k']
    alpha = par['a']
    variance_ratio = par['var']
    pca_mode = par['pca_mode']
    voting_mode = par['voting_mode']
    features = ['h1', 'h2', 'h3']
    baffle_alg = BAFFLE(lag_size, window_size, k=k, alpha=alpha, explained_variance_ratio=variance_ratio, voting_mode=voting_mode, pca_mode=pca_mode)
    # mah_classifer = MahClassifer(lag_size, window_size)

    for info in data:
        part_data = info[1]
        dtest_y = part_data['error'].values[lag_size:]

        result, E_arr, W_arr, y, std_results, mean_results, projection_results = baffle_alg.fit_and_predict(part_data[features].values)

        result = result[lag_size:]
        first_real = np.argmax(dtest_y == True)
        first_predicted = np.argmax(result == True)
        times_real.append(first_real)
        times_predicted.append(first_predicted)
        acc = accuracy_score(dtest_y, result)
        print("\naccuracy_score :", acc)
        metrics.append(' '.join(["{0:.2f}".format(acc), "{0:.2f}".format(precision_score(dtest_y, result)), "{0:.2f}".format(recall_score(dtest_y, result)), "{0:.2f}".format(f1_score(dtest_y, result))]))
        all_results.append(result)
        all_test.append(dtest_y)

    name = '_'.join([str(k), str(alpha), str(variance_ratio), str(pca_mode), str(voting_mode)])
    detection_time_results = create_confusion_from_times(times_real, times_predicted, edz, name)
    summary = verify(np.concatenate(all_test, axis=None), np.concatenate(all_results, axis=None), name)
    with open(path+'_summary_' + name+'_.txt', 'w') as f:
        f.writelines(summary)
    with open(path+'_edz_' + name+'_.txt', 'w') as f:
        f.writelines(detection_time_results)
    with open(path+name+'_.txt', 'w') as f:
        for line in metrics:
            f.writelines(line)
            f.writelines("\n")
    with open(path+'_dt_' + name+'_.txt', 'w') as f:
        for x, y in zip(times_real, times_predicted):
            f.writelines(' '.join([str(x), str(y), str(abs(x-y))]))
            f.writelines("\n")