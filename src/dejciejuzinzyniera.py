import LoadData
from BAFFLE import BAFFLE
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix


def normalize_df(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


def visualize(df,fname=None):
    sns.set(rc={'figure.figsize': (8, 6)})
    sns.set(style="white", color_codes=True)
    pca = PCA(n_components=2)
    features = ['h1', 'h2', 'h3']
    features_to_show = ['h1 (cm)', 'h2 (cm)', 'h3 (cm)']

    df_plot = df[features].plot(subplots=True, legend=False, grid=True)
    for i, name in enumerate(features_to_show):
        df_plot[i].set_ylabel(name)
    plt.xlabel('czas (s)')
    df_plot=df_plot[0].get_figure()
    # plt.show()

    #histograms
    plt.figure()
    for feature in features:
        hist_plot = sns.distplot(df[feature], axlabel='wysokość (cm)', kde=False, label=feature)
    plt.legend()
    hist_plot = hist_plot.get_figure()
    # plt.show()

    pair_plot = sns.pairplot(df[features])
    #
    # plt.show()
    plt.figure()

    box_plot = sns.boxplot(data=df[features], width=0.2, showfliers=False)
    box_plot = box_plot.get_figure()

    # plt.show()

    normalized_df = normalize_df(df[features])
    reduced_np = pca.fit_transform(normalized_df.values)
    print(pca.explained_variance_ratio_)
    reduced_df = pd.DataFrame(reduced_np, columns=['x', 'y'])
    # density plot with histograms (KDE)


    dist_hex_plot = sns.jointplot("x", "y", data=reduced_df, kind='hex', height=10)
    # plt.show()

    dist_plot = sns.jointplot("x", "y", data=reduced_df, kind="kde", space=0, color="g", height=10)
    # plt.show()

    plt.show()
    if fname is not None:
        df_plot.savefig(path + 'wysokosci_' + fname + '.png')
        hist_plot.savefig(path + 'histogramy_' + fname + '.png')
        box_plot.savefig(path + 'boxploty_' + fname + '.png')
        dist_hex_plot.savefig(path + 'hex_' + fname + '.png')
        dist_plot.savefig(path + 'dist_' + fname + '.png')

def verif(dtest_y, predictions):
    plt.figure()
    plt.subplot(221)
    conf_mat = sns.heatmap(confusion_matrix(dtest_y, predictions), annot=True, fmt="d", linecolor="k", linewidths=3)
    plt.title("Macierz pomyłek", fontsize=20)

path = '../results/'
path_to_data = '../data/'

experiments = LoadData.LoadData()
data = experiments.build_data_frame()
k = 0.9
alpha = 0.5
lag_size = 2000
window_size = 2000
edz = 1000
voting_mode = 'majority'
pca_mode = 'normal'
variance_ratio=0.7
data = []
for filename in os.listdir(path_to_data):
    data.append((filename, pd.read_csv(path_to_data + filename, index_col=0)))
# stationary_data.index = (stationary_data.index - min(stationary_data.index))
# i = 0

features = ['h1', 'h2', 'h3']
baffle_alg = BAFFLE(lag_size, window_size, k=k, alpha=alpha, explained_variance_ratio=variance_ratio, voting_mode=voting_mode, pca_mode=pca_mode)
# mah_classifer = MahClassifer(lag_size, window_size)

for info in data:
    part_data = info[1]
    visualize(part_data)
    dtest_y = part_data['error'].values[lag_size:]

    # result, E_arr, W_arr, y, std_results, mean_results, projection_results = baffle_alg.fit_and_predict(part_data[features].values)
    # dim = E_arr.shape[1]
    # first_real = np.argmax(dtest_y == True)
    # first_predicted = np.argmax(result == True)
    # features_to_show = ['h1 (cm)', 'h2 (cm)', 'h3 (cm)', 'v1', 'v2', 'v3', 'u', 'error']
    # # plt.show()
    # even_plot = part_data.plot(subplots=True, legend=False, figsize=(10, 10), grid=True)
    # plt.xlabel('czas (s)')
    #
    # for i, namel in enumerate(features_to_show):
    #     even_plot[i].set_ylabel(namel)
    #
    # for i in range(3, 8):
    #     even_plot[i].set_ylim(0, 1)
    #
    # # plt.show()
    # plt.figure(figsize=(10, 3))
    # plt.xlabel("czas(ms)")
    # plt.plot(result)
    #
    # result = result[lag_size:]
    #
    # plt.figure()
    # plt.xlabel("czas(ms)")
    # for i in range(dim):
    #     plt.subplot(3, 1, i + 1)
    #     plt.plot(E_arr[:, i])
    #
    # plt.figure()
    # plt.xlabel("czas(ms)")
    # for i in range(dim):
    #
    #     plt.subplot(3, 1, i + 1)
    #     plt.plot(W_arr[:, i])
    #
    # for i in range(dim):
    #     plt.figure(figsize=(10, 6))
    #     plt.xlabel("czas(ms)")
    #     plt.plot(projection_results[:, i])
    #
    #     plt.plot(3 * std_results[:, i] + mean_results[:, i])
    #     plt.plot(mean_results[:, i] - 3 * std_results[:, i])
    #     plt.plot((3+k) * std_results[:, i] + mean_results[:, i])
    #     plt.plot(mean_results[:, i] - (3+k) * std_results[:, i])
    #     # plt.plot(3 * (std_results[:,i]+0.9))
    #     plt.plot(mean_results[:, i])
    #     plt.legend(('y', 'mean + 3*std', 'mean - 3*std', 'mean + (3+k)*std', 'mean + (3+k)*std', 'mean'),
    #               loc='lower left')
    #
    # verif(dtest_y, result)
    # plt.show()
    #
    # input("next")
