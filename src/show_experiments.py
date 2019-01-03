from BAFFLE import *
from LoadData import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
def normalize_df(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df
experiments = LoadData()
tanks_data = experiments.build_data_frame()
tanks_data['017'].plot(subplots=True)
plt.show()
valid_data = tanks_data['010'][400:500][['h1', 'h2', 'h3']]
valid_data.plot(subplots=True)
valid_data.hist()
plt.show()
normalize_df = normalize_df(valid_data)
normalize_df.plot(subplots=True)
normalize_df.hist()
plt.show()
#data_pca = pca.fit_transform(normalize_df)
#plt.figure()
#plt.plot(data_pca)
#plt.figure()
#plt.hist(data_pca, bins ="auto")
#plt.show()