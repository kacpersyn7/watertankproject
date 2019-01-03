# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 15:50:47 2019

@author: Kacper
"""

import LoadData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize_df(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


experiments = LoadData.LoadData()
data = experiments.build_data_frame()
all_data = data['023']
all_data.plot(subplots=True, figsize=(20,15))
plt.show()
features = ['h1', 'h2', 'h3', 'error']
window_size = 6000
e1 = all_data['h1'].rolling(window=window_size)
e2 = all_data['h2'].rolling(window=window_size)
e3 = all_data['h3'].rolling(window=window_size)
#make new features
cor12 = e1.corr(e2)
cor13 = e1.corr(all_data['h3'])
cor23 = e2.corr(all_data['h3'])
m1 = e1.mean()
m2 = e2.mean()
m3 = e3.mean()
vari1 = e1.std()
vari2 = e2.std()
vari3 = e3.std()
cov1 = e1.cov(e1)
cov2 = e2.cov(e2)
cov3 = e3.cov(e3)

data_merged = np.array([cov1, cov2, cov3, all_data['error']])
df_merged = pd.DataFrame(data_merged.T, columns=['cov1', 'cov2', 'cov3','error'])
df_merged = df_merged.dropna()
df_merged.plot(subplots=True, figsize=(20,15))
plt.show()
