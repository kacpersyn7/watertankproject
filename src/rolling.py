import LoadData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def normalize_df(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


experiments = LoadData.LoadData()
data = experiments.build_data_frame()
all_data = data['011']
features = ['h1', 'h2', 'h3', 'error']
window_size = 10000
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
data_merged = np.array([cor12, cor13, cor23, m1, m2, m3, vari1, vari2, vari3, all_data['error']])
df_merged = pd.DataFrame(data_merged.T, columns=['cor12', 'cor13', 'cor23', 'm1', 'm2', 'm3', 'vari1', 'vari2', 'vari3','error'])
df_merged = df_merged.dropna()
df_merged.plot(subplots=True)
plt.show()
all_data.plot(subplots=True)
plt.show()

normalize_df = normalize_df(all_data[150:400][['h1', 'h2', 'h3']])
data = normalize_df.values
data = all_data[150:400][['h1', 'h2', 'h3']].values
# data = all_data[['h1', 'h2', 'h3']].values
from BAFFLE import BAFFLE
baffle_alg = BAFFLE(8000, 1000, 0.1, 0.2)
E, W = baffle_alg.fit_and_predict(data)
E_arr = np.array(E)
W_arr = np.array(W)

plt.figure()
plt.subplot(3,1,1)
plt.plot(E_arr[:,0])
plt.subplot(3,1,2)
plt.plot(E_arr[:,1])
plt.subplot(3,1,3)
plt.plot(E_arr[:,2])

plt.figure()
plt.subplot(3,1,1)
plt.plot(W_arr[:,0])
plt.subplot(3,1,2)
plt.plot(W_arr[:,1])
plt.subplot(3,1,3)
plt.plot(W_arr[:,2])

plt.show()
# from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# svd = TruncatedSVD(n_components=2)
# result_pca = pca.fit_transform(data)
# result_svd = svd.fit_transform(data)