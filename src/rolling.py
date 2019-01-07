import LoadData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#def normalize_df(df):
#    normalized_df = (df - df.min()) / (df.max() - df.min())
#    return normalized_df
#
#
experiments = LoadData.LoadData()
data = experiments.build_data_frame()
all_data = data['023']
#all_data.plot(subplots=True, figsize=(20,15))
#plt.show()
#features = ['h1', 'h2', 'h3', 'error']
#window_size = 10000
#e1 = all_data['h1'].rolling(window=window_size)
#e2 = all_data['h2'].rolling(window=window_size)
#e3 = all_data['h3'].rolling(window=window_size)
##make new features
#cor12 = e1.corr(e2)
#cor13 = e1.corr(all_data['h3'])
#cor23 = e2.corr(all_data['h3'])
#m1 = e1.mean()
#m2 = e2.mean()
#m3 = e3.mean()
#vari1 = e1.std()
#vari2 = e2.std()
#vari3 = e3.std()
# data_merged = np.array([cor12, cor13, cor23, m1, m2, m3, vari1, vari2, vari3, all_data['error']])
# df_merged = pd.DataFrame(data_merged.T, columns=['cor12', 'cor13', 'cor23', 'm1', 'm2', 'm3', 'vari1', 'vari2', 'vari3','error'])
# df_merged = df_merged.dropna()
# df_merged.plot(subplots=True)
# plt.show()
# all_data.plot(subplots=True)
# plt.show()

#normalize_df = normalize_df(all_data[150:300][['h1', 'h2', 'h3']])
#data = normalize_df.values
# all_data[150:260][['h1', 'h2', 'h3']].plot(subplots=True)
# plt.show()
#plt.figure(figsize=(20,10))
#data = all_data[150:550][['h1', 'h2', 'h3']].values
all_data = data['009']

event_data = all_data[210:270]
stationary_data = all_data[180:240]

data=stationary_data[['h1', 'h2', 'h3']].dropna().values
plt.plot(data)
# data = all_data[['h1', 'h2', 'h3']].values
from BAFFLE import BAFFLE
lag_size = 2000
window_size = 2000
baffle_alg = BAFFLE(lag_size, window_size, k=0.9, alpha=0.5, explained_variance_ratio=0.7)
E, W, y, std_results, mean_results, projection_results = baffle_alg.fit_and_predict(data)
E_arr = np.array(E)
W_arr = np.array(W)
std_results = np.array(std_results)
mean_results = np.array(mean_results)
dim = E_arr.shape[1]
E_arr = np.concatenate((np.zeros((lag_size, dim)), E_arr))
W_arr = np.concatenate((np.zeros((lag_size, dim)), W_arr))
std_results = np.concatenate((np.tile(std_results[0,:], (lag_size,1)), std_results))
mean_results = np.concatenate((np.tile(mean_results[0,:], (lag_size,1)), mean_results))
plt.figure()
for i in range(dim):
    plt.subplot(3,1,i+1)
    plt.plot(E_arr[:,i])
plt.figure()
for i in range(dim):
    plt.subplot(3,1,i+1)
    plt.plot(W_arr[:,i])

for i in range(dim):
    plt.figure(figsize=(10,6))
    plt.xlabel("czas(s)")
    plt.plot(projection_results[:,i])

    plt.plot(3*std_results[:,i] + mean_results[:,i])
    plt.plot(mean_results[:,i] - 3 * std_results[:, i])
    plt.plot(3.9 * std_results[:, i] + mean_results[:, i])
    plt.plot(mean_results[:, i] - 3.9 * std_results[:, i])
    # plt.plot(3 * (std_results[:,i]+0.9))
    plt.plot(mean_results[:, i])
    plt.legend(('y', 'mean + 3*std', 'mean - 3*std', 'mean + (3+k)*std', 'mean + (3+k)*std', 'mean'),
               loc='lower left')

plt.show()


#plt.subplot(3,1,3)
#plt.plot(E_arr[:,2])

# plt.figure()
# plt.subplot(3,1,1)
# plt.plot(W_arr[:,0])
# plt.subplot(3,1,2)
# plt.plot(W_arr[:,1])
#plt.subplot(3,1,3)
#plt.plot(W_arr[:,2])


# from sklearn.decomposition import TruncatedSVD
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# svd = TruncatedSVD(n_components=2)
# result_pca = pca.fit_transform(data)
# result_svd = svd.fit_transform(data)