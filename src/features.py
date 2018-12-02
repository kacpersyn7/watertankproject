#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 02:38:21 2018

@author: kacper
"""
from pyts.decomposition import SSA
from scipy import signal
from sklearn.preprocessing import StandardScaler
my = result['004'][['h1', 'h2', 'h3']][700:880]
features = my.values
#pca = PCA(n_components=1)
ssa = SSA(window_size = 100)
#X = ssa.fit_transform(features.T)
#deco = pca.fit_transform(features)

h1 = result['004'][['h1']]
h2 = result['004'][['h2']]
h3 = result['004'][['h3']]
h1 = h1.values
h2 = h2.values
h3 = h3.values
pattern = features # X[:,0]
scaler = StandardScaler()
h1_scale = scaler.fit_transform(h1)
h2_scale = scaler.fit_transform(h2)
h3_scale = scaler.fit_transform(h3)
pattern_xd_1 = scaler.fit_transform(pattern[:,0].reshape(-1,1))
pattern_xd_2 = scaler.fit_transform(pattern[:,1].reshape(-1,1))
pattern_xd_3 = scaler.fit_transform(pattern[:,2].reshape(-1,1))

aaa1 = signal.correlate(h1, pattern_xd_1, mode='same')
aaa2 = signal.correlate(h2, pattern_xd_2, mode='same')
aaa3 = signal.correlate(h3, pattern_xd_3, mode='same')

plt.plot(lul[70000:88000])
result['004'][['h1','h2', 'h3']][700:880].plot()
#plt.plot(pattern_xd_1)
#aaa2 = signal.correlate(h2, pattern_xd_2 mode='same')
#aaa2 = signal.correlate(h2, pattern_xd_2, mode='same')
#from scipy import signal
#aaa2 = signal.correlate(h2, pattern_xd_2, mode='same')
#plt.plot(aaa2)
#plt.plot(h2)
#aaa1 = signal.correlate(h1, pattern_xd_1, mode='same')
#aaa3 = signal.correlate(h3, pattern_xd_3, mode='same')
#plt.plot(aaa1)
#plt.plot(aaa1+aaa2+aaa3)
#result['004'].plot()
#result['004'].plot(subplots=True)
#plt.plot(aaa1+aaa2+aaa3)
#result['004'][750:900].plot(subplots=True)
#result['004'][['h1','h2', 'h3']][700:880].plot()
#lul = aaa1+aaa2+aaa3
#plt.plot(lul[70000:88000])
#plt.plot(np.arrange(700,880,0.1),lul[70000:88000])
#plt.plot(np.linspace(700,880,0.1),lul[70000:88000])
#plt.plot(lul[70000:88000])
#result['004'][['h1','h2', 'h3']][700:880].plot()