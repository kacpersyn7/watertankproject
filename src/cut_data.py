# -*- coding: utf-8 -*-
import LoadData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

experiments = LoadData.LoadData()
data = experiments.build_data_frame()
window_size = 6000


column_names = list(set([column[0] for column in data]))
column_names.sort()
time_intervals_dict = {}
for name in column_names: 
    all_data=data[name]
    all_data.plot(subplots=True, figsize=(20,15))
    e1 = all_data['h1'].rolling(window=window_size)
    e2 = all_data['h2'].rolling(window=window_size)
    e3 = all_data['h3'].rolling(window=window_size)
    
    cov1 = e1.cov(e1)
    cov2 = e2.cov(e2)
    cov3 = e3.cov(e3)
    data_merged = np.array([cov1, cov2, cov3, all_data['error']])
    df_merged = pd.DataFrame(data_merged.T, columns=['cov1', 'cov2', 'cov3','error'])
    df_merged = df_merged.fillna(1)
    df_merged.plot(subplots=True, figsize=(20,15))
    plt.show()
    time_intervals_dict[name] = []
    while True:
        start = input("type start time ")
        start = int(start)
        if start == 0:
            break
        end = start + 180
        time_intervals_dict[name].append((start, end))
    print(time_intervals_dict)
    
with open('time_intervals.json', 'w') as fp:
    json.dump(time_intervals_dict, fp)