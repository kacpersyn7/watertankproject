import LoadData
from BAFFLE import BAFFLE
from MahClassifer import MahClassifer
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib.pyplot as plt
import json

path = '../data/'
experiments = LoadData.LoadData()
data = experiments.build_data_frame()
with open('time_intervals.json', 'r') as fp:
    time_intervals_dict = json.load(fp)
features = ['h1', 'h2', 'h3']
column_names = list(time_intervals_dict.keys())
column_names.sort()
column_names.insert(0, column_names.pop(column_names.index('014')))
column_names.insert(0, column_names.pop(column_names.index('013')))
column_names.insert(0, column_names.pop(column_names.index('012')))
column_names.insert(0, column_names.pop(column_names.index('011')))
column_names.insert(0, column_names.pop(column_names.index('009')))
column_names.insert(0, column_names.pop(column_names.index('008')))
column_names.insert(0, column_names.pop(column_names.index('004')))


for name in column_names:
    i = 0
    for elem in time_intervals_dict[name]:
        part = data[name][elem[0]:elem[1]]
        part = part.fillna(0)
        part.index = (part.index - min(part.index))
        part_error = part.error
        k_er = 5500
        if name in ['004', '006', '008', '009', '011', '012', '013', '014']:
            if elem[0] == 200:
                k_er = 3500
            elif elem[0] == 700:
                k_er = 6500
        elif name == '007':
            if elem[0] == 1060:
                k_er = 1500
            else:
                k_er = 3500
        elif name == '019':
            if elem[0] == 1000:
                k_er = 7500
        elif name == '020':
            if elem[0] == 450:
                k_er = 6500
            elif elem[0] == 1090:
                k_er = 3500
        elif name == '027':
            if elem[0] == 1150:
                k_er = 3000
            elif elem[0] == 1450:
                k_er = 2500
        elif name == '028':
            if elem[0] == 200:
                k_er = 4500
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
        part.error = ((part_error == 1) & (part_error.shift(15) == 1) | (part_error.shift(k_er)) | (part_error.shift(k_er // 2)))
        part.to_csv(path+'exp_'+name+'_'+str(i))
        i+=1