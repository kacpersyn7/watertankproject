import seaborn as sns
import LoadData
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.set(style="white", color_codes=True)
pca = PCA(n_components=2)

def normalize_df(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df

def iterate_over_data(data):
    column_names = list(set([column[0] for column in data]))
    column_names.sort()
    for name in column_names:
        print(data[name].head())

def visualize(df,name):
    features = ['h1', 'h2', 'h3']
    features_to_show = ['h1 (cm)', 'h2 (cm)', 'h3 (cm)']
    df_plot = df[features].plot(subplots=True, legend=False, grid=True)
    for i, name in enumerate(features_to_show):
        df_plot[i].set_ylabel(name)
    plt.xlabel('czas (s)')
    plt.show()
    df_plot=df_plot[0].get_figure()
    df_plot.savefig('wysokosci_' + name + '.png')
    #histograms 
    for feature in features: 
        hist_plot = sns.distplot(df[feature], axlabel='wysokosc', kde=False, label=feature)
    plt.legend()
    hist_plot = hist_plot.get_figure()
    hist_plot.savefig('histogramy_' + name + '.png')
    plt.show()
    
    sns.pairplot(df[features])
    
    plt.show()
    box_plot = sns.boxplot(data=df[features], width=0.2)
    box_plot = box_plot.get_figure()

    box_plot.savefig('boxploty_' + name + '.png')
    plt.show()

    normalized_df = normalize_df(df[features])
    reduced_np = pca.fit_transform(normalized_df.values)
    print(pca.explained_variance_ratio_)
    reduced_df = pd.DataFrame(reduced_np, columns=['x', 'y'])
    # density plot with histograms (KDE)

    sns.jointplot("x", "y", data=reduced_df, kind='hex', height=10)
    sns.jointplot("x", "y", data=reduced_df, kind="kde", space=0, color="g", height=10)
    
features = ['h1', 'h2', 'h3', 'v1', 'v2', 'v3', 'u']
experiments = LoadData.LoadData()
data = experiments.build_data_frame()
all_data = data['009']
stationary_data = all_data[180:240]
stationary_data.index = (stationary_data.index - min(stationary_data.index))
event_data = all_data[210:270]
event_data.index = (event_data.index - min(event_data.index))
even_plot=event_data[features].plot(subplots=True, legend=False, figsize=(10,10), grid=True, lw=3)

features_to_show = ['h1 (cm)', 'h2 (cm)', 'h3 (cm)', 'v1', 'v2', 'v3', 'u']

for i, name in enumerate(features_to_show):
    even_plot[i].set_ylabel(name)
for i in range(0,3):
    even_plot[i].set_ylim(0,9)
for i in range(3,7):
    even_plot[i].set_ylim(0,1)

plt.xlabel('czas (s)')
even_plot=even_plot[0].get_figure()
even_plot.savefig('wszystkozdarzenie.png')
stationary_plot = stationary_data[features].plot(subplots=True, legend=False, figsize=(10,10), grid=True, lw=3)

for i, name in enumerate(features_to_show):
    print(i, name)
    stationary_plot[i].set_ylabel(name)
for i in range(0,3):
    stationary_plot[i].set_ylim(0,9)
for i in range(3,7):
    stationary_plot[i].set_ylim(0,1)
plt.xlabel('czas (s)')
stationary_plot=stationary_plot[0].get_figure()
stationary_plot.savefig('wszystkoniezdarzenie.png')



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
