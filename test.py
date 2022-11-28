import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
from utils import open_file, get_features, DEGREE, res_names

df =  pd.read_csv('./data/aux/aux_data_10000_S2.csv')

MEAN_AGE = 13

print(df.shape)
#df = df[1:5000]

print(df.groupby('studentId').age.agg([('age_diff',  lambda x : np.max(x) - np.min(x))]).reset_index().max())
#.reset_index()
print(df)
print(df.dtypes)
df['month'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m')
print(df)
stop

df_group1 = df[['studentId', 'scale', 'age', 'gender', 'estimate']].groupby(['studentId','scale']).apply(get_features).reset_index()
df_group2 = df[['studentId', 'scale', 'age', 'gender', 'estimate']].groupby(['studentId']).apply(get_features).reset_index()
df_group1 = df_group1.pivot(index=['studentId'], columns='scale', values=res_names[2:]).reset_index()

colnames = ['studentId'] + df_group1.columns.tolist()[1:] 
df_group1 = df_group1.set_axis(colnames, axis=1, inplace=False)
df_group = df_group1.merge(df_group2, on='studentId').reset_index()


#df_group = df_group.loc[df_group.N >= MIN_POINTS]

print(df_group)
print(len(df_group.studentId.unique()))
print(df_group.columns)
stope
#LD = df_group.iloc[:, 2]
#print(LD)
#dfx = pd.DataFrame({k: [dic[k] for dic in LD] for k in LD[0]})
#print(dfx)
#df_group = pd.concat((df_group[['studentId','scale']], dfx), axis=1)
#.agg(mean=('estimate', 'mean'), std=('estimate', 'std'), N=('estimate', 'count')).reset_index()
#df_group = df_group.loc[df_group.N >= MINPOINTS]
#print(df_group)
