import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error
from openpyxl import Workbook, load_workbook
from datetime import datetime
import sys
import random

from google.colab import drive
drive.mount('/content/drive')



file1_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/83.csv'
file2_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/115.csv'
file3_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/67.csv'
file4_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/170.csv'
file5_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/9.csv'
file6_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/71.csv'
file7_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/88.csv'
file8_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/122.csv'

file9_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/10.csv'
file10_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/11.csv'
file11_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/21.csv'
file12_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/66.csv'

file13_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/84.csv'
file14_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/89.csv'
file15_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/116.csv'
file16_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/117.csv'

file17_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/123.csv'
file18_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/143.csv'
file19_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/171.csv'
file20_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/172.csv'
file21_path = '/content/drive/My Drive/MoE Project/Dataset/Data/MoE Data/173.csv'

#create dataframe and set index
df1 = pd.read_csv(file1_path)
# Define the string to remove
remove_string1 = '_83.npy'
remove_string2 = '_115.npy'
remove_string3 = '_67.npy'
remove_string4 = '_170.npy'
remove_string5 = '_9.npy'
remove_string6 = '_71.npy'
remove_string8 = '_122.npy'
remove_string9 = '_10.npy'
remove_string10 = '_11.npy'
remove_string11 = '_21.npy'
remove_string12 = '_66.npy'
remove_string13 = '_84.npy'
remove_string15 = '_116.npy'
remove_string16 = '_117.npy'
remove_string17 = '_123.npy'
remove_string18 = '_143.npy'
remove_string19 = '_171.npy'
remove_string20 = '_172.npy'
remove_string21 = '_173.npy'

#remove prefix from column header
def remove_prefix_1(col_name):
    return col_name.replace(remove_string1, '')

def remove_prefix_2(col_name):
    return col_name.replace(remove_string2, '')

def remove_prefix_3(col_name):
    return col_name.replace(remove_string3, '')

def remove_prefix_4(col_name):
    return col_name.replace(remove_string4, '')

def remove_prefix_5(col_name):
    return col_name.replace(remove_string5, '')

def remove_prefix_6(col_name):
    return col_name.replace(remove_string6, '')

def remove_prefix_8(col_name):
    return col_name.replace(remove_string8, '')

def remove_prefix_9(col_name):
    return col_name.replace(remove_string9, '')

def remove_prefix_10(col_name):
    return col_name.replace(remove_string10, '')

def remove_prefix_11(col_name):
    return col_name.replace(remove_string11, '')

def remove_prefix_12(col_name):
    return col_name.replace(remove_string12, '')

def remove_prefix_13(col_name):
    return col_name.replace(remove_string13, '')

def remove_prefix_15(col_name):
    return col_name.replace(remove_string15, '')

def remove_prefix_16(col_name):
    return col_name.replace(remove_string16, '')

def remove_prefix_17(col_name):
    return col_name.replace(remove_string17, '')

def remove_prefix_18(col_name):
    return col_name.replace(remove_string18, '')

def remove_prefix_19(col_name):
    return col_name.replace(remove_string19, '')

def remove_prefix_20(col_name):
    return col_name.replace(remove_string20, '')

def remove_prefix_21(col_name):
    return col_name.replace(remove_string21, '')

# Apply the function to rename columns
df1.rename(columns=remove_prefix_1, inplace=True)

df2 = pd.read_csv(file2_path)
df2.rename(columns=remove_prefix_2, inplace=True)

df3 = pd.read_csv(file3_path)
df3.rename(columns=remove_prefix_3, inplace=True)

df4 = pd.read_csv(file4_path)
df4.rename(columns=remove_prefix_4, inplace=True)

df5 = pd.read_csv(file5_path)
df5.rename(columns=remove_prefix_5, inplace=True)

df6 = pd.read_csv(file6_path)
df6.rename(columns=remove_prefix_6, inplace=True)

df8 = pd.read_csv(file8_path)
df8.rename(columns=remove_prefix_8, inplace=True)

df9 = pd.read_csv(file9_path)
df9.rename(columns=remove_prefix_9, inplace=True)

df10 = pd.read_csv(file10_path)
df10.rename(columns=remove_prefix_10, inplace=True)

df11 = pd.read_csv(file11_path)
df11.rename(columns=remove_prefix_11, inplace=True)

df12 = pd.read_csv(file12_path)
df12.rename(columns=remove_prefix_12, inplace=True)

df13 = pd.read_csv(file13_path)
df13.rename(columns=remove_prefix_13, inplace=True)

df15 = pd.read_csv(file15_path)
df15.rename(columns=remove_prefix_15, inplace=True)

df16 = pd.read_csv(file16_path)
df16.rename(columns=remove_prefix_16, inplace=True)

df17 = pd.read_csv(file17_path)
df17.rename(columns=remove_prefix_17, inplace=True)

df18 = pd.read_csv(file18_path)
df18.rename(columns=remove_prefix_18, inplace=True)

df19 = pd.read_csv(file19_path)
df19.rename(columns=remove_prefix_19, inplace=True)

df20 = pd.read_csv(file20_path)
df20.rename(columns=remove_prefix_20, inplace=True)

df21 = pd.read_csv(file21_path)
df21.rename(columns=remove_prefix_21, inplace=True)

#Data Normalization
scaler = MinMaxScaler(feature_range=(-1,1))

def min_max_scaler(df):
    return 2 * ((df - df.min()) / (df.max() - df.min())) - 1

def normalize_df1(df):
    numeric_cols = df.select_dtypes(include='number').iloc[:, 1:-1]
    df.iloc[:, 1:-1] = min_max_scaler(numeric_cols)
    return df

def normalize_df(df):
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = min_max_scaler(df[numeric_cols])
    return df

# Apply the different normalization functions
df2 = normalize_df(df2.loc[:, df2.columns != 'Grid'])
df2 = df2.fillna(0)

df3 = normalize_df(df3.loc[:, df3.columns != 'Grid'])
df3 = df3.fillna(0)

df4 = normalize_df(df4.loc[:, df4.columns != 'Grid'])
df4 = df4.fillna(0)

df5 = normalize_df(df5.loc[:, df5.columns != 'Grid'])
df5 = df5.fillna(0)

df6 = normalize_df(df6.loc[:, df6.columns != 'Grid'])
df6 = df6.fillna(0)

df8 = normalize_df(df8.loc[:, df8.columns != 'Grid'])
df8 = df8.fillna(0)

df9 = normalize_df(df9.loc[:, df9.columns != 'Grid'])
df9 = df9.fillna(0)

df10 = normalize_df(df10.loc[:, df10.columns != 'Grid'])
df10 = df10.fillna(0)

df11 = normalize_df(df11.loc[:, df11.columns != 'Grid'])
df11 = df11.fillna(0)

df12 = normalize_df(df12.loc[:, df12.columns != 'Grid'])
df12 = df12.fillna(0)

df13 = normalize_df(df13.loc[:, df13.columns != 'Grid'])
df13 = df13.fillna(0)

df15 = normalize_df(df15.loc[:, df15.columns != 'Grid'])
df15 = df15.fillna(0)

df16 = normalize_df(df16.loc[:, df16.columns != 'Grid'])
df16 = df16.fillna(0)

df17 = normalize_df(df17.loc[:, df17.columns != 'Grid'])
df17 = df17.fillna(0)

df18 = normalize_df(df18.loc[:, df18.columns != 'Grid'])
df18 = df18.fillna(0)

df19 = normalize_df(df19.loc[:, df19.columns != 'Grid'])
df19 = df19.fillna(0)

df20 = normalize_df(df20.loc[:, df20.columns != 'Grid'])
df20 = df20.fillna(0)