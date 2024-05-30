import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from TSDS import *
import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cup'
VERData = pd.read_pickle('Data/33.pkl')
VERData['date'] = pd.to_datetime(VERData['date'])
VERData = VERData[['date','position']]


def prepData(data,steps):
    df = dc(data)
    df['date'] = pd.to_datetime(df['date'])

    df.set_index('date', inplace=True)

    for i in range(1,steps):
        df[f'position(t-{i})'] = df['position'].shift(i)

    df.dropna(inplace = True)

    return df

shiftData = prepData(VERData,10)
npShiftData = shiftData.to_numpy()

##Now I need to transfor the data so values are between -1,1

scaler = MinMaxScaler(feature_range=(-1,1))
npShiftData = scaler.fit_transform(npShiftData)

### Split the data into training and testing sets
y = npShiftData[:, 0] 
X = npShiftData[:,1:]
X = dc(np.flip(X, axis = 1))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##Reshape the data for pytorch
X_test = X_test.reshape((-1,9,1))
X_train = X_train.reshape((-1,9,1))
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()

print(X_train.shape)
print(X_test.shape)
print(y_test.shape)
print(y_train.shape)

train_data = TimeSeriesDataset(X_train,y_train)
test_data = TimeSeriesDataset(X_test,y_test)