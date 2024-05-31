import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
from sklearn.preprocessing import LabelEncoder

from TSDS import *

import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
VERData = pd.read_pickle('Data/33.pkl')
VERData['date'] = pd.to_datetime(VERData['date'])
VERData = VERData[['date','position']]
plt.scatter(VERData['date'],VERData['position'])
plt.savefig('Plots/MaxWinPlot.png')
plt.close()

##Make the position categorical
label_encoder = LabelEncoder()
VERData['position'] = label_encoder.fit_transform(VERData['position'])

class LSTM(nn.Module):
    def __init__(self, num_classes, hidden_size, num_stacked_layers):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_stacked_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

def prepData(data,steps):
    df = dc(data)
    df['date'] = pd.to_datetime(df['date'])

    df.set_index('date', inplace=True)

    for i in range(1,steps):
        df[f'position(t-{i})'] = df['position'].shift(i)

    df.dropna(inplace = True)

    return df

shiftData = prepData(VERData,6)
npShiftData = shiftData.to_numpy()

##Now I need to transfor the data so values are between -1,1

scaler = MinMaxScaler(feature_range=(-1,1))
npShiftData = scaler.fit_transform(npShiftData)


X = npShiftData[:, 1:]
y = npShiftData[:, 0]
X = dc(np.flip(X, axis=1))
split_index = int(len(X) * 0.80)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##Reshape the data for pytorch
X_test = X_test.reshape((-1,5,1))
X_train = X_train.reshape((-1,5,1))
y_train = y_train.reshape((-1,1))
y_test = y_test.reshape((-1,1))

X_train = torch.tensor(X_train, dtype=torch.long)
X_test = torch.tensor(X_test, dtype=torch.long)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

train_data = TimeSeriesDataset(X_train,y_train)
test_data = TimeSeriesDataset(X_test,y_test)

train_loader = DataLoader(train_data,batch_size=3,shuffle=True)
test_loader = DataLoader(test_data,batch_size=3,shuffle=False)

num_classes = len(label_encoder.classes_)
model = LSTM(num_classes, hidden_size=16, num_stacked_layers=2)
model.to(device)

def train_one_epoch(epoch):
    learning_rate = 0.001
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train(True)
    print(f'Epoch:{epoch+1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output = model(x_batch)
        loss = loss_function(output,y_batch)
        running_loss += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 5 == 4:
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,avg_loss_across_batches))
            running_loss = 0.0
    print()


def validate_one_epoch():
    loss_function = nn.CrossEntropyLoss()
    model.train(False)
    running_loss = 0.0
    
    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        
        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    
    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()

num_epochs = 10

for epoch in range(num_epochs):
    train_one_epoch(epoch)
    validate_one_epoch()

print(shiftData)

with torch.no_grad():
    train_predictions = model(X_train.to(device)).argmax(dim=1).cpu().numpy()
    test_predictions = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

# Inverse transform the predictions and actual values - CHANGED
train_predictions = label_encoder.inverse_transform(train_predictions)
test_predictions = label_encoder.inverse_transform(test_predictions)
y_train = label_encoder.inverse_transform(y_train)
y_test = label_encoder.inverse_transform(y_test)

# Plotting - UNCHANGED
plt.plot(y_train, label='Actual Position')
plt.plot(train_predictions, label='Predicted Position')
plt.xlabel('Race')
plt.ylabel('Position')
plt.legend()
plt.savefig('Plots/TrainingPredictions.png')
plt.close()

plt.plot(y_test, label='Actual Position')
plt.plot(test_predictions, label='Predicted Position')
plt.xlabel('Race')
plt.ylabel('Position')
plt.legend()
plt.savefig('Plots/TestingPredictions.png')
plt.close()