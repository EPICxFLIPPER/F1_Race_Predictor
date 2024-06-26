import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim as optim
import createData
from sklearn.model_selection import train_test_split

##TODO Add Qualifying position
##TODO Current Driver standings to break ties
##TODO Use driver standings after a race

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
label_encoder = LabelEncoder()

class TimeSeriesDataset(Dataset):
    def __init__(self, X_pos, X_grid, y):
        self.X_pos = X_pos
        self.X_grid = X_grid
        self.y = y

    def __len__(self):
        return len(self.X_pos)

    def __getitem__(self, idx):
        return self.X_pos[idx], self.X_grid[idx], self.y[idx]

# Define the LSTM model with an embedding layer
class LSTM(nn.Module):
    def __init__(self, num_classes, hidden_size, num_stacked_layers):
        super().__init__()
        self.embedding_pos = nn.Embedding(num_classes, hidden_size)
        self.embedding_grid = nn.Embedding(num_classes, hidden_size)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size, num_stacked_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x_pos, x_grid):
        x_pos = self.embedding_pos(x_pos)
        x_grid = self.embedding_grid(x_grid)
        # Ensure sequence lengths match
        if x_pos.size(1) != x_grid.size(1):
            min_len = min(x_pos.size(1), x_grid.size(1))
            x_pos = x_pos[:, :min_len, :]
            x_grid = x_grid[:, :min_len, :]
        x = torch.cat((x_pos, x_grid), dim=2)
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = torch.relu(self.fc1(out[:, -1, :]))
        out = self.fc2(out)
        return out
##Effects: Feteches the data and takes only the date, grid and position columns. 
##         If the data does not exist, makes the data and places it in the data folder.
##         When Creating the data, the program will default to the past 10 years.
def fetchData(number):
    try:
        posData = pd.read_pickle('Data/' + str(number) + '.pkl')
    except FileNotFoundError as e:
        print(e)
        createData.getDriverData(number,10)
        posData = pd.read_pickle('Data/' + str(number) + '.pkl')


    try:
        qualData = pd.read_pickle('Data/qual' + str(number) + '.pkl')
    except FileNotFoundError as e:
        print(e)
        createData.getQuliData(number,10)
        qualData = pd.read_pickle('Data/qual' + str(number) + '.pkl')

    posData['date'] = pd.to_datetime(posData['date'])
    qualData['date'] = pd.to_datetime(qualData['date'])

    combinedData = pd.merge(posData,qualData, on=['date'])

    combinedData = combinedData[['date', 'position','grid']]

    print(combinedData)
    return combinedData

##Effects: Turns the position column into a numberic category for prediction
def categorize(data):
    # Ensure all entries in the 'position' column are integers
    data['position'] = data['position'].astype(int)
    data['grid'] = data['grid'].astype(int)

    all_positions = pd.DataFrame({'position': np.arange(1, 21)})
    all_grid = pd.DataFrame({'grid': np.arange(1,21)})

    data_with_all_positions = pd.concat([data, all_positions], ignore_index=True)
    data_with_all_grid = pd.concat([data, all_grid], ignore_index=True)
    label_encoder.fit(data_with_all_positions['position'])
    label_encoder.fit(data_with_all_positions['grid'])

    data['position'] = label_encoder.fit_transform(data['position'])
    data['grid'] = label_encoder.fit_transform(data['grid'])
    return data

##Effects: Shifts the data for the recursive model, adding in as many columns as steps needed.
def shiftDataFunc(data, steps):
    df = dc(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    for i in range(1, steps + 1):
        df[f'position(t-{i})'] = df['position'].shift(i)
        df[f'grid(t-{i})'] = df['grid'].shift(i)

    df.dropna(inplace=True)
    return df

##Effects: Returns the training and testing sets as Pytorch tensors, based on the split percentage
def trainTest(data,split,num_classes):
    X_pos = data.iloc[:, [1] + list(range(3, len(data.columns), 2))].values
    X_grid = data.iloc[:, list(range(2, len(data.columns), 2))].values
    y = data.iloc[:, 0].values

    X_pos = dc(np.flip(X_pos, axis=1))
    X_grid = dc(np.flip(X_grid, axis=1))

    min_length = min(X_pos.shape[1], X_grid.shape[1])
    X_pos = X_pos[:, :min_length]
    X_grid = X_grid[:, :min_length]

    # Use sklearn's train_test_split for random splitting
    X_train_pos, X_test_pos, X_train_grid, X_test_grid, y_train, y_test = train_test_split(
        X_pos, X_grid, y, test_size=(1 - split), random_state=42
    )

    X_train_pos = torch.tensor(X_train_pos, dtype=torch.long).clamp(0, num_classes - 1)
    X_test_pos = torch.tensor(X_test_pos, dtype=torch.long).clamp(0, num_classes - 1)
    X_train_grid = torch.tensor(X_train_grid, dtype=torch.long).clamp(0, num_classes - 1)
    X_test_grid = torch.tensor(X_test_grid, dtype=torch.long).clamp(0, num_classes - 1)
    y_train = torch.tensor(y_train, dtype=torch.long).clamp(0, num_classes - 1)
    y_test = torch.tensor(y_test, dtype=torch.long).clamp(0, num_classes - 1)

    return X_train_pos, X_test_pos, X_train_grid, X_test_grid, y_train, y_test

# Training and validation functions
def train_one_epoch(epoch,model,train_loader,scheduler,optimizer):
    # learning_rate = 0.001
    model.train(True)
    loss_function = nn.CrossEntropyLoss()
    running_loss = 0.0

    for batch_index, (x_pos_batch, x_grid_batch, y_batch) in enumerate(train_loader):
        x_pos_batch, x_grid_batch, y_batch = x_pos_batch.to(device), x_grid_batch.to(device), y_batch.to(device)
        output = model(x_pos_batch, x_grid_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 5 == 4:
            avg_loss_across_batches = running_loss / 5
            print(f'Epoch {epoch + 1}, Batch {batch_index + 1}, Loss: {avg_loss_across_batches:.3f}')
            running_loss = 0.0

def validate_one_epoch(model,test_loader):
    loss_function = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x_pos_batch, x_grid_batch, y_batch in test_loader:
            x_pos_batch, x_grid_batch, y_batch = x_pos_batch.to(device), x_grid_batch.to(device), y_batch.to(device)
            output = model(x_pos_batch, x_grid_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    print(f'Validation Loss: {avg_loss_across_batches:.3f}')
    print('***************************************************')

##Effects: Produces the training and testing plots
def plot(model, X_train_pos, X_test_pos, X_train_grid, X_test_grid, y_train, y_test, number):
    with torch.no_grad():
        train_predictions = model(X_train_pos.to(device), X_train_grid.to(device)).argmax(dim=1).cpu().numpy()
        test_predictions = model(X_test_pos.to(device), X_test_grid.to(device)).argmax(dim=1).cpu().numpy()

    train_predictions = label_encoder.inverse_transform(train_predictions)
    test_predictions = label_encoder.inverse_transform(test_predictions)
    y_train = label_encoder.inverse_transform(y_train)
    y_test = label_encoder.inverse_transform(y_test)

    train_predictions = pd.to_numeric(train_predictions)
    test_predictions = pd.to_numeric(test_predictions)
    y_train = pd.to_numeric(y_train)
    y_test = pd.to_numeric(y_test)

    plt.plot(y_train, label='Actual Position')
    plt.plot(train_predictions, label='Predicted Position')
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(1, max(y_train.max(), train_predictions.max()) + 1))
    plt.xlabel('Race')
    plt.ylabel('Position')
    plt.legend()
    plt.savefig('Plots/' + str(number) + 'TrainingPredictions.png')
    plt.close()

    plt.plot(y_test, label='Actual Position')
    plt.plot(test_predictions, label='Predicted Position')
    plt.gca().invert_yaxis()
    plt.yticks(np.arange(1, max(y_train.max(), train_predictions.max()) + 1))
    plt.xlabel('Race')
    plt.ylabel('Position')
    plt.legend()
    plt.savefig('Plots/' + str(number) + 'TestingPredictions.png')
    plt.close()

##Effects: Creates the LSTM NN model to predict drive race positons based off of their previous races
def createModel(number):
    RAWData = fetchData(number)
    catData = categorize(RAWData)

    steps = 5
    shiftedData = shiftDataFunc(catData,steps)
    ##shiftedData = shiftedData.to_numpy()
    num_classes = len(label_encoder.classes_)

    X_train_pos,X_test_pos, X_train_grid,X_test_grid, y_train,y_test = trainTest(shiftedData,.70,num_classes)

    train_data = TimeSeriesDataset(X_train_pos,X_train_grid, y_train)
    test_data = TimeSeriesDataset(X_test_pos,X_test_grid, y_test)

    train_loader = DataLoader(train_data, batch_size=3, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=3, shuffle=False)

    model = LSTM(num_classes, hidden_size=16, num_stacked_layers=3)
    model.to(device)


    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1) 
    ##Training Loop
    num_epochs = 15
    for epoch in range(num_epochs):
        train_one_epoch(epoch,model,train_loader,scheduler,optimizer)
        validate_one_epoch(model,test_loader)

    plot(model, X_train_pos, X_test_pos, X_train_grid, X_test_grid, y_train, y_test, number)

    return model

def getDevice():
    return device

def getLabelEncoder():
    return label_encoder


createModel(33)