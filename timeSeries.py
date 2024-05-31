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

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


##Effects: Feteches the data and takes only the date and position columns. 
##         If the data does not exist, makes the data and places it in the data folder.
##         When Creating the data, the program will default to the past 30 years.
def fetchData(number):
    try:
        RAWData = pd.read_pickle('Data/' + str(number) + '.pkl')
    except FileNotFoundError as e:
        createData.getDriverData(number,30)
        RAWData = pd.read_pickle('Data/' + str(number) + '.pkl')

    RAWData['date'] = pd.to_datetime(RAWData['date'])
    RAWData = RAWData[['date', 'position']]

    return RAWData


##Effects: Creates the LSTM NN model to predict drive race positons based off of their previous races
def createModel(number):
    RAWData = fetchData(number)


    print("stub")


# Load and preprocess data
VERData = pd.read_pickle('Data/33.pkl')
VERData['date'] = pd.to_datetime(VERData['date'])
VERData = VERData[['date', 'position']]

plt.scatter(VERData['date'], VERData['position'])
plt.savefig('Plots/MaxWinPlot.png')
plt.close()

# Encode positions
label_encoder = LabelEncoder()
VERData['position'] = label_encoder.fit_transform(VERData['position'])

# Verify the range of encoded labels
print(f"Encoded position labels range from {VERData['position'].min()} to {VERData['position'].max()}")

# Prepare the data with shifted positions
def prepData(data, steps):
    df = dc(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    for i in range(1, steps):
        df[f'position(t-{i})'] = df['position'].shift(i)

    df.dropna(inplace=True)
    return df

shiftData = prepData(VERData, 6)
npShiftData = shiftData.to_numpy()

# Split data into features and labels
X = npShiftData[:, 1:]
y = npShiftData[:, 0]
X = dc(np.flip(X, axis=1))
split_index = int(len(X) * 0.80)

X_train = X[:split_index]
X_test = X[split_index:]
y_train = y[:split_index]
y_test = y[split_index:]

# Ensure all values are within the range
num_classes = len(label_encoder.classes_)
print(f"Number of classes: {num_classes}")

# Convert data to PyTorch tensors and ensure within range
X_train = torch.tensor(X_train, dtype=torch.long).clamp(0, num_classes - 1)
X_test = torch.tensor(X_test, dtype=torch.long).clamp(0, num_classes - 1)
y_train = torch.tensor(y_train, dtype=torch.long).clamp(0, num_classes - 1)
y_test = torch.tensor(y_test, dtype=torch.long).clamp(0, num_classes - 1)

# Custom dataset class
class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = TimeSeriesDataset(X_train, y_train)
test_data = TimeSeriesDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=3, shuffle=True)
test_loader = DataLoader(test_data, batch_size=3, shuffle=False)

# Define the LSTM model with an embedding layer
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

model = LSTM(num_classes, hidden_size=16, num_stacked_layers=3)
model.to(device)

# Training and validation functions
def train_one_epoch(epoch):
    learning_rate = 0.001
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    model.train()
    running_loss = 0.0

    for batch_index, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 5 == 4:
            avg_loss_across_batches = running_loss / 5
            print(f'Epoch {epoch + 1}, Batch {batch_index + 1}, Loss: {avg_loss_across_batches:.3f}')
            running_loss = 0.0

def validate_one_epoch():
    loss_function = nn.CrossEntropyLoss()
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)
    print(f'Validation Loss: {avg_loss_across_batches:.3f}')
    print('***************************************************')

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(epoch)
    validate_one_epoch()

# Predictions
with torch.no_grad():
    train_predictions = model(X_train.to(device)).argmax(dim=1).cpu().numpy()
    test_predictions = model(X_test.to(device)).argmax(dim=1).cpu().numpy()

# Inverse transform the predictions and actual values
train_predictions = label_encoder.inverse_transform(train_predictions)
test_predictions = label_encoder.inverse_transform(test_predictions)
y_train = label_encoder.inverse_transform(y_train)
y_test = label_encoder.inverse_transform(y_test)

# Plotting
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