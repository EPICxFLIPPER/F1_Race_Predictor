from timeSeries import createModel
from timeSeries import getDevice
from timeSeries import getLabelEncoder
from createData import getPastPostion
import pandas as pd
import torch

driverNumberToPostion = ({2:0, 3:0, 4:0, 10:0, 11:0,
                          14:0, 16:0, 18:0, 20:0, 22:0,
                          23:0, 24:0, 27:0, 31:0, 33:0,
                          44:0, 55:0, 63:0, 77:0, 81:0})

##Effects: Tanks in an array of 5 previous scoring positons, and returns an interger with the predicted postions
def predict_next_position(past_positions,model):
    label_encoder = getLabelEncoder()
    device = getDevice()
    # Ensure past_positions length is 5 (t-1 to t-5)
    if len(past_positions) != 5:
        raise ValueError("Input past_positions must have exactly 5 elements.")

    # Encode the past positions
    encoded_positions = label_encoder.transform(past_positions)

    # Convert to PyTorch tensor and reshape
    input_tensor = torch.tensor(encoded_positions, dtype=torch.long).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
        predicted_index = output.argmax(dim=1).cpu().numpy()[0]

    # Decode the predicted index
    predicted_position = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_position


##Turns the past data dataframe from pandas into an array
##         Most Recent first in array = [(t-1) (t-2) (t-3) ... ]
def pastToArray(past_positions):
    print("stub")

def encodeResults():
    for key in driverNumberToPostion.keys():
        ##Pull the data if it is there
        try:
            past_positions = pd.read_pickle('Data/Past' + str(key) + '.pkl')
        except FileNotFoundError as e:
            past_positions = getPastPostion(key)

        pastToArray(past_positions)
        model = createModel(key)
        predicted_position = predict_next_position(past_positions,model)
        driverNumberToPostion[key] = predicted_position

        



