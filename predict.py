from timeSeries import createModel
from timeSeries import getDevice
from timeSeries import getLabelEncoder
from createData import getPastPostion
from createData import getPastQualifying
import pandas as pd
import torch

driverNumberToPostion = ({2:0, 3:0, 4:0, 10:0, 11:0,
                          14:0, 16:0, 18:0, 20:0, 22:0,
                          23:0, 24:0, 27:0, 31:0, 33:0,
                          44:0, 55:0, 63:0, 77:0, 81:0})

driverNumberToName = ({2:"SAR", 3:"RIC", 4:"NOR", 10:"GAS", 11:"PER",
                          14:"ALO", 16:"LEC", 18:"STR", 20:"MAG", 22:"TSU",
                          23:"ALB", 24:"ZHO", 27:"HUL", 31:"OCN", 33:"VER",
                          44:"HAM", 55:"SAI", 63:"RUS", 77:"BOT", 81:"PIA"})

##Effects: Tanks in an array of 5 previous scoring positons, and returns an interger with the predicted postions
def predict_next_position(past_positions,past_grid,model):
    label_encoder = getLabelEncoder()
    device = getDevice()
    # Ensure past_positions length is 5 (t-1 to t-5)
    if len(past_positions) != 5  or len(past_grid) != 5:
        raise ValueError("Input past_positions must have exactly 10 elements.")

    # Encode the past positions
    encoded_positions = label_encoder.transform(past_positions)
    encoded_grid = label_encoder.transform(past_grid)

    # Convert to PyTorch tensor and reshape
    input_tensor_pos = torch.tensor(encoded_positions, dtype=torch.long).unsqueeze(0).to(device)
    input_tensor_grid = torch.tensor(encoded_grid, dtype=torch.long).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        model.eval()
        output = model(input_tensor_pos, input_tensor_grid)
        predicted_index = output.argmax(dim=1).cpu().numpy()[0]

    # Decode the predicted index
    predicted_position = label_encoder.inverse_transform([predicted_index])[0]

    return predicted_position


def encodeResults(year,round):
    for key in driverNumberToPostion.keys():
        past_positions = getPastPostion(key,year,round)
        past_qualifying = getPastQualifying(key,year,round)

        model = createModel(key)
        predicted_position = predict_next_position(past_positions, past_qualifying, model)
        driverNumberToPostion[key] = predicted_position

    for key, value in driverNumberToPostion.items():
        print("Predicted Position for " + driverNumberToName[key] + ":" + str(value))

        

encodeResults(2024,9)

