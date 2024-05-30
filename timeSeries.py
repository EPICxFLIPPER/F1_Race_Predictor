import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

device = 'cuda:0' if torch.cuda.is_available() else 'cup'
VERData = pd.read_pickle('Data/33.pkl')
VERData['date'] = pd.to_datetime(VERData['date'])
print(VERData)
plt.plot(VERData['date'],VERData['position'])
plt.savefig("Plots/test.png")
plt.close()