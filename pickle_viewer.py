#%%
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
device=torch.device('cuda:0')
data = pickle.load(open('testing_data.pkl', 'rb'))
print("testing",data.shape)
data1=pickle.load(open('training_data.pkl', 'rb'))
print("training",data1.shape)
# %%
training_dataset = torch.from_numpy(np.load('testing_data.pkl', allow_pickle=True))
data_loader = DataLoader(training_dataset, batch_size=512, shuffle=True)  # Reduced batch s
for batch in data_loader:
    ray_origins = batch[:, :3].to(device)
    ray_directions = batch[:, 3:6].to(device)
    ground_truth_px_values = batch[:, 6:].to(device)
# %%
ray_directions

# %%
ray_origins.shape
# %%
