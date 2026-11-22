import numpy as np
import pandas as pd
from beams import ss_center
import torch
import torch.nn as nn

rng = np.random.default_rng(42)
np_beams = 500
np_points = 21

data = []

for beam_id in range(np_beams):
    L = rng.uniform(2, 5)
    E = rng.uniform(100e9, 210e9)
    I = rng.uniform(1e-6, 10e-6)
    P = rng.uniform(500, 5000)

    x_values = np.linspace(0, L, np_points)
    y_values = [ss_center(x, L, P, E, I) for x in x_values]

    for x, y in zip(x_values, y_values):
        data.append({
            "beam_id": beam_id,
            "x": x,
            "y": y,
            "L": L,
            "E": E,
            "I": I,
            "P": P
        })

df = pd.DataFrame(data)
df.to_csv("beam_data.csv", index=False)
print(df.head())
print("Dataset shape:", df.shape)

# Splitting dataset
# beam_ids = df['beam_id'].unique()
# rng.shuffle(beam_ids)
# train_ids = beam_ids[:350]
# val_ids = beam_ids[350:425]
# test_ids = beam_ids[425:500]

# train_df = df[df['beam_id'].isin(train_ids)].copy()
# val_df = df[df['beam_id'].isin(val_ids)].copy()
# test_df = df[df['beam_id'].isin(test_ids)].copy()
# print(len(train_df), len(val_df), len(test_df))

# # Nomalizing process
# input_columns = ['x', 'L', 'E', 'I', 'P']
# output_columns = ['y']
# X_train = train_df[input_columns].to_numpy()
# y_train = train_df[output_columns].to_numpy()
# X_mean = X_train.mean(axis=0)
# X_std = X_train.std(axis=0)
# y_mean = y_train.mean(axis=0)
# y_std = y_train.std(axis=0)

# def normalize_X(data):
#     return (data - X_mean) / X_std
# def normalize_y(data):
#     return (data - y_mean) / y_std

# X_train = normalize_X(train_df[input_columns].to_numpy())
# X_val = normalize_X(val_df[input_columns].to_numpy())
# X_test = normalize_X(test_df[input_columns].to_numpy())
# y_train = normalize_y(train_df[output_columns].to_numpy())
# y_val = normalize_y(val_df[output_columns].to_numpy())
# y_test = normalize_y(test_df[output_columns].to_numpy())

# class BeamMLP(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.network = nn.Sequential(
#             nn.Linear(5, 64),
#             nn.ReLU(),
#             nn.Linear(64, 64),
#             nn.ReLU(),
#             nn.Linear(64, 1)
#         )
#     def forward(self, x):
#         return self.network(x)

# X_train_tensor = torch.tensor(
#     X_train,
#     dtype=torch.float32
# )
# y_train_tensor = torch.tensor(
#     y_train,
#     dtype=torch.float32
# )
# print(X_train_tensor.shape, y_train_tensor.shape)

# # Dataset creation
# from torch.utils.data import TensorDataset

# train_dataset = TensorDataset(
#     X_train_tensor,
#     y_train_tensor
# )

# # Data load
# from torch.utils.data import DataLoader

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=64,
#     shuffle=True
# )


