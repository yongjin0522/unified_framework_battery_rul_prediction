import numpy as np
import torch
import random

device = "cuda" if torch.cuda.is_available() else "cpu"


def build_dataset(data_x, data_y, seq_length):
    data_out_x = []
    data_out_y = []

    for i in range(0, len(data_x) - seq_length):
        _x = data_x[i:i + seq_length, :]
        data_out_x.append(_x)

    for i in range(0, len(data_y) - seq_length):
        _y = data_y[i+seq_length, :]
        data_out_y.append(_y)

    data_out_x = np.array(data_out_x)
    data_out_x = torch.FloatTensor(data_out_x)
    data_out_x = data_out_x.to(device)

    data_out_y = np.array(data_out_y)
    data_out_y = torch.FloatTensor(data_out_y)
    data_out_y = data_out_y.to(device)

    return data_out_x, data_out_y


def generate_randomness(max_num_features, num_features):
    seed = random.randint(1, 1000)
    num_selected = random.sample(range(0, max_num_features), num_features)

    return num_selected, seed

