import numpy as np
import torch
import random
import os

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


def load_all_datasets(data_list, data_path, num_selected, seq_length, rul_factor):
    loaded_data_list = []

    for data_name in data_list:
        data = torch.load(os.path.join(data_path, data_name))

        x = data[:, num_selected]
        y = data[:, -1][:, None] / rul_factor

        train_x_np, train_y_np = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)

        train_x = torch.FloatTensor(train_x_np).to(device)
        train_y = torch.FloatTensor(train_y_np).to(device)

        loaded_data_list.append((data_name, train_x, train_y))

    print("Data loading complete.")

    return loaded_data_list
