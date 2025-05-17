import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
import csv
from models import *
from utils import *

####################################################################################
data_path = "Dataset/"
model_path = "Model/"
result_path = "Results/"

data_list = os.listdir(data_path)

rul_factor = 600  # Normalization factor for RUL
####################################################################################
all_results_lstm = []

csv_filename_lstm = os.path.join(result_path, "results_lstm.csv")
####################################################################################


def run_experiment_lstm(iteration, num_selected, seed):
    ####################
    epochs = 1000
    seq_length = 10
    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=seed)

    loss_train_list = []
    ####################
    print(f"[LSTM] Experiment {iteration} started.")

    results_filename = os.path.join(result_path, f"results_lstm.txt")

    iteration_path = os.path.join(result_path, f"LSTM_iter_{iteration}") # Create a directory for the current iteration
    os.makedirs(iteration_path, exist_ok=True)

    rmse_scores = []
    r2_scores = []
    rmse_dict = {}  # To store RMSE for each data_name
    r2_dict = {}    # To store R2 for each data_name

    # train
    model = RNN_LSTM(input_dim=len(num_selected), hidden_dim=10, output_dim=1, layers=3).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    for epoch in range(epochs):
        random.shuffle(train_list)
        epoch_loss = 0.0
        num_data = len(train_list)
        for data_name in train_list:
            data = torch.load(data_path + data_name)

            x = data[:, num_selected]
            y = data[:, -1][:, None] / rul_factor  # RUL

            train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)

            loss = model.loss(train_x, train_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().cpu().numpy()

        scheduler.step()

        if epoch % 10 == 9:  # Evaluating the test set for every 10 epochs
            rmse_scores_train_tmp = []

            ############## Train Files ##############
            for data_name in train_list:
                data = torch.load(data_path + data_name)

                x = data[:, num_selected]
                y = data[:, -1][:, None]  # RUL

                train_x, train_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():

                    y_ = model(train_x) * rul_factor

                    y_true_tmp.append(train_y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(
                        mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))
                    rmse_scores_train_tmp.append(rmse_loss_tmp)

            average_rmse_train_tmp = np.mean(rmse_scores_train_tmp)

            loss_train_list.append(average_rmse_train_tmp)

    model_name = f"lstm_{iteration}"
    model_save_path = model_path + f"{iteration}/"

    try:
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
    except OSError:
        print('Error: Creating directory. ' + model_save_path)

    torch.save(model, model_save_path + model_name)

    # Evaluating the model
    model = torch.load(model_path + model_name)

    for file_name in test_list:
        test_data = torch.load(data_path + file_name)

        x = test_data[:, num_selected]
        y = test_data[:, -1][:, None]  # RUL

        test_x, test_y = build_dataset(x.detach().numpy(), y.detach().numpy(), seq_length)
        y_true, y_pred = [], []
        model.eval()

        with torch.no_grad():

            y_ = model(test_x) * rul_factor

            y_true.append(test_y)
            y_pred.append(y_)
            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            rmse_loss = np.sqrt(mean_squared_error(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy()))
            r2 = r2_score(y_true.cpu().detach().numpy(), y_pred.cpu().detach().numpy())

            rmse_scores.append(rmse_loss)
            r2_scores.append(r2)
            rmse_dict[file_name] = rmse_loss  # Store RMSE with its corresponding data_name
            r2_dict[file_name] = r2

        with open(results_filename, "a") as file:
            file.write(f"Iteration {iteration} // {file_name}\n")
            file.write(f"RMSE: {rmse_loss}\n")
            file.write(f"R2: {r2}\n\n")

    average_rmse = np.mean(rmse_scores)
    print(average_rmse)
    std_dev_rmse = np.std(rmse_scores)
    average_r2 = np.mean(r2_scores)
    std_dev_r2 = np.std(r2_scores)
    max_rmse_data_name = max(rmse_dict, key=rmse_dict.get)
    min_r2_data_name = min(r2_dict, key=r2_dict.get)

    with open(results_filename, "a") as file:
        file.write(f"Iteration {iteration}, Selected Features: {num_selected}, Split_Seed: {seed},"
                   f" Epochs: {epochs}, Sequence Length: {seq_length}\n")
        file.write(f"Average RMSE: {average_rmse}, Std Dev: {std_dev_rmse}\n")
        file.write(f"Average R2: {average_r2}, Std Dev: {std_dev_r2}\n")
        file.write(f"Data Name with Highest RMSE: {max_rmse_data_name}, RMSE: {rmse_dict[max_rmse_data_name]}\n")
        file.write(f"Data Name with Smallest R2: {min_r2_data_name}, R2: {r2_dict[min_r2_data_name]}\n\n")
        file.write("=============================================================================\n\n")

    all_results_lstm.append({
        "Feature Index": num_selected,
        "Iteration": iteration,
        "Average RMSE": average_rmse,
        "RMSE Std Dev": std_dev_rmse,
        "Average R2": average_r2,
        "R2 Std Dev": std_dev_r2,
        "Max RMSE Data Name": max_rmse_data_name,
        "Max RMSE Value": rmse_dict[max_rmse_data_name],
        "Min R2 Data Name": min_r2_data_name,
        "Min R2 Value": r2_dict[min_r2_data_name]
    })
    print(f"[LSTM] Experiment {iteration} completed and results saved.")
    print("")


for i in range(100):
    seed = 17 * i
    num_selected = [2, 1, 146, 57, 141, 68, 25, 99, 95, 87, 78, 108, 152]  # 13 HIs

    run_experiment_lstm(i, num_selected, seed)


with open(csv_filename_lstm, 'w', newline='') as csvfile:
    fieldnames = list(all_results_lstm[0].keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for result in all_results_lstm:
        writer.writerow(result)


print("All results saved to CSV.")
