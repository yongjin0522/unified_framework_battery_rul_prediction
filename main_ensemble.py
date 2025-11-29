import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os
import csv
from utils import *
from models import *

####################################################################################
data_path = "Dataset/"
result_path = "Results/"

data_list = os.listdir(data_path)

csv_filename_ensemble = os.path.join(result_path, "results_ensemble.csv")

rul_factor = 600  # Normalization factor for RUL

num_selected = [2, 1, 146, 57, 141, 68, 25, 99, 95, 87, 78, 108, 152]

all_results_ensemble = []
results_filename = os.path.join(result_path, f"results_ensemble.txt")
####################################################################################


for iteration in range(100):
    seed = 17 * iteration

    model_path = f"Model/{iteration}/"

    # Initialize the weights
    weights = torch.tensor([0.25, 0.25, 0.25, 0.25], requires_grad=True)
    optimizer = optim.Adam([weights], lr=1e-2)
    num_epochs = 50

    rmse_scores = []
    r2_scores = []
    rmse_dict = {}  # To store RMSE for each data_name
    r2_dict = {}  # To store R2 for each data_name

    seq_length = 10
    train_list, test_list = train_test_split(data_list, test_size=0.2, random_state=seed)

    ####################
    loaded_train_data = load_all_datasets(train_list, data_path, num_selected, seq_length, rul_factor)
    loaded_test_data = load_all_datasets(test_list, data_path, num_selected, seq_length, rul_factor)
    ####################

    model_mlp = torch.load(model_path + f"mlp_{iteration}")
    model_gru = torch.load(model_path + f"gru_{iteration}")
    model_lstm = torch.load(model_path + f"lstm_{iteration}")
    model_transformer = torch.load(model_path + f"Transformer_{iteration}")

    for epoch in range(num_epochs):

        for data_name, train_x, train_y in loaded_train_data:

            y_true, y_pred = [], []

            y_mlp = model_mlp(train_x)
            y_gru = model_gru(train_x)
            y_lstm = model_lstm(train_x)
            y_transformer = model_transformer(train_x)

            y_ = weights[0] * y_mlp + weights[1] * y_gru + weights[2] * y_lstm + weights[3] * y_transformer

            y_true.append(train_y)
            y_pred.append(y_)
            y_true = torch.cat(y_true, axis=0)
            y_pred = torch.cat(y_pred, axis=0)

            loss = torch.nn.MSELoss()(y_true, y_pred)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                weights.data /= weights.data.sum()

        if epoch % 10 == 9:  # Evaluating the test set for every 10 epochs
            rmse_scores_test_tmp = []
            r2_scores_test_tmp = []

            # Intermediate Test
            for file_name, test_x, test_y in loaded_test_data:

                y_true_tmp, y_pred_tmp = [], []

                with torch.no_grad():

                    y_mlp = model_mlp(test_x) * rul_factor
                    y_gru = model_gru(test_x) * rul_factor
                    y_lstm = model_lstm(test_x) * rul_factor
                    y_transformer = model_transformer(test_x) * rul_factor

                    y_ = weights[0] * y_mlp + weights[1] * y_gru + weights[2] * y_lstm + weights[3] * y_transformer

                    y_true_tmp.append(test_y)
                    y_pred_tmp.append(y_)
                    y_true_tmp = torch.cat(y_true_tmp, axis=0)
                    y_pred_tmp = torch.cat(y_pred_tmp, axis=0)

                    rmse_loss_tmp = np.sqrt(
                        mean_squared_error(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy()))

                    r2_tmp = r2_score(y_true_tmp.cpu().detach().numpy(), y_pred_tmp.cpu().detach().numpy())

                    if not np.isnan(rmse_loss_tmp) and not np.isnan(r2_tmp):
                        rmse_scores_test_tmp.append(rmse_loss_tmp)
                        r2_scores_test_tmp.append(r2_tmp)

                if rmse_scores_test_tmp and r2_scores_test_tmp:
                    average_rmse_test_tmp = np.mean(rmse_scores_test_tmp)
                    average_r2_test_tmp = np.mean(r2_scores_test_tmp)
                else:
                    average_rmse_test_tmp = np.nan
                    average_r2_test_tmp = np.nan

            print(f"Iteration: {epoch}, Test RMSE = {average_rmse_test_tmp}, Test R2 = {average_r2_test_tmp}, "
                f"Weights = {weights.detach().numpy()}")

    ##### Comparing each models with the ensembled result using the train data #####
    loss_mlp = 0
    loss_gru = 0
    loss_lstm = 0
    loss_transformer = 0
    loss_ensemble = 0

    for data_name, train_x, train_y in loaded_train_data:

        y_true, y_pred_ensemble, y_pred_mlp, y_pred_gru, y_pred_lstm, y_pred_transformer = [], [], [], [], [], []

        with torch.no_grad():
            y_mlp = model_mlp(train_x)
            y_gru = model_gru(train_x)
            y_lstm = model_lstm(train_x)
            y_transformer = model_transformer(train_x)

        y_ensemble = weights[0] * y_mlp + weights[1] * y_gru + weights[2] * y_lstm + weights[3] * y_transformer

        y_true.append(train_y)
        y_true = torch.cat(y_true, axis=0)

        y_pred_ensemble.append(y_ensemble)
        y_pred_ensemble = torch.cat(y_pred_ensemble, axis=0)

        y_pred_mlp.append(y_mlp)
        y_pred_mlp = torch.cat(y_pred_mlp, axis=0)

        y_pred_gru.append(y_gru)
        y_pred_gru = torch.cat(y_pred_gru, axis=0)

        y_pred_lstm.append(y_lstm)
        y_pred_lstm = torch.cat(y_pred_lstm, axis=0)

        y_pred_transformer.append(y_transformer)
        y_pred_transformer = torch.cat(y_pred_transformer, axis=0)

        loss_ensemble += torch.nn.MSELoss()(y_true, y_pred_ensemble)
        loss_mlp += torch.nn.MSELoss()(y_true, y_pred_mlp)
        loss_gru += torch.nn.MSELoss()(y_true, y_pred_gru)
        loss_lstm += torch.nn.MSELoss()(y_true, y_pred_lstm)
        loss_transformer += torch.nn.MSELoss()(y_true, y_pred_transformer)
    
    if min(loss_mlp, loss_gru, loss_lstm, loss_transformer, loss_ensemble) == loss_mlp:
        weights = torch.FloatTensor([1, 0, 0, 0])
        print(f"MLP Loss({loss_mlp}) is the lowest")
    
    elif min(loss_mlp, loss_gru, loss_lstm, loss_transformer, loss_ensemble) == loss_gru:
        weights = torch.FloatTensor([0, 1, 0, 0])
        print(f"GRU Loss({loss_gru}) is the lowest")

    elif min(loss_mlp, loss_gru, loss_lstm, loss_transformer, loss_ensemble) == loss_lstm:
        weights = torch.FloatTensor([0, 0, 1, 0])
        print(f"LSTM Loss({loss_lstm}) is the lowest")

    elif min(loss_mlp, loss_gru, loss_lstm, loss_transformer, loss_ensemble) == loss_transformer:
        weights = torch.FloatTensor([0, 0, 0, 1])
        print(f"Transformer Loss({loss_transformer}) is the lowest")

    print("----")
    print(loss_mlp, loss_gru, loss_lstm, loss_transformer, loss_ensemble)
    print("----")

    #######################################################################################

    # Test the Ensemble
    for file_name, test_x, test_y in loaded_test_data:

        y_true, y_pred = [], []

        with torch.no_grad():

            y_mlp = model_mlp(test_x) * rul_factor
            y_gru = model_gru(test_x) * rul_factor
            y_lstm = model_lstm(test_x) * rul_factor
            y_transformer = model_transformer(test_x) * rul_factor

            y_ = weights[0] * y_mlp + weights[1] * y_gru + weights[2] * y_lstm + weights[3] * y_transformer

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
        file.write(f"Ensemble #{iteration}\n")
        file.write(f"Weights: {weights.detach().numpy()}\n")
        file.write(f"Average RMSE: {average_rmse}, Std Dev: {std_dev_rmse}\n")
        file.write(f"Average R2: {average_r2}, Std Dev: {std_dev_r2}\n")
        file.write(f"Data Name with Highest RMSE: {max_rmse_data_name}, RMSE: {rmse_dict[max_rmse_data_name]}\n")
        file.write(f"Data Name with Smallest R2: {min_r2_data_name}, R2: {r2_dict[min_r2_data_name]}\n\n")
        file.write("=============================================================================\n\n")

    all_results_ensemble.append({
        "Iteration": iteration,
        "Average RMSE": average_rmse,
        "RMSE Std Dev": std_dev_rmse,
        "Average R2": average_r2,
        "R2 Std Dev": std_dev_r2,
        "Max RMSE Data Name": max_rmse_data_name,
        "Max RMSE Value": rmse_dict[max_rmse_data_name],
        "Min R2 Data Name": min_r2_data_name,
        "Min R2 Value": r2_dict[min_r2_data_name],
        "Weights": weights.detach().numpy()
    })

    with open(csv_filename_ensemble, 'w', newline='') as csvfile:
        fieldnames = list(all_results_ensemble[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in all_results_ensemble:
            writer.writerow(result)

    print(f"[Ensemble] Experiment {iteration} completed and results saved.")
    print("")

