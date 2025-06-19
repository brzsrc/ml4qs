import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split, GroupKFold
from webencodings import labels


def create_sequences(data: pd.DataFrame, labels: pd.DataFrame, seq_type: str,
                     group: str, time_window: int, stride = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    sequences = []
    targets = []

    for _, sdf in list(data.groupby(group)):
        for i in range(0, len(sdf) - time_window + 1, stride):
            sdf_time_wind = sdf.drop(columns=[group]).iloc[i:i + time_window].to_numpy()
            sequences.append(sdf_time_wind)

    for _, sdf in list(labels.groupby(group)):
        for i in range(0, len(sdf) - time_window + 1, stride):
            sdf_time_wind = sdf.drop(columns=[group]).iloc[i + time_window].to_numpy()
            targets.append(sdf_time_wind)

    sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    if seq_type == "classification":
        targets_tensor = torch.tensor(np.array(targets).flatten(), dtype=torch.long)
    else:
        targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)
    return sequences_tensor, targets_tensor


class LSTMClassifier(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, dropout = 0.2):
        super().__init__()
        self.output_size = output_size  # number of classes
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected
        self.fc_2 = nn.Linear(128, output_size)  # fully connected last layer
        self.relu = nn.ReLU()
        # self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))  # (input, hidden, and cell state)
        hn = hn[-1]  # last layer's hidden state
        out = self.relu(hn)
        out = self.dropout(out)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)
        # out = self.softmax(out)

        return out


class LSTMRegression(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers):
        super().__init__()
        self.output_size = output_size  # number of classes
        self.num_layers = num_layers  # number of recurrent layers in the lstm
        self.input_size = input_size  # input size
        self.hidden_size = hidden_size  # neurons in each lstm layer
        # LSTM model
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.02 if num_layers > 1 else 0)
        self.fc_1 = nn.Linear(hidden_size, 128)  # fully connected
        self.fc_2 = nn.Linear(128, output_size)  # fully connected last layer
        self.relu = nn.ReLU()

    def forward(self, x):
        # hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h0, c0))  # (input, hidden, and cell state)
        hn = hn[-1]  # last layer's hidden state
        out = self.relu(hn)
        out = self.fc_1(out)  # first dense
        out = self.relu(out)  # relu
        out = self.fc_2(out)
        return out

def evaluate_model_classification(model, test_loader, loss_func) -> (float, float):
    prediction = []
    y_labels = []
    model.eval()
    # Disable gradient calc
    with torch.no_grad():
        test_loss = 0.0
        # Compute classes and losses
        for inputs, label in test_loader:
            outputs = model(inputs)
            # print(outputs)
            _, predicted_labels = torch.max(outputs, 1)
            # print("predicted_labels: ", predicted_labels)
            # print(predicted_labels.shape)
            loss = loss_func(outputs, label)
            test_loss += loss.item() * inputs.size(0)
            prediction.extend(predicted_labels.numpy())
            y_labels.extend(label.numpy())
    test_loss /= len(test_loader.dataset)

    predictions = np.array(prediction)
    y_labels = np.array(y_labels)

    equal_labels = predictions == y_labels
    accuracy = np.sum(equal_labels) / len(equal_labels)
    return test_loss, accuracy

def train_model(n_epochs, model, optimiser, loss_func, train_loader, test_loader):
    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0
        # Training mode
        for inputs, label in train_loader:
            # Reset gradients
            optimiser.zero_grad()
            # Forward propagation
            outputs = model(inputs)
            # Training loss
            loss = loss_func(outputs, label)
            loss.backward()
            # Update weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimiser.step()

            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss /= len(train_loader.dataset)

        if epoch % 10 == 0:
            val_loss, val_accuracy = evaluate_model_classification(model, test_loader, loss_func)
            print(f"Epoch {epoch} - Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}")
    return model

# Interval time: 1800, 7200, 21600, 36000
# Learning rate 0.1, 0.05, 0.001, 0.0005
# Hidden_size 2, 4, 10, 15, 25, 35
# Num layers 1, 2, 4,8
def training_pipeline(train_loader: DataLoader, test_loader: DataLoader, model_type: str,
                      learning_rate: float, hidden_size: int, num_layers: int, n_epochs: int) -> float:
    input_size = 12  # number of features
    num_classes = 5  # number of output classes
    print(f"Testing learning rate: {learning_rate}, features in hidden layer {hidden_size}, stacked LSTM {num_layers}")

    if model_type == "regression":
        lstm_regression = LSTMRegression(num_classes, input_size, hidden_size, num_layers)
        optimiser = torch.optim.Adam(lstm_regression.parameters(), lr=learning_rate)
        loss_func = torch.nn.MSELoss()  # mean-squared error for regression
        trained_model_regression = train_model(n_epochs=n_epochs, model=lstm_regression,
                                               optimiser=optimiser,
                                               loss_func=loss_func, train_loader=train_loader,
                                               test_loader=test_loader)
        return 0
    else:
        lstm_classifier = LSTMClassifier(num_classes, input_size, hidden_size, num_layers, dropout=0.2)
        # optimiser = torch.optim.SGD(lstm_classifier.parameters(), lr=learning_rate, momentum=0.9)
        optimiser = torch.optim.Adam(lstm_classifier.parameters(), lr=learning_rate)
        loss_func = nn.CrossEntropyLoss()
        trained_model_classification = train_model(n_epochs=n_epochs, model=lstm_classifier, optimiser=optimiser,
                        loss_func=loss_func, train_loader=train_loader, test_loader=test_loader)
        val_loss, accuracy = evaluate_model_classification(trained_model_classification, test_loader, loss_func)
        return accuracy


def run_cv(title: str, x: pd.DataFrame, y: pd.DataFrame, qid: pd.Series,
           learning_rate: float, hidden_size: int, num_layers: int, n_epochs: int, bach_size: int):
    folds = GroupKFold(n_splits=3).split(X=x, y=y, groups=qid)
    print("=" * 20, title, "=" * 20)
    scores_v = []
    for i, (idx_t, idx_v) in enumerate(folds):
        x_t, y_t = [s.iloc[idx_t].reset_index(drop=True) for s in (x, y)]
        x_v, y_v = [s.iloc[idx_v].reset_index(drop=True) for s in (x, y)]

        train_sequences, train_targets = create_sequences(x_t, y_t, "classification", "sample_number", 100, 50)
        val_sequences, val_targets = create_sequences(x_v, y_v, "classification", "sample_number", 100, 50)

        train_dataset = TensorDataset(train_sequences, train_targets)
        val_dataset = TensorDataset(val_sequences, val_targets)

        train_loader = DataLoader(train_dataset, batch_size=bach_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=bach_size, shuffle=False)

        score_v = training_pipeline(train_loader, val_loader, "classification",
                          learning_rate, hidden_size, num_layers, n_epochs)

        scores_v.append(score_v)

    val = pd.Series(scores_v)
    print(f"val: mu={val.mean():.6f} min={val.min():.6f} max={val.max():.6f} std={val.std():.5f}")
    return val.max()


if __name__ == '__main__':
    CSV_FILE = 'datasets/lstm2.csv'
    df = pd.read_csv(CSV_FILE)
    mapping = {'clear': 0, 'drive': 1, 'drop': 2, 'lob': 3, 'smash': 4}
    df['skill_type'] = df['skill_type'].map(mapping)
    train_df = df[df['user'] != 'keeley']
    test_df = df[df['user'] == 'keeley']

    X_train = train_df.drop(columns=['Unnamed: 0', 'skill_type', 'user', "time"])
    y_train = train_df[['sample_number','skill_type']]

    X_test = test_df.drop(columns=['Unnamed: 0', 'skill_type', 'user', "time"])
    y_test = test_df[['sample_number','skill_type']]


    # Create sequences for training and testing datasets
    train_sequences, train_targets = create_sequences(X_train, y_train, "classification", "sample_number", 100, 50)
    test_sequences, test_targets = create_sequences(X_test, y_test, "classification", "sample_number", 100, 50)

    train_dataset = TensorDataset(train_sequences, train_targets)
    test_dataset = TensorDataset(test_sequences, test_targets)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # training_pipeline(train_loader, test_loader, "classification")

    run_cv("lstm", X_train, y_train, X_train["sample_number"],
           0.01, 25, 5, 100, 30)





