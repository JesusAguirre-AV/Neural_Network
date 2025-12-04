"""For the multilayer perceptron portion of Project 3, this will implement pytorch in a manner that makes it compatible
with the data being implemented in this project"""
from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"

#Model definiton
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 dropout_rate: float = 0.3):
        super().__init__()
        #layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        #layer 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        #layer 3
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.cost = nn.CrossEntropyLoss() #loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    # Here we will take the training values and their data that will be extracted from, should be similar to other projects

    """""
    # Create PyTorch datasets and dataloaders
    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    # Initialize MLP model
    input_size = X_train.shape[1]
    hidden_size = 64
    output_size = len(np.unique(y_train))  # Number of classes

    model = MLP(input_size, hidden_size, output_size)
    print(model)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

    # Training loop
    num_epochs = 10
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.cost(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_accuracy = correct_train / total_train
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / len(train_loader):.4f}, Train Accuracy: {100 * train_accuracy:.2f}%")

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = model.cost(outputs, targets)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()
    """
def load_feature_data(processed_dir: Path):
    #Load processed CSVs and label mapping created by database.py
    train_csv = processed_dir / "train_features.csv"
    test_csv = processed_dir / "test_features.csv"
    label_json = processed_dir / "label_map.json"

    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    #load label mapping
    with open(label_json, "r", encoding="utf-8") as f:
        lm = json.load(f)
    cls2id_raw = lm["class_to_id"]
    id2cls_raw = lm["id_to_class"]

    #ensure proper int typing
    class_to_id = {k: int(v) if isinstance(v, str) and v.isdigit() else v
                   for k, v in cls2id_raw.items()}
    id_to_class = {int(k): v for k, v in id2cls_raw.items()}

    #map class labels to integer indices
    df_tr["label_idx"] = df_tr["label"].map(class_to_id)

    #feature column
    feature_cols = [c for c in df_tr.columns if c.startswith("f")]

    #structured features and labels
    X = df_tr[feature_cols].values.astype(np.float32)
    y = df_tr["label_idx"].values.astype(np.int64)
    X_test = df_te[feature_cols].values.astype(np.float32)

    return X, y, X_test, df_te, feature_cols, class_to_id, id_to_class


def make_dataloaders(X: np.ndarray, y: np.ndarray,
                     batch_size: int = 64,
                     val_size: float = 0.2,
                     random_state: int = 42):
    """"Split structured data into train/validation sets and return PyTorch dataloaders"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )

    #standardization for MLP training
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    #convert to PyTorch tensors
    X_train_t = torch.from_numpy(X_train_scaled).float()
    y_train_t = torch.from_numpy(y_train).long()
    X_val_t = torch.from_numpy(X_val_scaled).float()
    y_val_t = torch.from_numpy(y_val).long()

    #datasets and loaders
    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, scaler


def train_mlp(X: np.ndarray, y: np.ndarray, num_classes: int,
              hidden_size: int = 256, batch_size: int = 64,
              num_epochs: int = 30, lr: float = 1e-3,
              weight_decay: float = 1e-4):
    #Train a 2-layer MLP on structured feature matrices
    device = torch.device("cpu")

    train_loader, val_loader, scaler = make_dataloaders(
        X, y, batch_size=batch_size
    )

    input_size = X.shape[1]  #number of structured features
    model = MLP(input_size, hidden_size, num_classes).to(device)

    #adam optimizer with regularization
    optimizer = optim.Adam(model.parameters(),
                           lr=lr,
                           weight_decay=weight_decay)

    for epoch in range(num_epochs):
        #training loop
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.cost(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        train_loss /= len(train_loader)
        train_acc = correct_train / total_train

        #validation loop
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = model.cost(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, dim=1)
                total_val += targets.size(0)
                correct_val += (predicted == targets).sum().item()

        val_loss /= len(val_loader)
        val_acc = correct_val / total_val

        print(
            f"Epoch {epoch+1:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} Acc: {100*train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {100*val_acc:.2f}%"
        )

    return model, scaler


def predict_on_test(model: nn.Module,
                    scaler: StandardScaler,
                    X_test: np.ndarray,
                    id_to_class: dict[int, str]):
    #Run trained MLP on test features and map predicted indices to class names
    device = next(model.parameters()).device

    #apply same standardization as training
    X_test_scaled = scaler.transform(X_test)
    X_test_t = torch.from_numpy(X_test_scaled).float().to(device)

    model.eval()
    preds = []

    with torch.no_grad():
        test_loader = DataLoader(X_test_t, batch_size=64, shuffle=False)
        for batch in test_loader:
            outputs = model(batch)
            _, predicted = torch.max(outputs, dim=1)
            preds.extend(predicted.cpu().numpy().tolist())

    pred_labels = [id_to_class[int(i)] for i in preds]
    return preds, pred_labels


def main():
    print("Loading feature data from:", PROC)
    X, y, X_test, df_te, feature_cols, class_to_id, id_to_class = load_feature_data(PROC)
    num_classes = len(class_to_id)

    print(f"Training samples: {X.shape[0]}, features per sample: {X.shape[1]}")
    print(f"Number of classes: {num_classes}")

    model, scaler = train_mlp(
        X, y, num_classes=num_classes,
        hidden_size=256, batch_size=64,
        num_epochs=30, lr=1e-3, weight_decay=1e-4
    )

    print("Running model on test set and writing submission_mlp.csv ...")

    _, pred_labels = predict_on_test(model, scaler, X_test, id_to_class)

    #convert full test path to filename only
    file_ids = df_te["path"].apply(lambda p: Path(p).name)

    submission = pd.DataFrame({
        "id": file_ids,
        "label": pred_labels
    })

    out_csv = ROOT / "submission_mlp.csv"
    submission.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)


if __name__ == "__main__":
    main()