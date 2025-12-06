"""Transfer Learning portion of the assignment, for this we will use pretrained models that may be compared to our
hand trained models as well as work on fine-tuning."""
import torch
import torchvision.models as models
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from torchvision import transforms
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
PROC = ROOT / "data" / "processed"

class ResNetProject(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()

        #Loading the pretrained ResNet, our model of choice
        self.resnet = models.resnet50(pretrained=True)

        #Here we can freeze certain layers, currently set to halt training
        """for param in self.resnet.parameters():
            param.requires_grad = False"""
        for param in self.resnet.conv1.parameters():
            param.requires_grad = False
        for param in self.resnet.bn1.parameters():
            param.requires_grad = False
        #Here there are individual layers that we can fine tune such that they can be experimented with by switching
        #them to True or False
        for param in self.resnet.layer1.parameters():
            param.requires_grad = False
        for param in self.resnet.layer2.parameters():
            param.requires_grad = False
        for param in self.resnet.layer3.parameters():
            param.requires_grad = False
        for param in self.resnet.layer4.parameters():
            param.requires_grad = False

        #Here we set the identity to use global pooled output
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()

        #Map input features to ResNet feature dimension
        self.structure = nn.Linear(input_size, in_features)
        #Final classification layer
        self.classifier = nn.Linear(in_features, num_classes)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, x):
        #Map structured to feature space
        proj = self.structure(x)
        extracted = self.resnet(proj)
        return self.classifier(extracted)

def load_feature_data(processed_dir: Path):
    # Load processed CSVs and label mapping created by database.py
    train_csv = processed_dir / "train_features.csv"
    test_csv = processed_dir / "test_features.csv"
    label_json = processed_dir / "label_map.json"

    df_tr = pd.read_csv(train_csv)
    df_te = pd.read_csv(test_csv)

    # load label mapping
    with open(label_json, "r", encoding="utf-8") as f:
        lm = json.load(f)
    cls2id_raw = lm["class_to_id"]
    id2cls_raw = lm["id_to_class"]

    # ensure proper int typing
    class_to_id = {k: int(v) if isinstance(v, str) and v.isdigit() else v
                   for k, v in cls2id_raw.items()}
    id_to_class = {int(k): v for k, v in id2cls_raw.items()}

    # map class labels to integer indices
    df_tr["label_idx"] = df_tr["label"].map(class_to_id)

    # feature column
    feature_cols = [c for c in df_tr.columns if c.startswith("f")]

    # structured features and labels
    X = df_tr[feature_cols].values.astype(np.float32)
    y = df_tr["label_idx"].values.astype(np.int64)
    X_test = df_te[feature_cols].values.astype(np.float32)

    return X, y, X_test, df_te, feature_cols, class_to_id, id_to_class

"""Because this is a pretrained model, this will be significantly shorter tha that of other, similar code"""
def load_pretrained_resnest(input_size: int, num_classes: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNetProject(input_size, num_classes).to(device)
    print("Loading pretrained model ResNest")
    return model

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
    print("Running Transfer Learning.")
    X, y, X_test, df_te, feature_cols, class_to_id, id_to_class = load_feature_data(PROC)
    num_classes = len(id_to_class)

    print(f"Training samples: {X.shape[0]}, features per sample: {X.shape[1]}")
    print(f"Number of classes: {num_classes}")

    #Key difference with pretrained models is scaling and model only require to be given parameters
    scaler = StandardScaler()
    scaler.fit(X)

    input_Size = X.shape[1]
    model = load_pretrained_resnest(input_Size, num_classes)

    print("Running model on test set and writing submission_mlp.csv ...")

    _, pred_labels = predict_on_test(model, scaler, X_test, id_to_class)

    # convert full test path to filename only
    file_ids = df_te["path"].apply(lambda p: Path(p).name)

    submission = pd.DataFrame({
        "id": file_ids,
        "label": pred_labels
    })

    out_csv = ROOT / "submission_TrL.csv"
    submission.to_csv(out_csv, index=False)
    print("Wrote:", out_csv)

