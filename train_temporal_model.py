import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_dataset(path: Path) -> tuple[list[dict[str, str]], list[str], np.ndarray]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        meta_fields = {"label", "sequence_id", "frame_id", "timestamp_sec"}
        feature_names = [name for name in fieldnames if name not in meta_fields]
        rows_meta: list[dict[str, str]] = []
        rows: list[list[float]] = []
        for row in reader:
            rows_meta.append(
                {
                    "label": row["label"],
                    "sequence_id": row.get("sequence_id", ""),
                    "frame_id": row.get("frame_id", ""),
                    "timestamp_sec": row.get("timestamp_sec", ""),
                }
            )
            rows.append([float(row[name]) for name in feature_names])
    return rows_meta, feature_names, np.asarray(rows, dtype=np.float32)


def select_features(
    feature_names: list[str],
    features: np.ndarray,
    feature_set: str,
) -> tuple[list[str], np.ndarray]:
    if feature_set == "all":
        indices = list(range(len(feature_names)))
    elif feature_set == "optimized":
        indices = [i for i, name in enumerate(feature_names) if name.startswith("optimized_")]
    else:
        prefix = f"{feature_set}_"
        indices = [i for i, name in enumerate(feature_names) if name.startswith(prefix)]

        # Allow raw-only CSVs that already contain only frame features without prefixes.
        if not indices and feature_set == "raw":
            indices = list(range(len(feature_names)))

    selected_names = [feature_names[i] for i in indices]
    selected_features = features[:, indices]
    if selected_features.shape[1] == 0:
        raise SystemExit(f"No columns found for feature set `{feature_set}` in dataset.")
    return selected_names, selected_features


def resample_sequence(sequence_array: np.ndarray, target_len: int) -> np.ndarray:
    original_len, feature_dim = sequence_array.shape
    if original_len == target_len:
        return sequence_array.astype(np.float32)

    old_time = np.linspace(0.0, 1.0, original_len)
    new_time = np.linspace(0.0, 1.0, target_len)
    resampled = np.zeros((target_len, feature_dim), dtype=np.float32)
    for index in range(feature_dim):
        resampled[:, index] = np.interp(new_time, old_time, sequence_array[:, index])
    return resampled


def build_sequence_dataset(
    row_meta: list[dict[str, str]],
    feature_names: list[str],
    features: np.ndarray,
    target_len: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    grouped: dict[str, list[tuple[int, float, np.ndarray]]] = {}
    sequence_labels: dict[str, str] = {}

    for index, (meta, feature_row) in enumerate(zip(row_meta, features, strict=True)):
        sequence_id = meta.get("sequence_id") or f"single_{index}_{meta['label']}"
        frame_id = int(meta.get("frame_id") or 0)
        timestamp_sec = float(meta.get("timestamp_sec") or 0.0)
        grouped.setdefault(sequence_id, []).append((frame_id, timestamp_sec, feature_row))
        sequence_labels[sequence_id] = meta["label"]

    sequence_ids: list[str] = []
    labels: list[str] = []
    sequences: list[np.ndarray] = []
    for sequence_id in sorted(grouped):
        entries = sorted(grouped[sequence_id], key=lambda item: (item[0], item[1]))
        sequence = np.stack([item[2] for item in entries], axis=0).astype(np.float32)
        sequences.append(resample_sequence(sequence, target_len))
        labels.append(sequence_labels[sequence_id])
        sequence_ids.append(sequence_id)

    x = np.stack(sequences, axis=0)
    y = np.asarray(labels)
    groups = np.asarray(sequence_ids)
    return x, y, groups, feature_names


class GestureSequenceDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]


class CNN1DGestureClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.feature_extractor(x)
        x = x.squeeze(-1)
        return self.classifier(x)


class GRUGestureClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_dim, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(x)
        return self.classifier(hidden[-1])


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * batch_x.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy().tolist())
        all_labels.extend(batch_y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, float(acc)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            total_loss += float(loss.item()) * batch_x.size(0)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.detach().cpu().numpy().tolist())
            all_labels.extend(batch_y.detach().cpu().numpy().tolist())

    avg_loss = total_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, float(acc), np.asarray(all_preds), np.asarray(all_labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "cohen_kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a temporal CNN/GRU gesture classifier from sequence CSV data.")
    parser.add_argument("--dataset", default="gesture_sequence_dataset_optimized_v2.csv")
    parser.add_argument(
        "--feature-set",
        choices=[
            "raw",
            "geom",
            "corrected",
            "optimized",
            "optimized_action",
            "optimized_sparse",
            "optimized_full",
            "optimized_loss",
            "all",
        ],
        default="raw",
    )
    parser.add_argument("--target-len", type=int, default=128, help="Resample all sequences to this number of frames.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--model-type", choices=["cnn", "gru"], default="cnn")
    parser.add_argument("--gru-hidden-dim", type=int, default=64)
    parser.add_argument("--gru-num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--save-path", default="best_gesture_sequence_model.pth")
    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"CSV file not found: {dataset_path}")

    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    row_meta, all_feature_names, all_features = load_dataset(dataset_path)
    feature_names, selected_features = select_features(all_feature_names, all_features, args.feature_set)

    x, y, groups, feature_cols = build_sequence_dataset(
        row_meta=row_meta,
        feature_names=feature_names,
        features=selected_features,
        target_len=args.target_len,
    )

    print("Feature set:", args.feature_set)
    print("Sequence-level X shape:", x.shape)
    print("Sequence-level y shape:", y.shape)
    print("Number of sequences:", len(groups))
    print("Original labels:", sorted(set(y.tolist())))

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(label_encoder.classes_)
    input_dim = x.shape[2]

    print("Label mapping:")
    for original_label, encoded_label in zip(label_encoder.classes_, range(num_classes), strict=True):
        print(f"  original label {original_label} -> encoded label {encoded_label}")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=args.test_size,
        random_state=args.random_seed,
    )
    train_idx, test_idx = next(splitter.split(x, y_encoded, groups=groups))

    x_train = x[train_idx]
    x_test = x[test_idx]
    y_train = y_encoded[train_idx]
    y_test = y_encoded[test_idx]

    print("Train X shape:", x_train.shape)
    print("Test X shape:", x_test.shape)
    print("Train labels:", np.unique(y_train, return_counts=True))
    print("Test labels:", np.unique(y_test, return_counts=True))

    scaler = StandardScaler()
    num_train, time_len, feature_dim = x_train.shape
    num_test = x_test.shape[0]
    x_train_scaled = scaler.fit_transform(x_train.reshape(-1, feature_dim))
    x_test_scaled = scaler.transform(x_test.reshape(-1, feature_dim))
    x_train = x_train_scaled.reshape(num_train, time_len, feature_dim)
    x_test = x_test_scaled.reshape(num_test, time_len, feature_dim)

    train_loader = DataLoader(
        GestureSequenceDataset(x_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        GestureSequenceDataset(x_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
    )

    if args.model_type == "cnn":
        model = CNN1DGestureClassifier(input_dim=input_dim, num_classes=num_classes, dropout=args.dropout)
    else:
        model = GRUGestureClassifier(
            input_dim=input_dim,
            hidden_dim=args.gru_hidden_dim,
            num_classes=num_classes,
            num_layers=args.gru_num_layers,
            dropout=args.dropout,
        )

    model = model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_test_acc = -1.0
    best_model_path = Path(args.save_path).resolve()

    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label_classes": label_encoder.classes_,
                    "feature_cols": feature_cols,
                    "feature_set": args.feature_set,
                    "target_len": args.target_len,
                    "model_type": args.model_type,
                    "input_dim": input_dim,
                    "num_classes": num_classes,
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_,
                },
                best_model_path,
            )

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"Epoch [{epoch:03d}/{args.num_epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_acc:.4f} "
                f"Test Loss: {test_loss:.4f} "
                f"Test Acc: {test_acc:.4f}"
            )

    print("\nBest test accuracy:", best_test_acc)

    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    original_test_preds = label_encoder.inverse_transform(test_preds)
    original_test_labels = label_encoder.inverse_transform(test_labels)
    metrics = compute_metrics(original_test_labels, original_test_preds)

    print("\nFinal test accuracy:", test_acc)
    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"  {name}={value:.4f}")

    print("\nClassification report:")
    print(classification_report(original_test_labels, original_test_preds, zero_division=0))

    print("\nConfusion matrix:")
    print(confusion_matrix(original_test_labels, original_test_preds))

    print("\nTrue labels:")
    print(original_test_labels)

    print("\nPredicted labels:")
    print(original_test_preds)

    print(f"\nBest model saved to: {best_model_path}")


if __name__ == "__main__":
    main()
