"""
Train the EEGNet-TCN decoder on the PhysioNet EEG Motor Movement/Imagery
dataset.

    python -m src.train_pytorch_real --dataset-root /path/to/files

The dataset path comes from the command line or from the BCI_DATASET_ROOT
environment variable. Nothing is hard-coded to one developer's machine.
"""

from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core import config                                    # noqa: E402
from src.core.model import EEGNetTCN, count_parameters, resolve_device  # noqa: E402
from src.core.preprocessing import (                            # noqa: E402
    find_subject_edf_files,
    load_edf_epochs,
    normalize_per_channel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s")
logger = logging.getLogger("train")


class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


def load_dataset(dataset_root: Path, subjects, runs) -> tuple[np.ndarray, np.ndarray]:
    epochs_all, labels_all, loaded = [], [], 0
    for subject in subjects:
        files = find_subject_edf_files(dataset_root, subject, runs)
        if not files:
            continue
        for edf_file in files:
            try:
                data, labels, _ = load_edf_epochs(edf_file)
            except Exception as exc:                     # noqa: BLE001
                logger.warning("Skipping %s: %s", edf_file.name, exc)
                continue
            if data.shape[0] == 0:
                continue
            epochs_all.append(data)
            labels_all.append(labels)
        loaded += 1
        logger.info("Subject %03d loaded", subject)

    if not epochs_all:
        raise SystemExit(
            f"No EEG data found under {dataset_root}. Expected files such as "
            f"{dataset_root}/S001/S001R04.edf"
        )

    X = np.concatenate(epochs_all, axis=0)
    y = np.concatenate(labels_all, axis=0)
    logger.info("Loaded %d epochs from %d subjects, shape %s", X.shape[0], loaded, X.shape)
    return X, y


def prepare(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = normalize_per_channel(X)[:, np.newaxis, :, :]
    distribution = {name: int((y == index).sum()) for index, name in enumerate(config.CLASS_NAMES)}
    logger.info("Class distribution: %s", distribution)
    return X, y


def run_epoch(model, loader, criterion, device, optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for data, targets in loader:
            data, targets = data.to(device), targets.to(device)
            if training:
                optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            correct += outputs.argmax(1).eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / max(1, len(loader)), 100.0 * correct / max(1, total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Adaptive Neuro-AI decoder")
    parser.add_argument("--dataset-root", default=str(config.DATASET_ROOT))
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--runs", default="imagery", choices=["imagery", "execution", "both"],
        help="PhysioNet run group: imagined movement (4/8/12), executed (3/7/11) or both.",
    )
    args = parser.parse_args()

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    config.ensure_directories()

    runs = {
        "imagery": config.MOTOR_IMAGERY_RUNS,
        "execution": config.MOTOR_EXECUTION_RUNS,
        "both": config.MOTOR_IMAGERY_RUNS + config.MOTOR_EXECUTION_RUNS,
    }[args.runs]

    dataset_root = Path(args.dataset_root)
    device = resolve_device(args.device)
    logger.info("Device: %s | runs: %s | dataset: %s", device, runs, dataset_root)

    X_train_full, y_train_full = load_dataset(dataset_root, config.TRAIN_SUBJECTS, runs)
    X_test, y_test = load_dataset(dataset_root, config.TEST_SUBJECTS, runs)

    X_train_full, y_train_full = prepare(X_train_full, y_train_full)
    X_test, y_test = prepare(X_test, y_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=config.VAL_SPLIT, random_state=config.RANDOM_SEED,
        stratify=y_train_full,
    )

    loaders = {
        "train": DataLoader(EEGDataset(X_train, y_train), batch_size=args.batch_size, shuffle=True),
        "val": DataLoader(EEGDataset(X_val, y_val), batch_size=args.batch_size),
        "test": DataLoader(EEGDataset(X_test, y_test), batch_size=args.batch_size),
    }

    model = EEGNetTCN(n_channels=X_train.shape[2], n_timesteps=X_train.shape[3]).to(device)
    logger.info("Model parameters: %s", f"{count_parameters(model):,}")

    # Class weights counter the heavy REST imbalance in the PhysioNet protocol.
    counts = np.bincount(y_train, minlength=config.N_CLASSES).astype(np.float64)
    weights = torch.tensor((counts.sum() / (config.N_CLASSES * np.maximum(counts, 1))),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate,
                           weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.5)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_accuracy = 0.0

    for epoch in range(args.epochs):
        train_loss, train_accuracy = run_epoch(model, loaders["train"], criterion, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, loaders["val"], criterion, device)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_accuracy)
        history["val_acc"].append(val_accuracy)

        logger.info(
            "Epoch %3d/%d  train %.4f/%.2f%%  val %.4f/%.2f%%",
            epoch + 1, args.epochs, train_loss, train_accuracy, val_loss, val_accuracy,
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_accuracy": val_accuracy,
                    "epoch": epoch,
                    "class_names": config.CLASS_NAMES,
                    "n_channels": X_train.shape[2],
                    "n_timesteps": X_train.shape[3],
                },
                config.MODEL_PATH,
            )
            logger.info("New best checkpoint saved (%.2f%%)", val_accuracy)

    logger.info("Best validation accuracy: %.2f%%", best_accuracy)

    checkpoint = torch.load(config.MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    _, test_accuracy = run_epoch(model, loaders["test"], criterion, device)
    logger.info("Held-out test accuracy: %.2f%%", test_accuracy)

    model.eval()
    predictions = []
    with torch.no_grad():
        for data, _ in loaders["test"]:
            predictions.extend(model(data.to(device)).argmax(1).cpu().numpy().tolist())
    predictions = np.array(predictions)

    print("\nClassification report\n" + classification_report(
        y_test, predictions, target_names=config.CLASS_NAMES, zero_division=0))
    print("Confusion matrix\n", confusion_matrix(y_test, predictions))

    history["test_accuracy"] = test_accuracy
    with open(config.HISTORY_PATH, "wb") as handle:
        pickle.dump(history, handle)
    logger.info("Training history written to %s", config.HISTORY_PATH)


if __name__ == "__main__":
    main()
