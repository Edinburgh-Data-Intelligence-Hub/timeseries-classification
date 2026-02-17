import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from momentfm import MOMENTPipeline

import pickle
import numpy as np
from src.tsclassifier import Embedder, tsClassifier
from src.config import *
import math

from tqdm import tqdm
import time
import mlflow 

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc
)


def get_dataloader(
        batch_size: int = 32, 
        X=None, 
        y=None, 
    ) -> tuple[DataLoader, int]:
    """
    Creates DataLoaders for training and validation datasets.
    args:
        batch_size: The batch size for the DataLoaders.
        X: data
        y: label
    Returns:
        data_loader.
    
    """
    for name, value in [("X", X), ("y", y)]:
        if value is None:
            raise ValueError(f"{name} must be provided, but got None.")
    
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y)
    
    ds = TensorDataset(X, y)
    
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    return loader

def create_model(
    device: torch.device = torch.device,
    num_classes: int = 2,
    hidden_dims: list = [20, 10],
    dropout: float = 0.1,
    freeze_embedder: bool = True,
    embedder_name: str = None,
    embedder: Embedder = None,
    embedder_path: str = None
) -> tsClassifier:
    """
    Factory to create a tsClassifier.
    
    If shared_embedder is provided, reuse it.
    Otherwise create a new Embedder(name="timesfm").
    """
    if embedder is None and embedder_name is None:
        raise ValueError("Either embedder or embedder_name must be provided.")
    elif embedder is not None and embedder_name is not None:
        raise Warning("Provided both embedder and embedder_name; using only embedder") # check this
    elif embedder is None:
        embedder = Embedder(
            name=embedder_name, 
            device=device,
            embedder_dir=embedder_path)
    
    model = tsClassifier(
        embedder=embedder,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        freeze_embedder=freeze_embedder,
    ).to(device)
    
    if freeze_embedder:
        for p in model.embedder.parameters():
            p.requires_grad_(False)
    
    return model

def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        with torch.no_grad():
            preds = torch.argmax(logits, dim=1)
            running_correct += (preds == targets).sum().item()
            running_total += targets.size(0)
        
    epoch_loss = running_loss / running_total
    epoch_acc = running_correct / running_total
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    return_predictions: bool = False,
    return_loss: bool = False
):
    """
    Returns a dict with:
      - loss
      - accuracy
      - f1
      - precision
      - recall
      - auroc (binary only, else None)
      - auprc (binary only, else None)
    If return_predictions=True, also returns probs, y_pred and targets tensors.
    """
    if return_loss and criterion!=None:
        Warning("Return loss set true, but criterion not provided, defaulting to return_loss=False")
        
    model.eval()
    
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    
    all_logits = []
    all_targets = []
    
    
    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        logits = model(inputs)
        
        if return_loss:
            loss = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)
        preds = torch.argmax(logits, dim=1)
        
        running_correct += (preds == targets).sum().item()
        running_total += targets.size(0)
        
        all_logits.append(logits.detach().cpu())
        all_targets.append(targets.detach().cpu())
    if return_loss:    
        epoch_loss = running_loss / running_total
    accuracy = running_correct / running_total
    
    # stack over whole dataset
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    y_logits = all_logits.numpy()
    y_true = all_targets.numpy()
    y_pred = all_logits.argmax(dim=1).numpy()
    
    # defaults
    auroc = None
    auprc = None
    
    if model.num_classes == 2:
        # binary: use prob of class 1
        probs = all_logits.softmax(dim=1)[:, 1].numpy()
        
        # handle edge cases where only one class present in y_true
        try:
            auroc = roc_auc_score(y_true, probs)
        except ValueError:
            auroc = None
        
        try:
            precision, recall, _ = precision_recall_curve(y_true, probs)
            auprc = auc(recall, precision)
        except ValueError:
            auprc = None
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
    else:
        # multi-class: macro-averaged
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        # AUROC/AUPRC can be added in one-vs-rest style 
    
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "auprc": auprc,
    }
    
    if return_loss:
        metrics["loss"] = epoch_loss
    
    if return_predictions:
        return metrics, probs, y_pred, y_true
    else:
        return metrics



def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_epochs: int = 500,
    patience: int = 20,
    monitor_metric: str = "f1",  # "auprc" -> "f1" -> "accuracy"
    return_model: bool = False,
):
    """
    Train model with early stopping on a chosen validation metric.

    Returns:
        history: dict of lists (per-epoch metrics)
        epoch: int (total epochs run)
        best_epoch: int
        best_val_metrics: dict (metrics at best epoch)
    """
    model.to(device)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "val_auroc": [],
        "val_auprc": [],
    }
    
    best_score = -float("inf")
    best_epoch = 0
    best_val_metrics = None
    epochs_without_improvement = 0
    
    for epoch in tqdm(range(max_epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        val_metrics = evaluate(
            model, val_loader, criterion, device, return_loss=True
        )
        
        # log history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_precision"].append(val_metrics["precision"])
        history["val_recall"].append(val_metrics["recall"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auroc"].append(val_metrics["auroc"])
        history["val_auprc"].append(val_metrics["auprc"])
        
        # choose metric to monitor
        if monitor_metric in val_metrics.keys():
            monitor = val_metrics[monitor_metric]
        else:
            raise ValueError(f"Unknown monitor_metric: {monitor_metric}")
        
        # print progress
        
        print(
            f"Epoch {epoch+1}/{max_epochs} "
            f"- train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} "
            f"- val_loss: {val_metrics['loss']:.4f} - val_acc: {val_metrics['accuracy']:.4f} "
            f"- val_f1: {val_metrics['f1'] if val_metrics['f1'] is not None else float('nan'):.4f} "
            f"- val_auroc: {val_metrics['auroc'] if val_metrics['auroc'] is not None else float('nan'):.4f}"
            f"- val_auprc: {val_metrics['auprc'] if val_metrics['auprc'] is not None else float('nan'):.4f}"
        )
        
        # MLflow per-epoch logging (assumes an active run) ----
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_precision": val_metrics["precision"],
                "val_recall": val_metrics["recall"],
                "val_f1": val_metrics["f1"],
                "val_auroc": val_metrics["auroc"] if val_metrics["auroc"] is not None else float("nan"),
                "val_auprc": val_metrics["auprc"] if val_metrics["auprc"] is not None else float("nan"),
            },
            step=epoch,
        )
        
        # early stopping bookkeeping
        if monitor > best_score:
            best_score = monitor
            best_epoch = epoch
            best_val_metrics = val_metrics
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            print(
                f"Early stopping at epoch {epoch+1}; "
                f"best epoch was {best_epoch+1} with monitored metric = {best_score:.4f}"
            )
            break
            
    if return_model:
        return model, history, epoch, best_epoch, best_val_metrics
    else:
        return history, epoch, best_epoch, best_val_metrics
    

def train_without_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_epochs: int = 50,
    return_model: bool = False
):
    """
    Train model with early stopping on a chosen validation metric.

    Returns:
        history: dict of lists (per-epoch metrics)
    """
    model.to(device)
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "train_precision": [],
        "train_recall": [],
        "train_f1": [],
        "train_auroc": [],
        "train_auprc": [],
    }
    
    for epoch in tqdm(range(max_epochs)):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        
        _metrics = evaluate(
            model, train_loader, criterion, device, return_loss=True
        )
        
        # log history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_precision"].append(_metrics["precision"])
        history["train_recall"].append(_metrics["recall"])
        history["train_f1"].append(_metrics["f1"])
        history["train_auroc"].append(_metrics["auroc"])
        history["train_auprc"].append(_metrics["auprc"])
        
        # print progress
        
        print(
            f"Epoch {epoch+1}/{max_epochs} "
            f"- train_loss: {train_loss:.4f} - train_acc: {train_acc:.4f} "
            f"- train_f1: {_metrics['f1'] if _metrics['f1'] is not None else float('nan'):.4f} "
            f"- train_auroc: {_metrics['auroc'] if _metrics['auroc'] is not None else float('nan'):.4f}"
            f"- train_auprc: {_metrics['auprc'] if _metrics['auprc'] is not None else float('nan'):.4f}"
        )
        
        # MLflow per-epoch logging (assumes an active run) ----
        mlflow.log_metrics(
            {
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_precision": _metrics["precision"],
                "train_recall": _metrics["recall"],
                "train_f1": _metrics["f1"],
                "train_auroc": _metrics["auroc"] if _metrics["auroc"] is not None else float("nan"),
                "train_auprc": _metrics["auprc"] if _metrics["auprc"] is not None else float("nan"),
            },
            step=epoch,
        )

    if return_model:
        return model, history
    else:
        return history



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MLflow setup 
    mlflow.set_experiment("tsclassifier_MOMENT-1-base")
    with mlflow.start_run(run_name="train_trial"):

        with open(PROCESSED_DATA_DIR / "X_train.pkl", 'rb') as f:
            X = np.array(pickle.load(f))


        with open(PROCESSED_DATA_DIR / "y_train.pkl", 'rb') as f:
            y = pickle.load(f)
            num_classes = len(set(y))
            y = np.array(y)
        f.close()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            stratify=y,
            test_size=0.2
        )

        # generate dataloaders and split into train and val
        train_loader, val_loader = get_dataloader(
            batch_size=32, 
            X_train=X_train, 
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        model = create_model(
            device=device,
            num_classes=num_classes,
            hidden_dims=[20, 10],
            dropout=0.1,
            freeze_embedder=True,
            embedder_name="MOMENT-1-base",
        )

        optimizer = torch.optim.Adam(model.mlp.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        history, final_epoch, best_epoch, best_val_metrics = train_with_early_stopping(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            max_epochs=10,
            patience=2,
            monitor_metric="f1",  # or "f1" / "accuracy"
        )

        print(f"Best epoch: {best_epoch+1}")
        print("Best val metrics:", best_val_metrics)

        # log summary metrics
        mlflow.log_metric("best_epoch", best_epoch + 1)

        for k, v in best_val_metrics.items():
            mlflow.log_metric(f"best_val_{k}", v if v is not None else float("nan"))

    # (optional) you could also save the model as an artifact here
    # torch.save(model.state_dict(), "model.pt")
    # mlflow.log_artifact("model.pt")


if __name__ == "__main__":
    main()