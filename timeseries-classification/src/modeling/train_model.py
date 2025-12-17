import torch
import torch.nn as nn
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from momentfm import MOMENTPipeline
from ast import literal_eval

import pickle
import numpy as np
from src.tsclassifier import Embedder, tsClassifier
from src.config import *
from src.modeling.utils import train_without_early_stopping, evaluate, create_model, get_dataloaders
import math

from tqdm import tqdm
import time
import mlflow 

# Train Final model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder_name = "MOMENT-1-base" # "timesfm" or "MOMENT-1-base"
ename = "timesfm" if embedder_name == "timesfm" else "moment"

random.seed(SEED)

# MLflow setup 
mlflow.set_experiment(f"tsclassifier_{ename}_final_train")
with mlflow.start_run(run_name="train_final_model") as run:
    # Get best hyperparameters from HPO results
    HPO_df = pd.read_csv(f"{RESULTS_DIR}/HPO/tsclassifier_optuna_cv_{ename}_final_HPO.csv")
    best_run = HPO_df.sort_values("metrics.mean_best_val_f1", ascending=False).iloc[0]
    best_trial = best_run["tags.mlflow.runName"]
    best_params = best_run.filter(like="params.").to_dict()
    best_params = {k.replace("params.", ""): v for k, v in best_params.items()}
    max_epochs = int(best_run['metrics.mean_final_epoch'])
    batch_size = int(best_params['batch_size'])
    hidden_dims = literal_eval(best_params['hidden_dims'])
    dropout = best_params['dropout']
    lr = best_params['lr']
    
    mlflow.log_params({
        "hidden_dims": str(hidden_dims),
        "dropout": dropout,
        "lr": lr,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "embedder_name": embedder_name,
    })
    
    with open(PROCESSED_DATA_DIR / "X_train.pkl", 'rb') as f:
        X_train = np.array(pickle.load(f))
    
    with open(PROCESSED_DATA_DIR / "X_test.pkl", 'rb') as f:
        X_test = np.array(pickle.load(f))
    
    with open(PROCESSED_DATA_DIR / "y_train.pkl", 'rb') as f:
        y_train = pickle.load(f)
        num_classes = len(set(y_train))
        y_train = np.array(y_train)
    
    with open(PROCESSED_DATA_DIR / "y_test.pkl", 'rb') as f:
        y_test = pickle.load(f)
        y_test = np.array(y_test)
    
    if num_classes != len(set(y_test)):
        raise ValueError(f"Number of classes in train {num_classes} and test {len(set(y_test))} do not match!")
    
    # generate dataloaders and split into train and val
    train_loader = get_dataloaders(
        batch_size=batch_size, 
        X=X_train, 
        y=y_train,
    )
    
    test_loader = get_dataloaders(
        batch_size=batch_size, 
        X=X_test, 
        y=y_test,
    )
    
    # Create model
    model = create_model(
        device=device,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        freeze_embedder=True,
        embedder_name=embedder_name,
    )
    
    # Criterion and optimizer
    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model, history = train_without_early_stopping(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        max_epochs=max_epochs,
        return_model=True,
    )
    
    # Save model
    save_tsclassifier(model, MODELS_DIR / "tsclassifier_{ename}.pt")

    # Evaluate on test set
    test_metrics, y_probs, y_pred, y_true = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=criterion,
        device=device,
        return_predictions=True,
    )
    
    print(f"Final Test Metrics: {test_metrics}")
    # MLflow logging 
    mlflow.log_metrics(
        {
            "test_acc": test_metrics["accuracy"],
            "test_precision": test_metrics["precision"],
            "test_recall": test_metrics["recall"],
            "test_f1": test_metrics["f1"],
            "test_auroc": test_metrics["auroc"] if test_metrics["auroc"] is not None else float("nan"),
            "test_auprc": test_metrics["auprc"] if test_metrics["auprc"] is not None else float("nan"),
        }
    )
    
    # Save test predictions
    results_df = pd.DataFrame(
        {   
            "y_true": y_true,
            "y_pred": y_pred,
            "y_probs": y_probs,
        }
    )
    
    results_df.to_csv(f"{RESULTS_DIR}/predictions/final_test_predictions_tsclassifier_{ename}.csv", index=False)

# (optional) you could also save the model as an artifact here
# torch.save(model.state_dict(), "model.pt")
# mlflow.log_artifact("model.pt")





