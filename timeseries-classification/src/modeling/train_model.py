import torch
import torch.nn as nn
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
# from momentfm import MOMENTPipeline
from ast import literal_eval

import pickle
import numpy as np
from src.tsclassifier import save_tsclassifier
from src.config import *
from src.modeling.utils import (
    train_without_early_stopping, 
    evaluate, 
    create_model,
    get_dataloader
)
import math

from tqdm import tqdm
import time
import mlflow 

# Train Final model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedder_name = "vae" # "timesfm" or "MOMENT-1-base" or "vae"
task = 'ciptet' #cip, tet or ciptet
ename = EMBEDDER_NAME_MAP[embedder_name]

random.seed(SEED)

# MLflow setup 
mlflow.set_experiment(f"tsclassifier_{ename}_{task}_final_train")
with mlflow.start_run(run_name="train_final_model") as run:
    # Get best hyperparameters from HPO results
    # HPO_df = pd.read_csv(f"{RESULTS_DIR}/HPO/tsclassifier_optuna_cv_{ename}_final_HPO.csv")
    # best_run = HPO_df.sort_values("metrics.mean_best_val_f1", ascending=False).iloc[0]
    # best_trial = best_run["tags.mlflow.runName"]
    # best_params = best_run.filter(like="params.").to_dict()
    # best_params = {k.replace("params.", ""): v for k, v in best_params.items()}
    # max_epochs = int(best_run['metrics.mean_final_epoch'])
    # batch_size = int(best_params['batch_size'])
    # hidden_dims = literal_eval(best_params['hidden_dims'])
    # dropout = best_params['dropout']
    # lr = best_params['lr']

    max_epochs = 50
    batch_size = 200
    hidden_dims = [20,10]
    dropout = 0.0
    lr = 0.001

    mlflow.log_params({
        "hidden_dims": str(hidden_dims),
        "dropout": dropout,
        "lr": lr,
        "max_epochs": max_epochs,
        "batch_size": batch_size,
        "embedder_name": embedder_name,
    })
    
    with open(PROCESSED_DATA_DIR / f"X_{task}_train.pkl", 'rb') as f:
        X_train = np.array(pickle.load(f))

    with open(PROCESSED_DATA_DIR / f"y_{task}_train.pkl", 'rb') as f:
        y_train = pickle.load(f)
        num_classes = len(set(y_train))
        y_train = np.array(y_train)
    
    # if num_classes != len(set(y_test)):
    #     raise ValueError(f"Number of classes in train {num_classes} and test {len(set(y_test))} do not match!")
    
    # generate dataloaders and split into train and val
    train_loader = get_dataloader(
        batch_size=batch_size, 
        X=X_train, 
        y=y_train,
    )

    embedder_path = MODELS_DIR 

    # Create model
    model = create_model(
        device=device,
        num_classes=num_classes,
        hidden_dims=hidden_dims,
        dropout=dropout,
        freeze_embedder=True,
        embedder_name=embedder_name,
        embedder_path=embedder_path,
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
    save_tsclassifier(model, MODELS_DIR / f"tsclassifier_{ename}_{task}.pt")


# (optional) you could also save the model as an artifact here
# torch.save(model.state_dict(), "model.pt")
# mlflow.log_artifact("model.pt")


