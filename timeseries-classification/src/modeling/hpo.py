import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold

from src.modeling.train import train_with_early_stopping, evaluate, create_model, get_dataloaders
from src.tsclassifier import Embedder, tsClassifier
from src.config import *
import math

import optuna
import mlflow
import pickle


# -------------- Optuna + MLflow integration --------------

class HPO:
    def __init__(
        self,
        experiment_name: str = "tmp",
        embedder_name: str = "timesfm",
        max_epochs: int = 500,
        patience: int = 20,
        monitor_metric: str = "f1",
        n_splits: int = 5,
        num_classes: int = 2,
        X = None,
        y = None,
        embedder = None,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.experiment_name = experiment_name
        self.embedder_name = embedder_name
        self.max_epochs = max_epochs
        self.monitor_metric = monitor_metric
        self.X = X
        self.y = y
        self.patience = patience
        self.n_splits = n_splits
        self.num_classes = num_classes
        self.embedder = Embedder(name=self.embedder_name, device=self.device)
        
        
    def objective(self, trial: optuna.Trial) -> float:

        # --- sample hyperparameters ---
        # hidden_dim1 = trial.suggest_int("hidden_dim1", 16, 128, log=True)
        # hidden_dim2 = trial.suggest_int("hidden_dim2", 8, 64, log=True)
        hidden_dim1 = 20
        hidden_dim2 = 10
        hidden_dim = [hidden_dim1, hidden_dim2]
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_int("batch_size", 16, 128, log=True)

        # k-fold (or self.n_splits) stratified CV
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=42,
        )

        fold_scores = []  # store best metric per fold

        # One MLflow run per trial
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=False):
            # log hyperparameters (once per trial)
            mlflow.log_params({
                "hidden_dims": str(hidden_dim),
                "dropout": dropout,
                "lr": lr,
                "max_epochs": self.max_epochs,
                "patience": self.patience,
                "batch_size": batch_size,
                "embedder_name": self.embedder_name,
                "monitor_metric": self.monitor_metric,
                "n_splits": self.n_splits,
            })

            # loop over folds
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(self.X, self.y)):
                #  make fold data
                X_train_fold = self.X[train_idx]
                y_train_fold = self.y[train_idx]
                X_val_fold = self.X[val_idx]
                y_val_fold = self.y[val_idx]

                train_loader, val_loader = get_dataloaders(
                    batch_size=batch_size, 
                    X_train=X_train_fold, 
                    y_train=y_train_fold,
                    X_test=X_val_fold,
                    y_test=y_val_fold
                )

                #  child run per FOLD (nested=True) 
                with mlflow.start_run(
                    run_name=f"trial_{trial.number}_fold_{fold_idx}",
                    nested=True,
                ):
                    # model / optimizer for this fold 
                    model = create_model(
                        device=self.device,
                        num_classes=self.num_classes,
                        hidden_dims=hidden_dim,
                        dropout=dropout,
                        freeze_embedder=True,
                        embedder=self.embedder,
                    )

                    optimizer = torch.optim.Adam(model.mlp.parameters(), lr=lr)
                    criterion = nn.CrossEntropyLoss()

                    #  train with early stopping on this fold 
                    _, best_epoch, best_val_metrics = train_with_early_stopping(
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=self.device,
                        max_epochs=self.max_epochs,
                        patience=self.patience,
                        monitor_metric=self.monitor_metric,  # or "f1" / "accuracy"
                    )

                    # score for this fold based on chosen metric
                    fold_score = best_val_metrics[self.monitor_metric]
                    fold_scores.append(fold_score)

                    # log summary metrics for THIS FOLD (child run)
                    mlflow.log_metric("best_epoch", best_epoch + 1)
                    for k, v in best_val_metrics.items():
                        mlflow.log_metric(
                            f"best_val_{k}",
                            v if v is not None else float("nan"),
                        )

                # (optional) report intermediate performance to Optuna for pruning
                mean_so_far = float(np.mean(fold_scores))
                trial.report(mean_so_far, step=fold_idx)

                if trial.should_prune():
                    mlflow.log_metric("pruned", 1)
                    raise optuna.exceptions.TrialPruned()

            # final objective: mean metric across folds
            mean_score = float(np.mean(fold_scores))
            print(f"Trial {trial.number} done, {self.monitor_metric} (mean over folds) = {mean_score:.4f}")

            # log final CV score for this trial
            mlflow.log_metric(f"mean_best_val_{self.monitor_metric}", mean_score)

        return mean_score

def main():

    hpo = HPO(
        experiment_name="tsclassifier_optuna_cv_moment_final",
        embedder_name="MOMENT-1-base",
        max_epochs=250,
        patience=20,
        monitor_metric='f1',
        n_splits=5
    )

    # load data ONCE instead of every trial
    with open(PROCESSED_DATA_DIR / "X_train.pkl", "rb") as f:
        X = pickle.load(f)
        hpo.X = np.array(X)

    with open(PROCESSED_DATA_DIR / "y_train.pkl", "rb") as f:
        y = pickle.load(f)
        num_classes = len(set(y))
        hpo.num_classes = num_classes
        hpo.y = np.array(y)


    # Set MLflow experiment (creates it if not exists)
    mlflow.set_experiment(hpo.experiment_name)

    # create study and run optimization
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
    )

    print("Starting hyperparameter optimization...")
    study.optimize(
        hpo.objective, 
        n_trials=50,
        n_jobs=1,
    )

    print("Best trial:")
    print("  value (val_f1):", study.best_trial.value)
    print("  params:", study.best_trial.params)

    # Save csv of all runs
    exp = mlflow.get_experiment_by_name(hpo.experiment_name)  # change to your exp name
    experiment_id = exp.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    runs.to_csv(f"{RESULTS_DIR}/HPO/{hpo.experiment_name}_HPO.csv", index=False)

if __name__ == "__main__":
    main()