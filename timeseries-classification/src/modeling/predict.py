from vrae.config import MODELS_DIR, PROCESSED_DATA_DIR

import torch
import torch.nn as nn
import pandas as pd
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


def main(
    # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
    features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
    model_path: Path = MODELS_DIR / "model.pkl",
    predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
    # -----------------------------------------
):
    embedder_name = "MOMENT-1-base" # "timesfm" or "MOMENT-1-base"
    ename = "timesfm" if embedder_name == "timesfm" else "moment"

    with open(PROCESSED_DATA_DIR / "X_test.pkl", 'rb') as f:
        X_test = np.array(pickle.load(f))

    with open(PROCESSED_DATA_DIR / "y_test.pkl", 'rb') as f:
        y_test = pickle.load(f)
        y_test = np.array(y_test)


    test_loader = get_dataloaders(
        batch_size=batch_size, 
        X=X_test, 
        y=y_test,
    )
    
    # Evaluate on test set
    test_metrics, y_probs, y_pred, y_true = evaluate(
        model=model,
        data_loader=test_loader,
        criterion=None,
        device=device,
        return_predictions=True,
    )
    
    print(f"Final Test Metrics: {test_metrics}")

        # Save test predictions
    results_df = pd.DataFrame(
        {   
            "y_true": y_true,
            "y_pred": y_pred,
            "y_probs": y_probs,
        }
    )
    
    results_df.to_csv(f"{RESULTS_DIR}/predictions/final_test_predictions_tsclassifier_{ename}.csv", index=False)

    


if __name__ == "__main__":
    app()
