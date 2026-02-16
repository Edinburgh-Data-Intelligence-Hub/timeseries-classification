# from ..vrae.config import MODELS_DIR, PROCESSED_DATA_DIR
import torch
import pandas as pd
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.model_selection import train_test_split
# from momentfm import MOMENTPipeline
from ast import literal_eval
import random

import pickle
import numpy as np
from src.tsclassifier import load_tsclassifier
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

random.seed(SEED)

def main():
    batch_size = 32 # standard
    embedder_name = "vae" # "timesfm" or "MOMENT-1-base" or "vae" or "ts2vec"
    ename = EMBEDDER_NAME_MAP[embedder_name]
    task = 'cip' #cip, tet or ciptet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  

    with open(PROCESSED_DATA_DIR / f"X_{task}_test.pkl", 'rb') as f:
        X_test = np.array(pickle.load(f))

    with open(PROCESSED_DATA_DIR / f"y_{task}_test.pkl", 'rb') as f:
        y_test = pickle.load(f)
        y_test = np.array(y_test)

    test_loader = get_dataloader(
        batch_size=batch_size, 
        X=X_test, 
        y=y_test,
    )

    model = load_tsclassifier(
        model_path = MODELS_DIR / f"tsclassifier_{ename}_{task}.pt", 
        model_dir = MODELS_DIR,
        device = device,
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
    
    # results_df.to_csv(f"{RESULTS_DIR}/predictions/final_test_predictions_tsclassifier_{ename}_{task}.csv", index=False)


if __name__ == "__main__":
    main()
