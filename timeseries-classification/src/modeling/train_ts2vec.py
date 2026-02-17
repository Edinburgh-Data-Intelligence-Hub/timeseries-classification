from src.ts2vec import TS2Vec, save_checkpoint_callback
import src.utils_ts2vec

import torch
import torch.nn as nn
import pandas as pd
import random
from torch.utils.data import DataLoader, TensorDataset
# from momentfm import MOMENTPipeline
from ast import literal_eval
import mlflow 

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

random.seed(SEED)

def main():
    # Train a TS2Vec model 
    # The training parameters are selected from the recommendations in the TS2Vec Paper: https://arxiv.org/pdf/2106.10466
    # Since most parameters are already mentioned, except epochs, we will just check best epoch value by saving model every 20 epochs.
    # we will also save the log loss file to plot curve
    output_dims = 320 
    batch_size = 8
    n_epochs = 100
    mlflow.set_experiment(f"ts2vec_train")
    run_name="train_final_model"

    # Train Final model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(PROCESSED_DATA_DIR / f"X_autoencoder_train.pkl", 'rb') as f:
        X_train = np.array(pickle.load(f))


    # MLflow setup 
    with mlflow.start_run(run_name=run_name) as run:
        after_epoch_callback = save_checkpoint_callback(
            save_every=20, 
            unit='epoch',
            output_dims=output_dims,
            model_dir=f'{MODELS_DIR}/ts2vec_encoders/'
        )

        model = TS2Vec(     
            input_dims=1, # 1D time series
            device=device,
            output_dims=output_dims,
            batch_size=batch_size,
            after_epoch_callback=after_epoch_callback,
        )


        loss_log = model.fit(
            X_train,
            verbose=True,
            n_epochs=n_epochs,
        )

        loss_df = pd.DataFrame({
            'Epoch': list(range(n_epochs)),
            'Loss': loss_log
        })

        loss_df.to_csv(
            f'{RESULTS_DIR}/ts2vec/loss_ts2vec.csv', 
            index=False
        )

        mlflow.log_params({
            "output_dims": output_dims,
            "lr": model.lr,
            "max_epochs": model.n_epochs,
            "batch_size": model.batch_size,
        })

    model.save(f'{MODELS_DIR}/ts2vec_encoder.pkl')

if __name__ == "__main__":
    main()
