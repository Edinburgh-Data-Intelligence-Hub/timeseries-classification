import pandas as pd
import pickle
import numpy as np
from src.config import *
from ast import literal_eval
import json

def main():
    embedder_names = ["timesfm", "MOMENT-1-base", "vae", "ts2vec"]
    best_params_dict = {}
    metric = "metrics.mean_best_val_f1"

    for embedder_name in embedder_names:
        ename = EMBEDDER_NAME_MAP[embedder_name]

        if embedder_name!="vae":
            # Get best hyperparameters from HPO results
            HPO_df = pd.read_csv(f"{RESULTS_DIR}/HPO/tsclassifier_optuna_cv_{ename}_final_HPO.csv")
            best_run = HPO_df.sort_values(metric, ascending=False).iloc[0]
            best_params = best_run.filter(like="params.").to_dict()
            best_params = {k.replace("params.", ""): v for k, v in best_params.items()}
            max_epochs = int(best_run['metrics.mean_final_epoch'])
            batch_size = int(best_params['batch_size'])
            hidden_dims = literal_eval(best_params['hidden_dims'])
            dropout = best_params['dropout']
            lr = best_params['lr']
        else:
            # Parameters hardcoded for VAE antibiotic_control_classification.ipynb
            # change/automate as needed
            max_epochs = 50 
            batch_size = 200
            hidden_dims = [20,10]
            dropout = 0.0
            lr = 0.001

        best_params_dict[embedder_name] = {
            "max_epochs": max_epochs,
            "batch_size": batch_size,
            "hidden_dims": hidden_dims,
            "dropout": dropout,
            "lr": lr
        }
        
    # Save best_params_dict to a pickle file
    with open(RESULTS_DIR / "HPO" / "best_params.json", 'w') as f:
        json.dump(best_params_dict, f, indent=4)
    
if __name__ == "__main__":
    main()