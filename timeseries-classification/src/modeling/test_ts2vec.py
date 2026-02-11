from src.ts2vec import TS2Vec, hierarchical_contrastive_loss
import src.utils_ts2vec
from src.config import *

import torch
import pandas as pd

import pickle
import numpy as np

from tqdm import tqdm
import random
random.seed(SEED)

with open(PROCESSED_DATA_DIR / f"X_autoencoder_test.pkl", 'rb') as f:
    X_test = np.array(pickle.load(f))

test_loss_log_df = []
k = 10 # number of times to repeat the test to get a sense of variability in the loss
epochs = [20,40,60,80,100]

for k in range(k): 
    test_loss_log = []
    for epoch in epochs:
        model = TS2Vec(     
            input_dims=1, # 1D time series
            device="cpu",
        )
        model.load(f'{MODELS_DIR}/ts2vec_encoders/model_epoch_{epoch}_outdims_320.pkl') 
        model.net.eval() # set model to evaluation mode for testing

        x = torch.as_tensor(X_test)
        
        if model.max_train_length is not None and x.size(1) > model.max_train_length:
            window_offset = np.random.randint(x.size(1) - model.max_train_length + 1)
            x = x[:, window_offset : window_offset + model.max_train_length]
        x = x.to(model.device).to(torch.float)
        
        ts_l = x.size(1)
        crop_l = np.random.randint(low=2 ** (model.temporal_unit + 1), high=ts_l+1)
        crop_left = np.random.randint(ts_l - crop_l + 1)
        crop_right = crop_left + crop_l
        crop_eleft = np.random.randint(crop_left + 1)
        crop_eright = np.random.randint(low=crop_right, high=ts_l + 1,)
        crop_offset = np.random.randint(low=-crop_eleft, high=ts_l - crop_eright + 1, size=x.size(0), )
        
    
        out1 = model._net(src.utils_ts2vec.take_per_row(x, crop_offset + crop_eleft, crop_right - crop_eleft))
        out1 = out1[:, -crop_l:]
        
        out2 = model._net(src.utils_ts2vec.take_per_row(x, crop_offset + crop_left, crop_eright - crop_left))
        out2 = out2[:, :crop_l]
        
        loss = hierarchical_contrastive_loss(
            out1,
            out2,
            temporal_unit=model.temporal_unit
        )

        test_loss_log.append(loss.item()) # loss is a tensor, convert to scalar for logging

    test_loss_log_ = pd.DataFrame({
        "Epoch": [20,40,60,80,100],
        "Loss": test_loss_log,
        "k": [k]*len(epochs) # to keep track of which run the loss corresponds to
    })

    test_loss_log_df = test_loss_log_df + [test_loss_log_]

test_loss_log_df = pd.concat(test_loss_log_df, ignore_index=True)
print(test_loss_log_df)
test_loss_log_df.to_csv(f'{RESULTS_DIR}/ts2vec/test_loss_ts2vec_k.csv', index=False)
