# Time-Series Classification for *E. coli* Antibiotic Response

This repository implements a modular pipeline for **binary classification of *E. coli* time-series signals** across three tasks:

- **cip** – An antibiotic causing DNA damage in teh cell
- **tet** – An antibiotic slowing down cell growth
- **ciptet** – A combination of the two treatments

More information about the experimental setup, doses and data generation can be found in [[1]](#1).

---

## Model Overview

The core model, `tsClassifier`, consists of two components:

### 1 Embedder (Representation Model)

The embedder converts a raw time series into a fixed-length embedding vector. The following embedders are supported:

- **VRAE**  
  A variational Recurrent Auto-Encoder architecture from https://github.com/tejaslodaya/timeseries-clustering-vae. Model was pre-trained on control data from this dataset (it has never seen antibiotic data, more information in [[2]](#2)), model checkpoint provided at: models/model_12.pth.

- **MOMENT**  
  Foundation model used as-is (no additional training or fine-tuning in this repository), from [[3]](#3), the "MOMENT1-base" version was used.

- **TimesFM**  
  Foundation model used as-is (no additional training or fine-tuning in this repository), from [[4]](#4).

- **TS2Vec**  
Self-supervised representation model trained within this project, from [[5]](#5)

---

### 2 Classifier (MLP Head)

A two-layer feed-forward neural network (MLP) that:

- Takes the embedding vector as input
- Performs binary classification for the selected task

The MLP architecture:
- Fully connected layer
- ReLU activation
- Dropout
- Final linear layer → binary output

---

## Model Variants

Each model is defined as: Embedder + MLP

Examples:
- `timesfm + MLP`
- `ts2vec + MLP`
- `moment + MLP`
- `vrae + MLP`

For each task (**cip**, **tet**, **ciptet**), a **separate MLP classifier is trained per embedder**.

Example (TimesFM):
- tsclassifier_timesfm_cip.pt
- tsclassifier_timesfm_tet.pt
- tsclassifier_timesfm_ciptet.pt


This results in three independently trained classifier heads per embedder.

---

## ⚙️ Environment Setup and code

### 1. Clone the repository

```bash
git clone https://github.com/Edinburgh-Data-Intelligence-Hub/timeseries-classification.git
cd timeseries-classification/
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate ts-classification-env
cd ./timeseries-classification/timeseries-classification/
```

### 3. Create dataset split (for Autoencoders and MLP)
Processes raw data (growth_antibiotic_dataset.csv) and then creates split for experiments. You need to define which variable to use (size = cell length, sos = cell fluorescence), wether the data need to be re-scaled or not and the task (cip, tet or ciptet).
```bash
python3 -m src.dataset
```

This generates datasplits for experiments 
- *X_autoencoder_train.pkl*, 
- *X_autoencoder_test.pkl*, 
- *y_autoencoder_train.pkl*, 
- *y_autoencoder_test.pkl*,
- *X_{task}_full.pkl*,
- *X_{task}_train.pkl*,
- *X_{task}_test.pkl*,
- *y_{task}_full.pkl*,
- *y_{task}_train.pkl*,
- *y_{task}_test.pkl*,

### 4. Train TS2VEC encoder
Train TS2VEC using *X_autoencoder_train.pkl* and *y_autoencoder_train.pkl*, trained encoder is stored in models/ts2vec_encoders
```bash
python3 -m src.modeling.train_ts2vec
```

### 5. Hyperparameter tuning for MLPs of tsClassifier
Tune parameters for MLPs for TimesFM, MOMENT and TS2VEC. Results are stored in results/HPO.
```bash
python3 -m src.modeling.hpo
```

### 6. Train final tsClassifier instances
First save best parameters using best_params.py for all embedders from HPO which can then be loaded for training. 
Train final model instances using best parameters. Models are stored in ./models.
```bash
python3 -m src.modeling.train_model
```

### 7. Prediction
Using trained models, predict using *X_{task}_test.pkl*, *y_{task}_test.pkl*,
```bash
python3 -m src.modeling.predict
```

## Project Organization

```
├── LICENSE
├── README.md
│
├── models             <- Trained models
    │
    ├── ts2vec_encoders      <- Trained ts2vec encoders saved at different epochs
│
├── notebooks          <- Jupyter notebooks for data analysis and visualisation
│
├── environment.yml
│
└── src
    │
    ├── __init__.py
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to generate data
    │
    ├── ts2vec.py               <- ts2vec code
    │
    ├── tsclassifier.py         <- tsclassifer code
    │
    ├── vrae.py                 <- VRAE code
    │
    ├── utils_ts2vec.py         <- Utility functions for ts2vec code
    │
    ├── utils_timesfm.py        <- Utility functions for ts2vec code
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── hpo.py              <- Code to do hyperparameter tuning with optuna  
    │   ├── test_ts2vec.py      <- Code to do calculate loss on testset for ts2vec AE at specific epochs
    │   ├── train_ts2vec.py     <- Code to train ts2vec
    │   ├── train_model.py      <- Code to train MLP models
    │   ├── predict.py          <- Code to run model inference with trained models    
    │   ├── best_params.py      <- Extract best parameters from hpo optuna study
    │   └── utils.py            <- Utility functions for modeling scripts
```

--------

## References
<a id="1">[1]</a> Broughton, J., Fraisse, A. & El Karoui, M. Suppression of bacterial cell death underlies the antagonistic interaction between ciprofloxacin and tetracycline. Mol Syst Biol 22, 102–118 (2026). https://doi.org/10.1038/s44320-025-00162-w

<a id="2">[2]</a> Achille Fraisse, Diego A. Oyarzún, Meriem El Karoui. Representation learning of single-cell time-series with deep variational autoencoders. bioRxiv 2025.09.22.677729; doi: https://doi.org/10.1101/2025.09.22.677729

<a id="3">[3]</a> Mononito Goswami and Konrad Szafer and Arjun Choudhry and Yifu Cai and Shuo Li and Artur Dubrawski, MOMENT: A Family of Open Time-series Foundation Models, https://arxiv.org/abs/2402.03885

<a id="4">[4]</a> Abhimanyu Das, Weihao Kong, Rajat Sen, Yichen Zhou. A decoder-only foundation model for time-series forecasting. https://arxiv.org/html/2310.10688v2

<a id="5">[5]</a> Zhihan Yue and Yujing Wang and Juanyong Duan and Tianmeng Yang and Congrui Huang and Yunhai Tong and Bixiong Xu. TS2Vec: Towards Universal Representation of Time Series. arXiv, https://arxiv.org/abs/2106.10466. 