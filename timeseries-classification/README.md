# Time-Series Classification for *E. coli* Antibiotic Response

This repository implements a modular pipeline for **binary classification of *E. coli* time-series signals** across three tasks:

- **cip** – [PLACEHOLDER: Define clearly, e.g., ciprofloxacin response]
- **tet** – [PLACEHOLDER: Define clearly, e.g., tetracycline response]
- **ciptet** – [PLACEHOLDER: Define clearly, e.g., combined or multi-label task]

[PLACEHOLDER: Clearly define what the binary label represents (e.g., resistant vs susceptible, growth vs no growth, etc.)]

---

## Model Overview

The core model, `tsClassifier`, consists of two components:

### 1 Embedder (Representation Model)

The embedder converts a raw time series into a fixed-length embedding vector. The following embedders are supported:

- **TimesFM**  
  Foundation model used as-is (no additional training or fine-tuning in this repository).  
  [PLACEHOLDER: Add link/reference to TimesFM paper/repository]

- **MOMENT**  
  Foundation model used as-is (no additional training or fine-tuning in this repository).  
  [PLACEHOLDER: Add link/reference to MOMENT paper/repository]

- **VRAE**  
  Pre-trained model checkpoint provided at: models/model_12.pth

[PLACEHOLDER: Clarify when/where this model was trained and on which dataset split]

- **TS2Vec**  
Self-supervised representation model trained within this project.  
[PLACEHOLDER: Add link/reference to TS2Vec paper/repository]

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

## 📌 Notes

- Foundation models (TimesFM, MOMENT) are not fine-tuned in this repository.
- TS2Vec is trained within this project.
- VRAE uses a previously trained checkpoint.

[PLACEHOLDER: Add citation instructions if required]


## ⚙️ Environment Setup and code

### 1. Clone the repository

```bash
git clone 
cd timeseries-classification/
```

### 2. Create conda environment
```bash
conda env create -f environment.yml
conda activate ts-classification-env
cd ./timeseries-classification/timeseries-classification/
```

### 3. Create dataset split (for Autoencoders and MLP)
Processes raw data (growth_antibiotic_dataset.csv) and then creates split for experiments
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
Using best parameters train final instances. Models are stored in models.
```bash
python3 -m src.modeling.train_model
```



## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── README.md          <- The top-level README for developers using this project.
│
├── models             <- Trained models
    │
    ├── ts2vec_encoders.py      <- Trained ts2vec encoders saved at different epochs
│
├── notebooks          <- Jupyter notebooks.
│
├── environment.yml   <- The requirements file for reproducing the analysis environment
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes a Python module
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
    │   ├── hpo.py              <- Code to do hyperparameter tuning with optuna and mlflow   
    │   ├── test_ts2vec.py      <- Code to do calculate loss on testset for ts2vec AE at specific epochs
    │   ├── train_ts2vec.py     <- Code to train ts2vec
    │   ├── train_model.py      <- Code to train MLP models
    │   ├── predict.py          <- Code to run model inference with trained models      
    │   └── utils.py            <- Utility functions for modeling scripts
```

--------

