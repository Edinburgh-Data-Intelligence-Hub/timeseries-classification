import numpy as np
import timesfm
import torch 
import torch.nn as nn
import torch.nn.functional as F

from . import utils_timesfm

class Embedder(nn.Module):
    def __init__(self, name: str, device="cuda"):
        super().__init__()
        self.name = name.lower()
        self.device = torch.device(device)

        # load the model 
        if self.name == "timesfm":
            self.model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=True)
            self.model.compile(
                timesfm.ForecastConfig(
                    max_context=1024,
                    max_horizon=256,
                    normalize_inputs=True,
                    use_continuous_quantile_head=True,
                    force_flip_invariance=True,
                    infer_is_positive=True,
                    fix_quantile_crossing=True,
                )
            )
            self.embedding_dim = self.model.model.config.stacked_transformers.transformer.hidden_dims
            self.model.model.eval()

        elif self.name == "vae":
            pass
            self.model = None # fill in with actual VAE model
            self.embedding_dim = None  # fill in with actual dimension
            self.model.to(self.device)
            self.model.eval()  # freeze by default

        else:
            raise ValueError(f"Unknown embedder: {name}")

        

    @torch.no_grad()
    def forward(self, inputs):
        if self.name == "timesfm":
            _, outout_embeddings = utils_timesfm.get_embeddings(
                horizon=12,
                model=self.model,
                inputs=inputs,
                layers_to_hook=-1,
            )
            return outout_embeddings[0][:,-1,:]

        elif self.name == "vae":
            # replace with your VAE logic
            return self.model.encode(inputs)

class EmbeddingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[20, 10], num_classes=2, dropout=0.1):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.network = nn.Sequential(*layers)
        self.out = nn.Linear(prev_dim, num_classes)

    def forward(self, x):
        """
        x: tensor of shape (batch_size, embedding_dim)
        """
        x = self.network(x)
        return self.out(x)
    
class tsClassifier(nn.Module):
    def __init__(self, embedder, num_classes,
                 hidden_dims=[20, 10], dropout=0.1,
                 freeze_embedder=True):
        """
        embed_fn: callable that maps raw_x -> embeddings (B, embedding_dim)
                  can be a plain function or an nn.Module with __call__
        """
        super().__init__()
        self.embedder = embedder
        self.freeze_embedder = freeze_embedder
        self.num_classes = num_classes

        self.mlp = EmbeddingMLP(
            input_dim=self.embedder.embedding_dim,
            hidden_dims=hidden_dims,
            num_classes=self.num_classes,
            dropout=dropout,
        )

    def forward(self, inputs):
        """
        raw_x: raw input (e.g. time series, images, whatever your embedder expects)
        """
        if self.freeze_embedder:
            with torch.no_grad():
                embeddings = self.embedder(inputs)
        else:
            embeddings = self.embedder(inputs)

        logits = self.mlp(embeddings)
        return logits