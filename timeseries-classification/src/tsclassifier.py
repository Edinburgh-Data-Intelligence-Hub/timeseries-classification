import numpy as np
import timesfm
import torch 
import torch.nn as nn
import torch.nn.functional as F

from . import utils_timesfm
from momentfm import MOMENTPipeline

class Embedder(nn.Module):
    def __init__(self, name: str, device="cuda"):
        super().__init__()
        self.name = name
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
        
        elif self.name == "MOMENT-1-base": 
            self.model = MOMENTPipeline.from_pretrained(
                 "AutonLab/MOMENT-1-base", 
                 model_kwargs={'task_name': 'embedding'}, # We are loading the model in `embedding` mode to learn representations
                 # local_files_only=True,  # Whether or not to only look at local files (i.e., do not try to download the model).
            )
            
            self.model.init()
            self.embedding_dim = self.model.config.d_model
            self.model.eval()
        
        elif self.name == "vae":
            pass
            self.model = None # fill in with actual VAE model
            self.embedding_dim = None  # fill in with actual dimension
            self.model.to(self.device)
            self.model.eval()  # freeze by default
        
        else:
            raise ValueError(f"Unknown embedder: {self.name}")
    
    
    @torch.no_grad()
    def forward(self, inputs):
        if self.name == "timesfm":
            _, output_embeddings = utils_timesfm.get_embeddings(
                horizon=12,
                model=self.model,
                inputs=inputs,
                layers_to_hook=-1,
            )
            return output_embeddings[0][:,-1,:]
        
        elif self.name == "MOMENT-1-base":
            if isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            if len(inputs.shape) == 0:
                raise ValueError("Input tensor must have at least one dimension")
            elif len(inputs.shape) > 3 :
                raise ValueError("Input tensor must have at most three dimensions (B, C, L), currenlty has shape {inputs.shape}")
            elif len(inputs.shape) != 3:
                inputs = inputs.reshape(inputs.shape[0], 1, inputs.shape[1]) if len(inputs.shape)==2 else inputs.reshape(1, 1, inputs.shape[0]) # (B, C, L)
                
            with torch.no_grad():
                embedd = self.model(x_enc=inputs)
            
            return embedd.embeddings
        
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
        self.hidden_dims = hidden_dims

        self.mlp = EmbeddingMLP(
            input_dim=self.embedder.embedding_dim,
            hidden_dims=self.hidden_dims,
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
    
def save_tsclassifier(
    model: tsClassifier,
    path: str,
):
    checkpoint = {
        "mlp_state_dict": model.mlp.state_dict(), 
        "config": {
            "embedder_name": model.embedder.name,
            "num_classes": model.num_classes,
            "hidden_dims": model.mlp_hidden_dims,
            "dropout": model.mlp_dropout,
            "freeze_embedder": model.freeze_embedder,
        },
    }
    
    torch.save(checkpoint, path)

def load_tsclassifier(
    path: str, 
    device: str
) -> tsClassifier:
    
    # Load MLP
    checkpoint = torch.load(path, map_location=device)
    
    # Load embedder
    embedder = Embedder(
        name=checkpoint["config"]["embedder_name"],
        device=device,
    )
    
    # Create tsClassifier
    model = tsClassifier(
        embedder=embedder,
        num_classes=checkpoint["config"]["num_classes"],
        hidden_dims=checkpoint["config"]["hidden_dims"],
        dropout=checkpoint["config"]["dropout"],
        freeze_embedder=checkpoint["config"]["freeze_embedder"],
    )
    
    model.mlp.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model