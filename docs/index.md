# <!-- intentionally left blank -->

<figure style="text-align: center;" markdown>
  <img src="assets/banner.png" alt="Logo" width="500"/>
  <figcaption style="margin-top: 0.5em; font-style: italic;">
    Extracting mono-semantic features from Sentence-BERT
  </figcaption>
</figure>


> ⚠️**WARNING** <br>
> This page is still under construction. 

This is the result of a Master's Thesis project which aims to develop a method for interpreting features 
in Sparse Autoencoders (SAEs) trained on Sentence-BERT embeddings. It is developed 100% in Python, using 
PyTorch as the main framework for the implementation of the SAEs. The project is based on the BatchTopK 
repository by Bart Bussmann, which provides a foundation for the implementation of the SAEs. Click
[here](https://github.com/bartbussmann/BatchTopK) to access the original repository. 

This codebase exists to provide a simple environment for:

- Training Sparse Autoencoders (SAEs) on Sentence-BERT embeddings. 
- Analyzing the features obtained from the SAEs. 
- Present a method for interpreting mono-semantic features in SAEs using keyword extraction via KeyBERT. 

First, you'll need to install the requirements: 
```bash 
pip install -r requirements.txt 
```
### Load a Sparse Autoencoder 
The following snippet shows how to load a Sparse Autoencoder from a wandb project. 
```python 
from sentence_transformers import SentenceTransformer
from config import get_default_cfg
from transformers import pipeline
from sae import JumpReLUSAE
import wandb
import torch
import json
import os

sbert = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
cfg = get_default_cfg()

run = wandb.init()
artifact = run.use_artifact('path_to_your_artifact', type='model')
artifact_dir = artifact.download()
config_path = os.path.join(artifact_dir, 'config.json')
with open(config_path, 'r') as f:
    config = json.load(f)

if "dtype" in config and isinstance(config["dtype"], str):
    if config["dtype"] == "torch.float32":
        config["dtype"] = torch.float32
    elif config["dtype"] == "torch.float16":
        config["dtype"] = torch.float16

sae = JumpReLUSAE(config).to(config["device"])
sae.load_state_dict(torch.load(os.path.join(artifact_dir, 'sae.pt')))
```
Once the model is loaded, you can use it to obtain the features from a specific text. 

## Structure of the documentation


