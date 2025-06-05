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

The documentation is structured as follows: 

- **Introduction to the project**: Provides an overview of the project, its objectives, and the 
methodology used. 

- **SAE models**: Brief introduction to the Sparse Autoencoders used in this project. 

- **Activation Store**: Explanation of the class used to interact with the dataset. 

- **Training**: Provides details on how to train the Sparse Autoencoders. 

- **Tutorials**: 
  - **How to use SAEs**: Tutorial on how to load a SAE, obtain the feature density histogram and the top-k
activating texts of each feature. 
  - **Keyword extraction**: Tutorial on how to implement the scoring method to interpret the features.

- **Results**: Shows the main results and conclusions of the project. 

- **References**: List of academic references used in the project.

- **Contact**: Information on how to contact the author of the project. 
