# SAEs for Sentence-BERT documentation
> ⚠️**WARNING** <br>
> This page is still under construction. 

This codebase exists to provide a simple environment for : 
- Training Sparse Autoencoders (SAEs) on Sentence-BERT embeddings. 
- Analyzing the features obtained from the SAEs. 

### Quick start 
Install the requirements: 
```bash 
pip install -r requirements.txt 
```
### Load a Sparse Autoencoder from the wandb public project 
```python 
from config import get_default_cfg 
import importlib
import explainer
import config
import utils
from explainer import Explainer

cfg = get_default_cfg()
cfg["artifact_path"] = 'ybiku-unir/SBERT-SAEs-csLG/sentence-transformers_paraphrase-mpnet-base-v2_blocks.0.hook_embed_2304_jumprelu_16_0.0003_389:v2'

explainer = Explainer(cfg)
sae = explainer.load_sae()
```
### Calculate feature fire rates, extract the top-k activating texts and obtain the feature keywords 
```python 
fire_rate = explainer.compute_fire_rate(column="text", save_path="fire_rate_test.npy")
top_activations = explainer.get_top_activating_texts(
    num_examples = 100,
    top_k = 2,
    save_to_json = True
)
keywords = explainer.extract_keywords_per_feature(
    top_activations, top_n_keywords = 3
)
```