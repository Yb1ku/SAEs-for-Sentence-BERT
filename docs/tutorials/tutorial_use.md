# How to use a SAE for analyzing a dataset 

Once you have trained a SAE and saved it as an artifact in your Weights & Biases project, 
you can use it to analyze a dataset. This section will guide you through the process. 

## 📲 Load the SAE artifact 
First, import the necessary libraries and load the SAE artifact.
```python 
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from config import get_default_cfg
from sae import JumpReLUSAE
import wandb
import os
import json
import torch
import numpy as np 

sbert = SentenceTransformer('sentence-transformers/paraphrase-mpnet-base-v2')
distilbert = pipeline("fill-mask", model="distilbert/distilbert-base-cased")
cfg = get_default_cfg()

run = wandb.init()
artifact = run.use_artifact('ybiku-unir/SBERT-SAEs-csLG/sentence-transformers_paraphrase-mpnet-base-v2_blocks.0.hook_embed_2304_jumprelu_16_0.0003_389:v0', type='model')
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
The script above will use the configuration stored in the artifact to load the model. This way, you will
not need to worry about defining the model architecture again. 

## 📊 Load the dataset 
Next, you need to create the dataset you want to analyze. It needs to be a custom class 
which inherits from the `IterableDataset` class. The following snippet shows an example 
of how to do it: 
```python 
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset

class HFDatasetWrapper(IterableDataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset

    def __iter__(self):
        for item in self.hf_dataset:
            if item is not None and item.get("text"):
                yield item

def collate_fn_skip_none(batch):
    return [item for item in batch if item is not None]

hf_dataset = load_dataset("UniverseTBD/arxiv-astro-abstracts-all", split="train", streaming=True)
dataset = HFDatasetWrapper(hf_dataset)

num_examples = config["num_examples"]
device = config["device"]
dict_size = config["dict_size"]

sbert = sbert.to(device)
sae = sae.to(device)
sae.eval()

dataloader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn_skip_none)
```
With this code, you are now ready to analyze the dataset. 

## 🔍 Get feature density histogram 
One of the indicators of whether the SAE has a good sparsity is the feature density
histogram. To get the feature activations, you can run the following snippet: 
```python 
feature_count = torch.zeros(dict_size, device=device)
processed = 0

for batch in dataloader:
    if processed >= num_examples:
        break

    texts = [item["text"] for item in batch]
    embeddings = sbert.encode(texts, convert_to_tensor=True, device=device)

    with torch.no_grad():
        sae_out = sae(embeddings)

    feature_acts = sae_out["feature_acts"]
    batch_count = (feature_acts > 0).float().sum(dim=0)
    feature_count += batch_count
    processed += len(texts)

    if processed % 20000 == 0:
        print(f"[INFO] {processed} examples processed.")

    temp = get_gpu_temperature()
    if temp is not None and temp >= 74:
        wait_for_gpu_cooling(threshold=74, resume_temp=60, check_interval=10)

feature_fire_rate = 100 * feature_count / num_examples
np.save("fire_rate_astro.npy", feature_fire_rate.cpu().numpy())
```
the `.npy` file will contain the number of times each feature was activated. If you 
want to plot the histogram, you can use the following code: 
```python 
import matplotlib.pyplot as plt 
log_feature_fire_rate = torch.log10((feature_fire_rate / 100) + 1e-10).cpu().numpy()

plt.style.use('default')
plt.hist(log_feature_fire_rate, bins=50, color='tab:blue')
plt.xlabel("Log10 Feature density")
plt.ylabel("Number of features")
plt.title("Log Feature Fire Rate Distribution (csLG)")
plt.xlim(-10, 0)
plt.show()
```
Features with a 0% fire rate will be placed in the -10 value in the x-axis, and 
features with a 100% fire rate will be placed in the 0 value. For further details on 
how to interpret the histogram, I highly recommend reading 
[this post](https://www.alignmentforum.org/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream). 



## 📑 Get the top-10 activating texts 
To generate the descriptions for the features, first you need to get examples of the 
texts which most activated each feature. You can do this by running the following code: 
```python 
import heapq

num_features = config["dict_size"]
top_activations = [[] for _ in range(num_features)]

for i, example in enumerate(dataset):
    if i >= num_examples: break
    text = example["abstr"]
    embedding = sbert.encode(text, convert_to_tensor=True).squeeze(0).to(config["device"])
    with torch.no_grad():
        sae_out = sae(embedding)
    feature_acts = sae_out["feature_acts"]

    for j in range(num_features):
        activation_value = feature_acts[j].item()
        heap = top_activations[j]
        if len(heap) < 10: # Change this to the number of examples you want 
            heapq.heappush(heap, (activation_value, text))
        else:
            heapq.heappushpop(heap, (activation_value, text))

    if i % 5000 == 0:
        print(f"Processed {i} examples")


top_activations = [sorted(heap, key=lambda x: x[0], reverse=True) for heap in top_activations]
with open("top_activations_astro.json", "w", encoding="utf-8") as f:
    json.dump(top_activations, f, indent=2, ensure_ascii=False)
```
Once you have the top-k activating texts, you are now redy to generate the descriptions. 















