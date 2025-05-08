# 🔧 SAE Training Script

This script defines the training pipeline for various Sparse Autoencoder (SAE) architectures over Sentence-BERT embeddings. It allows configuration of the SAE variant, dataset, training parameters, and model behavior.

---

## 📦 Overview

The script performs the following steps:

1. Load a pre-trained Sentence-BERT model.
2. Stream text data from a dataset.
3. Encode the texts into dense embeddings.
4. Train a selected SAE variant on these embeddings.

---

## ⚙️ Configuration: `get_experiment_cfg()`

The function `get_experiment_cfg()` sets up the experiment configuration. 
It returns a dictionary (`cfg`) with training hyperparameters and system settings:

| Key               | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `sae_type`        | SAE variant to use (`vanilla`, `topk`, `batchtopk`, `jumprelu`)             |
| `model_name`      | Pre-trained SBERT model name (Hugging Face)                                 |
| `dataset_path`    | Hugging Face dataset path (streamed)                                        |
| `dict_size`       | Number of latent features (SAE output dimensionality)                       |
| `top_k`           | Number of active features allowed in the SAE output                         |
| `lr`              | Learning rate                                                               |
| `aux_penalty`     | Auxiliary penalty (e.g., JumpReLU regularization)                           |
| `input_unit_norm` | Whether to normalize input embeddings                                       |
| `device`          | `"cuda"` or `"cpu"`                                                         |
| `wandb_project`   | Project name for Weights & Biases logging                                   |

The config is finalized by `post_init_cfg(cfg)`, which fills in 
derived or default values.

---

## 🧠 SAE Selection

The script dynamically selects the appropriate SAE class:

```python
sae_classes = {
    "vanilla": VanillaSAE,
    "topk": TopKSAE,
    "batchtopk": BatchTopKSAE,
    "jumprelu": JumpReLUSAE,
}
sae = sae_classes[cfg["sae_type"]](cfg)
```
Each SAE receives the same configuration dictionary.


## 📊 Data Loading: `ActivationsStoreSBERT`

The class `ActivationsStoreSBERT` is responsible for efficiently streaming and 
buffering text data as Sentence-BERT embeddings.

It performs the following tasks:

- **Streams raw text** from a Hugging Face dataset using `streaming=True`.
- **Encodes text batches** using a pre-trained Sentence-BERT model.
- **Buffers multiple batches** into memory to simulate a fixed-size dataset.
- **Provides minibatches** of embeddings for training or analysis.

```python
activation_store = ActivationsStoreSBERT(model, cfg)
```
This design enables large-scale training without loading the entire dataset into memory.
It also handles iteration limits and buffer regeneration automatically.