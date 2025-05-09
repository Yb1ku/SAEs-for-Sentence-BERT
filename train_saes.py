# Adapted from: https://github.com/bartbussmann/BatchTopK
# Original author: Bart Bussmann
# License: MIT
#
# Modifications made by: Yb1ku, 2025

import torch
from training import train_sae
from sae import VanillaSAE, TopKSAE, BatchTopKSAE, JumpReLUSAE
from activation_store import ActivationsStoreSBERT
from config import get_default_cfg, post_init_cfg
from sentence_transformers import SentenceTransformer


def get_experiment_cfg():
    cfg = get_default_cfg()
    cfg.update({
        "sae_type": "jumprelu",
        "model_name": "sentence-transformers/paraphrase-mpnet-base-v2",
        "dataset_path": "UniverseTBD/arxiv-astro-abstracts-all",
        "site": "embed",
        "layer": 0,
        "aux_penalty": 1 / 32,
        "lr": 3e-4,
        "input_unit_norm": True,
        "top_k": 16,
        "top_k_aux": 32,
        "dict_size": 768 * 3,
        "wandb_project": "SBERT-SAEs-csLG",
        "l1_coeff": 0.,
        "act_size": 768,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "bandwidth": 0.001,
    })
    return post_init_cfg(cfg)


def main():
    cfg = get_experiment_cfg()

    sae_classes = {
        "vanilla": VanillaSAE,
        "topk": TopKSAE,
        "batchtopk": BatchTopKSAE,
        "jumprelu": JumpReLUSAE,
    }

    sae_cls = sae_classes.get(cfg["sae_type"])
    if sae_cls is None:
        raise ValueError(f"SAE type '{cfg['sae_type']}' is not supported.")
    sae = sae_cls(cfg)

    model = SentenceTransformer(cfg["model_name"]).to(cfg["device"])
    activation_store = ActivationsStoreSBERT(model, cfg)

    train_sae(sae, activation_store, model, cfg)


if __name__ == "__main__":
    main()
