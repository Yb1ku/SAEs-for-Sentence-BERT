import torch

def get_default_cfg():
    default_cfg = {
        "seed": 49,
        "batch_size": 32, # Batch size of text embeddings used during SAE training
        "lr": 3e-4,
        "num_examples": int(100e3),
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.999,
        "max_grad_norm": 100000,
        "seq_len": 512,
        "dtype": torch.float32,
        "model_name": "sentence-transformers/paraphrase-mpnet-base-v2",
        "act_size": 768,
        "dict_size": 6144,
        "device": "cuda:0",
        "model_batch_size": 256, # Number of texts processed at once by the model
        "num_batches_in_buffer": 4, # How many num_batches_in_buffer batches to keep in memory
        "dataset_path": "UniverseTBD/arxiv-bit-flip-cs.LG",
        "wandb_project": "sparse_autoencoders",
        "input_unit_norm": True,
        "perf_log_freq": 1000,
        "sae_type": "topk",
        "checkpoint_freq": 10000,
        "n_batches_to_dead": 5,

        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1/32),
        # for jumprelu
        "bandwidth": 0.001,
    }
    default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    cfg["name"] = f"{cfg['model_name']}_{cfg['dict_size']}_{cfg['dataset_path']}"
    return cfg