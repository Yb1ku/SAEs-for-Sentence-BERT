# Adapted from: https://github.com/bartbussmann/BatchTopK
# Original author: Bart Bussmann
# License: MIT
#
# Modifications made by: Yb1ku, 2025

import torch
import tqdm
import pynvml
import time
from logs import init_wandb, log_wandb, log_model_performance, save_checkpoint
from utils import wait_for_gpu_cooldown


def train_sae(sae, activation_store, model, cfg):
    num_batches = cfg["num_examples"] // cfg["batch_size"]
    optimizer = torch.optim.Adam(sae.parameters(), lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]))
    pbar = tqdm.trange(num_batches)

    wandb_run = init_wandb(cfg)

    for i in pbar:
        try:
            batch_tokens = activation_store.get_batch_tokens()
        except RuntimeError as e:
            print(f"Training stopped: {e}")
            break

        activations = activation_store.get_activations(batch_tokens)
        sae_output = sae(activations)

        wandb_run.log({"examples_used": activation_store.examples_used}, step=i)

        log_wandb(sae_output, i, wandb_run)
        if i % cfg["perf_log_freq"] == 0:
            log_model_performance(wandb_run, i, model, activation_store, sae, cfg)

        if i % cfg["checkpoint_freq"] == 0:
            save_checkpoint(wandb_run, sae, cfg, i)

        loss = sae_output["loss"]
        postfix = {"Loss": f"{loss.item():.4f}"}
        if "l0_norm" in sae_output:
            postfix["L0"] = f"{sae_output['l0_norm']:.4f}"
        if "l2_loss" in sae_output:
            postfix["L2"] = f"{sae_output['l2_loss']:.4f}"
        if "l1_loss" in sae_output:
            postfix["L1"] = f"{sae_output['l1_loss']:.4f}"
        if "l1_norm" in sae_output:
            postfix["L1_norm"] = f"{sae_output['l1_norm']:.4f}"
        pbar.set_postfix(postfix)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg["max_grad_norm"])
        sae.make_decoder_weights_and_grad_unit_norm()
        optimizer.step()
        optimizer.zero_grad()

        if i % 10 == 0:
            wait_for_gpu_cooldown()


    save_checkpoint(wandb_run, sae, cfg, i)

