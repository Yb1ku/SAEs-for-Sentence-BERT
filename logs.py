import wandb
import torch
import os
import re
import json
import torch.nn.functional as F


def init_wandb(cfg):
    return wandb.init(project=cfg["wandb_project"], name=cfg["name"], config=cfg, reinit=True)

def log_wandb(output, step, wandb_run, index=None):
    metrics_to_log = ["loss", "l2_loss", "l1_loss", "l0_norm", "l1_norm", "aux_loss", "num_dead_features"]
    log_dict = {k: output[k].item() for k in metrics_to_log if k in output}
    log_dict["n_dead_in_batch"] = (output["feature_acts"].sum(0) == 0).sum().item()

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)


@torch.no_grad()
def log_model_performance(wandb_run, step, model, activations_store, sae, cfg, index=None, batch_tokens=None):
    """
    Registers the model's performance metrics with SentenceTransformer:
    calculates reconstruction degradation by measuring the MSE between original and reconstructed embeddings.
    """

    if batch_tokens is None:
        batch_tokens = activations_store.get_batch_tokens()

    batch = activations_store.get_activations(batch_tokens)

    sae_out = sae(batch)["sae_out"]

    reconstr_loss = F.mse_loss(sae_out, batch).item()

    log_dict = {
        "performance/mse_reconstruction": reconstr_loss,
    }

    if index is not None:
        log_dict = {f"{k}_{index}": v for k, v in log_dict.items()}

    wandb_run.log(log_dict, step=step)

def save_checkpoint(wandb_run, sae, cfg, step):
    safe_name = re.sub(r"[^\w\.-]", "_", cfg["name"])
    save_dir = f"checkpoints/{cfg['name']}_{step}"
    os.makedirs(save_dir, exist_ok=True)

    sae_path = os.path.join(save_dir, "sae.pt")
    torch.save(sae.state_dict(), sae_path)

    json_safe_cfg = {}
    for key, value in cfg.items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    artifact = wandb.Artifact(
        name=f"{safe_name}_{step}",
        type="model",
        description=f"Model checkpoint at step {step}",
    )
    artifact.add_file(sae_path)
    artifact.add_file(config_path)
    wandb_run.log_artifact(artifact)

    print(f"Model and config saved as artifact at step {step}")
