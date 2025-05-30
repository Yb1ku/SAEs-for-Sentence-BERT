import torch
import wandb
import importlib
import utils
importlib.reload(utils)
from utils import wait_for_gpu_cooldown
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from sae import JumpReLUSAE
import seaborn as sns
import warnings
from torch.utils.data import DataLoader
import heapq
from keybert import KeyBERT
from sentence_transformers import util
from collections import defaultdict
from itertools import combinations
from math import log
import json



class Explainer:
    def __init__(self, cfg, column: str = "text"):
        self.cfg = cfg
        self.model = SentenceTransformer(cfg["model_name"]).to(cfg["device"])
        self.column = column
        self.artifact_path = cfg["artifact_path"]

    def _load_dataset(self, streaming=True):
        """
        Load the dataset from the specified path and filter invalid examples.
        """
        if streaming:
            dataset = load_dataset(self.cfg["dataset_path"], split="train", streaming=True)
        else:
            dataset = load_dataset(self.cfg["dataset_path"], split="train")

        filtered_dataset = [example for example in dataset if
                            self.column in example and example[self.column] is not None]
        return filtered_dataset

    def load_sae(self):
        """
        Load the SAE model from the artifact path.
        """
        run = wandb.init()
        artifact = run.use_artifact(self.artifact_path, type='model')
        artifact_dir = artifact.download()
        config_path = os.path.join(artifact_dir, 'config.json')
        with open(config_path, 'r') as f:
            sae_config = json.load(f)
        if "dtype" in sae_config and isinstance(sae_config["dtype"], str):
            if sae_config["dtype"] == "torch.float32":
                sae_config["dtype"] = torch.float32
            elif sae_config["dtype"] == "torch.float16":
                sae_config["dtype"] = torch.float16
        sae = JumpReLUSAE(sae_config).to(sae_config["device"])
        sae.load_state_dict(torch.load(os.path.join(artifact_dir, 'sae.pt')))

        self.sae = sae
        return self.sae

    def plot_threshold_histogram(self, sae_model=None, bins=50):
        """
        Shows a histogram of the thresholds used in the JumpReLU activation function.
        """
        if sae_model is None:
            if not hasattr(self, "sae"):
                raise ValueError("No SAE model loaded. Please, load the model first.")
            sae_model = self.sae

        if not hasattr(sae_model, 'jumprelu'):
            raise TypeError("SAE model must be a JumpReLUSAE instance.")

        with torch.no_grad():
            thresholds = torch.exp(sae_model.jumprelu.log_threshold).cpu().numpy()

        plt.figure(figsize=(8, 5))
        sns.histplot(thresholds, bins=bins, kde=False, color="royalblue", edgecolor="black")
        plt.xlabel("Threshold (exp(log_threshold))")
        plt.ylabel("Number of features")
        plt.title("Histogram of Thresholds in JumpReLU features")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()

    def compute_fire_rate(self, num_examples=100, column=None, save_path=None):
        if not hasattr(self, "sae"):
            raise ValueError("SAE model not loaded. Usa self.load_sae() antes.")

        column = column or self.column
        device = self.sae.cfg["device"]
        dict_size = self.sae.cfg["dict_size"]

        # Cargar y filtrar el conjunto de datos
        dataset = self._load_dataset(streaming=False)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

        if save_path is None:
            save_path = f"{self.sae.cfg['dataset_path']}_fire_rate.npy"

        self.model.eval()
        self.sae.eval()

        feature_count = torch.zeros(dict_size, device=device)
        processed = 0

        for batch in tqdm(dataloader, desc="Processing examples"):
            if processed >= num_examples:
                break

            texts = batch[column]
            if any(text is None for text in texts):
                continue

            embeddings = self.model.encode(texts, convert_to_tensor=True, device=device)

            with torch.no_grad():
                sae_out = self.sae(embeddings)

            feature_acts = sae_out["feature_acts"]
            batch_count = (feature_acts > 0).float().sum(dim=0)
            feature_count += batch_count
            processed += len(texts)

            if processed % 20000 == 0:
                print(f"[INFO] {processed} examples processed.")

            wait_for_gpu_cooldown(threshold=75, cooldown=60)

        feature_fire_rate = 100 * feature_count / processed
        fire_rate_np = feature_fire_rate.cpu().numpy()

        if save_path:
            np.save(save_path, fire_rate_np)
            print(f"[INFO] Fire rate saved in: {save_path}")

        return fire_rate_np

    def get_top_activating_texts(self, num_examples=10000, top_k=10, column=None, save_to_json=False,
                                 json_path="top_activations.json"):
        """
        Get the top activating texts for each feature in the SAE model.
        """
        if not hasattr(self, "sae"):
            raise ValueError("SAE model not loaded. Usa self.load_sae() antes.")

        column = column or self.column
        device = self.sae.cfg["device"]
        dict_size = self.sae.cfg["dict_size"]

        dataset = self._load_dataset(streaming=True)
        self.model.eval()
        self.sae.eval()

        top_activations = [[] for _ in range(dict_size)]

        for i, example in enumerate(dataset):
            if i >= num_examples:
                break

            text = example[column]
            embedding = self.model.encode(text, convert_to_tensor=True).squeeze(0).to(device)

            with torch.no_grad():
                sae_out = self.sae(embedding.unsqueeze(0))
                feature_acts = sae_out["feature_acts"].squeeze(0)  # shape: (dict_size,)

            for j in range(dict_size):
                value = feature_acts[j].item()
                heap = top_activations[j]
                if len(heap) < top_k:
                    heapq.heappush(heap, (value, text))
                else:
                    heapq.heappushpop(heap, (value, text))

            if i % 5000 == 0 and i > 0:
                print(f"[INFO] Processed {i} examples")

            wait_for_gpu_cooldown(threshold=73, cooldown=60)

        if save_to_json:
            print(f"[INFO] Saving top activations to {json_path}...")
            data_to_save = {}
            for j, heap in enumerate(top_activations):
                top_sorted = sorted(heap, key=lambda x: -x[0])  # de mayor a menor activación
                data_to_save[f"feature_{j}"] = [
                    {"activation": round(score, 6), "text": text} for score, text in top_sorted
                ]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=2, ensure_ascii=False)

        return top_activations

    def extract_keywords_per_feature(self,
            top_activations,
            top_n_keywords=3,
            similarity_threshold=0.9,
            bigram_sim_threshold=0.4,
            group_sim_threshold=0.8,
            save_path=None
    ):
        """
        Extract keywords from the top activating texts for each feature.
        """
        kw_model = KeyBERT()
        embedding_model = self.model
        results = []

        for feature_id, heap in enumerate(top_activations):
            keyword_sums = defaultdict(float)
            keyword_counts = defaultdict(int)

            for _, text in heap:
                text_embedding = embedding_model.encode(text, convert_to_tensor=True)

                keywords = kw_model.extract_keywords(text)
                keywords = [(w.lower(), s) for w, s in keywords]
                if not keywords:
                    continue

                words, _ = zip(*keywords)
                embeddings = embedding_model.encode(words, convert_to_tensor=True)

                grouped_indices = set()
                local_groups = []
                for i, (word_i, coef_i) in enumerate(keywords):
                    if i in grouped_indices:
                        continue
                    group = [(word_i, coef_i)]
                    grouped_indices.add(i)
                    for j in range(i + 1, len(keywords)):
                        if j in grouped_indices:
                            continue
                        sim = util.cos_sim(embeddings[i], embeddings[j]).item()
                        if sim >= similarity_threshold:
                            group.append((keywords[j][0], keywords[j][1]))
                            grouped_indices.add(j)
                    local_groups.append(group)

                compact_keywords = []
                for group in local_groups:
                    representative, score = max(group, key=lambda x: x[1])
                    compact_keywords.append((representative, score))

                top_keywords = compact_keywords[:5]
                top_words = [w for w, _ in top_keywords]
                for w1, w2 in combinations(top_words, 2):
                    phrase = f"{w1} {w2}"
                    phrase_embedding = embedding_model.encode(phrase, convert_to_tensor=True)
                    sim_to_text = util.cos_sim(phrase_embedding, text_embedding).item()
                    if sim_to_text >= bigram_sim_threshold:
                        compact_keywords.append((phrase, sim_to_text))

                seen = set()
                for word, coef in compact_keywords:
                    keyword_sums[word] += coef
                    if word not in seen:
                        keyword_counts[word] += 1
                        seen.add(word)

            scored_keywords = {
                word: (keyword_sums[word] / keyword_counts[word]) * log(1 + keyword_counts[word])
                for word in keyword_sums
            }
            sorted_keywords = sorted(scored_keywords.items(), key=lambda x: x[1], reverse=True)
            terms = [term for term, _ in sorted_keywords]

            if terms:
                term_embeddings = embedding_model.encode(terms, convert_to_tensor=True)
            else:
                term_embeddings = None

            clustered = []
            assigned = set()
            for i, term in enumerate(terms):
                if term in assigned:
                    continue
                cluster = [term]
                assigned.add(term)
                for j in range(i + 1, len(terms)):
                    if terms[j] in assigned:
                        continue
                    sim = util.cos_sim(term_embeddings[i], term_embeddings[j]).item()
                    if sim >= group_sim_threshold:
                        cluster.append(terms[j])
                        assigned.add(terms[j])
                clustered.append(cluster)

            aggregated_keywords = []
            for cluster in clustered:
                S = sum(keyword_sums[w] for w in cluster)
                N = sum(keyword_counts[w] for w in cluster)
                if N > 0:
                    score = (S / N) * log(1 + N)
                    rep = max(cluster, key=lambda w: scored_keywords.get(w, 0))
                    aggregated_keywords.append((rep, score))

            sorted_aggregated_keywords = sorted(aggregated_keywords, key=lambda x: x[1], reverse=True)
            top_keywords_final = sorted_aggregated_keywords[:top_n_keywords]

            results.append({
                "feature_id": feature_id,
                "keywords": top_keywords_final
            })

            if (feature_id + 1) % 500 == 0:
                print(f"[INFO] Processed {feature_id + 1} features...")

            wait_for_gpu_cooldown(threshold=73, cooldown=60)

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"[INFO] Keywords saved to {save_path}")

        return results







