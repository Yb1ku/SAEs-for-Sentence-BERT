import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset


class ActivationsStoreSBERT:
    def __init__(
        self,
        model,
        cfg: dict,
    ):
        self.model = model
        self.dataset = iter(load_dataset(cfg["dataset_path"], split="train", streaming=True))
        self.model_batch_size = cfg["model_batch_size"]
        self.device = cfg["device"]
        self.num_batches_in_buffer = cfg["num_batches_in_buffer"]
        self.cfg = cfg
        self.tokens_column = "text"
        self.exhausted = False
        self.examples_used = 0
        self.max_examples = cfg["num_examples"]

    def get_batch_tokens(self):
        if self.exhausted:
            raise RuntimeError("Dataset has no examples left. Please reset the dataset iterator.")

        batch_texts = []
        try:
            while len(batch_texts) < self.model_batch_size:
                if self.examples_used >= self.max_examples:
                    self.exhausted = True
                    raise RuntimeError("🛑 Reached max number of allowed examples.")

                sample = next(self.dataset)
                batch_texts.append(sample[self.tokens_column])
                self.examples_used += 1
        except StopIteration:
            print("⚠️ Dataset has run out of examples.")
            self.exhausted = True
            if len(batch_texts) == 0:
                raise RuntimeError("No more data available in the dataset.")

        return batch_texts

    def get_activations(self, texts):
        with torch.no_grad():
            embeddings = self.model.encode(texts, convert_to_tensor=True, device=self.device)
        return embeddings

    def _fill_buffer(self):
        all_embeddings = []
        for _ in range(self.num_batches_in_buffer):
            batch_texts = self.get_batch_tokens()
            embeddings = self.get_activations(batch_texts)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def _get_dataloader(self):
        if not hasattr(self, "activation_buffer"):
            self.activation_buffer = self._fill_buffer()
        return DataLoader(TensorDataset(self.activation_buffer), batch_size=self.cfg["batch_size"], shuffle=True)

    def next_batch(self):
        try:
            return next(self.dataloader_iter)[0]
        except (StopIteration, AttributeError):
            self.activation_buffer = self._fill_buffer()
            self.dataloader = self._get_dataloader()
            self.dataloader_iter = iter(self.dataloader)
            return next(self.dataloader_iter)[0]