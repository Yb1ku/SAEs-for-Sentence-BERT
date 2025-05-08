> **Note**: The code implemented in this section is based on the implementation 
> available [here](https://github.com/bartbussmann/BatchTopK). 
> For this project, I have only modified it to adapt it to the `SentenceBERT` model. 

The `ActivationsStoreSBERT` class is a utility for efficiently extracting 
and buffering dense `Sentence-BERT` embeddings from a streaming dataset. 
It is used as an intermediate component between a sentence encoder and a 
sparse autoencoder.

## 🧠 Purpose

Its main goal is to:
- Stream examples from a large dataset.
- Encode them using a Sentence-BERT model.
- Buffer multiple batches of embeddings into memory.
- Serve these embeddings as mini-batches for training or analysis.

This is particularly useful when working with large-scale streaming datasets 
(e.g. Hugging Face `load_dataset(..., streaming=True)`) that do not fit in memory.

## ⚙️ Configuration Parameters

| Argument                | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| `model`                 | A Sentence-BERT model with `.encode()` method (e.g. from `sentence-transformers`). |
| `cfg`                   | A dictionary containing the following keys:                                |
| `dataset_path`          | Path to the Hugging Face dataset to load.                                  |
| `model_batch_size`      | Number of examples processed at once by the SBERT model.                   |
| `device`                | `"cuda"` or `"cpu"` device for embedding extraction.                        |
| `num_batches_in_buffer` | Number of batches to pre-load and concatenate into an activation buffer.    |
| `num_examples`          | Total number of examples to extract. Acts as a cap to avoid exhausting memory.|

## 🧪 Main Methods

### `get_batch_tokens()`

Streams a batch of raw text examples from the dataset.  
Raises an error if the dataset is exhausted or the example cap is reached.

---

### `get_activations(texts)`

Encodes a list of texts into dense embeddings using the SBERT model.

---

### `_fill_buffer()`

Fills an internal activation buffer with multiple batches of embeddings.  
This is used to simulate an in-memory dataset for later sampling.

---

### `_get_dataloader()`

Creates a PyTorch `DataLoader` over the activation buffer for minibatch training or analysis.

---

### `next_batch()`

Returns the next minibatch of SBERT embeddings. If the internal buffer is exhausted, it is automatically refreshed by re-encoding new examples.

---

## 🔁 Lifecycle and Usage

Typical usage pattern:
```python
store = ActivationsStoreSBERT(model=sbert, cfg=config)
batch = store.next_batch()  # returns a tensor of SBERT embeddings