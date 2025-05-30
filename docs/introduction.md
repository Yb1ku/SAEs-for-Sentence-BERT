# 📘 Introduction

## 🧠 Motivation

One of the main focus of mechanistic interpretability is to understand how language 
models represent the information in their embeddings. The field has mostly focused on 
studying the representations of the residual stream in autoregressive transformers. 
However, little work has been done to understand the representations of other types 
of language models, such as sentence embedding models. Till now, the only paper 
that has studied the representations of sentence embedding models is 
[this one](https://arxiv.org/abs/2408.00657). This project will use the ideas shown in
this paper to study the representations of `Sentence-BERT`, one of (if not the) 
most popular sentence embedding model. More details about the model can be found 
[here](https://arxiv.org/abs/1908.10084). 

One of the biggest obstacles when training a Sparse Autoencoder (SAE), especially 
when having a limited budget, is the amount of compute needed. A SAE is able to 
extract monosemantic features from a specific part of the model, and it can't be used 
in other parts. Because of this, one would need to train a SAE for each part of the 
model, which would need a huge amount of computation and training time. However, if 
we study a sentence embedding model, there is no need to train a SAE for each part of 
the model. Instead, we can train a SAE on the final embeddings and get a monosemantic 
representation of the embedding space used to encode the sentences. 

Once the SAE is trained, the features must be interpreted. A SAE usually has lots of 
features, making it not possible to analyze each one of them manually. Because of that, 
many efforts have been made to develop methods to interpret the features obtained from
the SAEs. The most common method is to call a LLM to generate a description of the 
feature. However, making a huge number of API calls to a LLM is costly, 
which makes it not possible to use this method in a limited budget. This project 
presents a new feature interpretation method that does not require any API calls. It
is based on keyword extraction via `Key-BERT`. 

---

## 🎯 Project Objectives

This project aims to explore the use of Sparse Autoencoders for interpreting 
`Sentence-BERT` embeddings. The main goals are: 

- Train a SAE on `Sentence-BERT` embeddings from different knowledge domains. 
- Present a method to interpret the features obtained from the SAEs. 

---

## 🔍 Scope

This work does not aim to improve downstream task performance, nor to benchmark 
new autoencoder architectures. Instead, it focuses on applying the existing 
architectures to a new domain and developing a method to interpret the features. 

The entire pipeline is modular and open-source, and can be adapted to other embedding 
models or datasets.

---

## 🧭 Documentation Structure



The documentation is organized as follows:

- **Quick start**: A minimal guide to set up the environment and run the pipeline.
- **Introduction to the project**: Motivation, objectives, and context for this research.
- **SAE models**: Overview of the different Sparse Autoencoder architectures used (Vanilla, TopK, JumpReLU).
- **Activation Store**: Details of the component responsible for extracting and buffering SBERT embeddings.
- **Training**: Description of the training pipeline and configuration options for SAE models.
- **Tutorials**: Practical examples and notebooks demonstrating how to use the codebase. 
- **Results**: Analysis of the features obtained from the SAEs and their interpretation. 
---

This documentation is intended to be modular and accessible. Each section can be read independently 
depending on your interest—whether you're focused on model training, interpretability methods, or 
practical application.





















