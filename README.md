<p align="center">
  <img src="docs/assets/banner.png" alt="Project Logo" width="300"/>
</p>

> ⚠️**WARNING** <br>
> This repository is a work in progress and is not yet complete. The code may not be fully functional or tested.
> The repository structure may not be accurate. 

# Master's Thesis – A method for interpreting features in Sparse Autoencoders 

[View full documentation here](https://yb1ku.github.io/SAEs-for-Sentence-BERT/)

This repository contains the code and resources used for the development of a method for interpreting mono-semantic features 
in Sparse Autoencoders. The SAE will take an embedding created by `Sentence-BERT` and obtain mono-semantic features from a specific
context (Machine Learning and Astrophysics). The novelty of this work is the use of a brand-new feature interpretation method 
based on keyword extraction via `KeyBERT`. 

## 📌 Objectives 
- Train a `JumpReLU` Sparse Autoencoder on embeddings obtained from different specific datasets. 
- Analyze the features obtained from the SAE and interpret them using a new method based on `KeyBERT`. 
- Build a codebase which allows easy experimentation with different datasets and hyperparameters. 


## 📁 Repository Structure
```bash 
├── 📖 README.md # This file 

├── 📁 base_files 
  ├── 🐍 config.py 
  ├── 🐍 sae.py 
  ├── 🐍 training.py 
  ├── 🐍 train_saes.py 
  ├── 🐍 activation_store.py 
  ├── 🐍 logs.py  
  ├── 🐍 utils.py 

├── 📁 tutorials 
  ├── 🧪 sae_analysis.ipynb 
``` 

<br>
<br>

---
## 📬 Contact

**Diego Marcos Quirós**  
Master's student at UNIR   
📧 diegomarcosquiros(at)gmail(dot)com  
🔗 https://www.linkedin.com/in/diego-marcos-quirós-803117290/
---
## 🔗 Acknowledgements

This project includes adapted components from the [BatchTopK](https://github.com/bartbussmann/BatchTopK) 
repository by Bart Bussmann, which is licensed under the MIT License.

The original code has been modified to fit the objectives and design of this project.  
[View original license on GitHub](https://github.com/bartbussmann/BatchTopK/blob/main/LICENSE)










