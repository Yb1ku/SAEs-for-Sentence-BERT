> ⚠️**WARNING** <br>
> This page is still under construction. 

This codebase exists to provide a simple environment for : 
- Training Sparse Autoencoders (SAEs) on Sentence-BERT embeddings. 
- Analyzing the features obtained from the SAEs. 

### Quick start 
Install the requirements: 
```bash 
pip install -r requirements.txt 
```
### Load a Sparse Autoencoder 
The following snippet shows how to load a Sparse Autoencoder from a wandb project. 
```python 
from config import get_default_cfg 
import importlib
import explainer
import config
import utils
from explainer import Explainer

cfg = get_default_cfg()
cfg["artifact_path"] = 'ybiku-unir/SBERT-SAEs-csLG/sentence-transformers_paraphrase-mpnet-base-v2_blocks.0.hook_embed_2304_jumprelu_16_0.0003_389:v2'

explainer = Explainer(cfg)
sae = explainer.load_sae()
```
### Sparse Autoencoder Analysis 
The following snippet shows how to analyze the features obtained from the Sparse Autoencoder. You can 
compute the fire rate of each feature, get the top activating texts for each feature and extract the 
keywords which describe them. 
```python 
fire_rate = explainer.compute_fire_rate(column="text", save_path="fire_rate_test.npy")
top_activations = explainer.get_top_activating_texts(
    num_examples = 100,
    top_k = 2,
    save_to_json = True
)
keywords = explainer.extract_keywords_per_feature(
    top_activations, top_n_keywords = 3
)
```

`top_activations` is a dictionary, each key corresponding to a feature. Inside each key, there is a list 
of tuples of the form `[(activation), text]`. `activation` is the activation produced by the text to the 
feature, and `text` is the text itself. For example, a possible output of `top_activations' could be: 
```bash 
[[2.9942984580993652,
  '  Trust region policy optimization (TRPO) is a popular and empirically\nsuccessful policy search 
  algorithm in Reinforcement Learning (RL) in which a\nsurrogate problem, that restricts consecutive 
  policies to be \'close\' to one\nanother, is iteratively solved. Nevertheless, TRPO has been considered 
  a\nheuristic algorithm inspired by Conservative Policy Iteration (CPI). We show\nthat the adaptive 
  scaling mechanism used in TRPO is in fact the natural "RL\nversion" of traditional trust-region methods 
  from convex analysis. We first\nanalyze TRPO in the planning setting, in which we have access to the 
  model and\nthe entire state space. Then, we consider sample-based TRPO and 
  establish\n$\\tilde O(1/\\sqrt{N})$ convergence rate to the global optimum. Importantly, 
  the\nadaptive scaling mechanism allows us to analyze TRPO in regularized MDPs for\nwhich we prove fast 
  rates of $\\tilde O(1/N)$, much like results in convex\noptimization. This is the first result in RL 
  of better rates when regularizing\nthe instantaneous cost or reward.\n'],
 [2.9936885833740234,
  '  We present a reinforcement learning (RL) framework to synthesize a control\npolicy from a given linear
   temporal logic (LTL) specification in an unknown\nstochastic environment that can be modeled as a Markov
    Decision Process (MDP).\nSpecifically, we learn a policy that maximizes the probability of 
    satisfying\nthe LTL formula without learning the transition probabilities. We introduce a\nnovel 
    rewarding and path-dependent discounting mechanism based on the LTL\nformula such that (i) an optimal 
    policy maximizing the total discounted reward\neffectively maximizes the probabilities of satisfying 
    LTL objectives, and (ii)\na model-free RL algorithm using these rewards and discount factors 
    is\nguaranteed to converge to such policy. Finally, we illustrate the applicability\nof our RL-based 
    synthesis approach on two motion planning case studies.\n']]
```

`keywords` has a similar structure. It is also a dictionary, each key corresponding to a feature. 
Inside each key, there is a list of tuples of the form `[(keyword), score]`. `keyword` is the keyword itself, 
and `score` is the score of the keyword. For example, a possible output of `keywords' could be: 
```bash 
{'feature_id': 2220,
 'keywords': [['reinforcement', 1.620540681324779],
  ['reinforcement learning', 0.9814733623507801],
  ['reward optimization', 0.9179529960192596]} 
``` 



