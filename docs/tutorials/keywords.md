# Keyword extraction for fature descriptions 

As stated in the introduction, the main focus of this project is to develop a method 
for interpreting the features obtained from a Sparse Autoencoder. The proposed method 
is based on [`KeyBERT`](https://maartengr.github.io/KeyBERT/), a keyword extraction 
library availiable in Python. The idea is to extract the most relevant words from each 
top-k activating text, and implement a scoring system to rank the terms. This tutorial 
will show an example of how to implement this method. More details about the 
theoretical background will be provided soon. 

You can use the following code to get the keywords for each feature. 
```python 
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict
from itertools import combinations
from math import log

kw_model = KeyBERT()
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
SIMILARITY_THRESHOLD = 0.9
BIGRAM_SIM_THRESHOLD = 0.4
GROUP_SIM_THRESHOLD = 0.75

all_keywords_results = []

for feature_id in range(len(top_activations)):
    keyword_sums = defaultdict(float)
    keyword_counts = defaultdict(int)
    original_forms = defaultdict(list)

    for i in range(len(top_activations[feature_id])):
        text = top_activations[feature_id][i][1]
        text_embedding = embedding_model.encode(text, convert_to_tensor=True)

        keywords = kw_model.extract_keywords(text)
        keywords = [(w.lower(), s) for w, s in keywords]
        if not keywords:
            continue

        words, scores = zip(*keywords)
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
                if sim >= SIMILARITY_THRESHOLD:
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
            if sim_to_text >= BIGRAM_SIM_THRESHOLD:
                compact_keywords.append((phrase, sim_to_text))

        seen_in_this_doc = set()
        for word, coef in compact_keywords:
            keyword_sums[word] += coef
            original_forms[word].append((word, coef))
            if word not in seen_in_this_doc:
                keyword_counts[word] += 1
                seen_in_this_doc.add(word)

    scored_keywords = {
        word: (keyword_sums[word] / keyword_counts[word]) * log(1 + keyword_counts[word])
        for word in keyword_sums
    }
    sorted_keywords = sorted(scored_keywords.items(), key=lambda x: x[1], reverse=True)

    terms = [term for term, score in sorted_keywords]
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
            if sim >= GROUP_SIM_THRESHOLD:
                cluster.append(terms[j])
                assigned.add(terms[j])
        clustered.append(cluster)

    aggregated_keywords = []
    for cluster in clustered:
        aggregated_S = sum(keyword_sums[w] for w in cluster)
        aggregated_N = sum(keyword_counts[w] for w in cluster)
        if aggregated_N > 0:
            agg_score = (aggregated_S / aggregated_N) * log(1 + aggregated_N)
            rep = max(cluster, key=lambda w: scored_keywords.get(w, 0))
            aggregated_keywords.append((rep, agg_score))

    sorted_aggregated_keywords = sorted(aggregated_keywords, key=lambda x: x[1], reverse=True)

    all_keywords_results.append({
        "feature_id": feature_id,
        "keywords": sorted_aggregated_keywords
    })

    if (feature_id + 1) % 500 == 0:
        print(f"Processed {feature_id + 1} features...")

    temp = get_gpu_temperature()
    if temp is not None and temp >= 77:
        wait_for_gpu_cooling(threshold=77, resume_temp=65, check_interval=5)
```

For example, if your SAE is trained on Machine Learning texts and it had a RL feature, 
the keywords for that feature could look like this: 
```bash
{'feature_id': 2220,
 'keywords': [['reinforcement', 1.620540681324779],
  ['reinforcement learning', 0.9814733623507801],
  ['reward optimization', 0.9179529960192596]]}
```
