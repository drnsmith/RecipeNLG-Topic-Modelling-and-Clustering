# RecipeNLG-Topic-Modelling-and-Clustering
An advanced analysis of the RecipeNLG dataset employing a unique combination of LDA and BERT embeddings for topic modelling (TM). With dimensionality reduction and K-Means clustering, this project aims to discover and visualise distinct recipe topics. Harness the power of modern NLP techniques to delve deep into the culinary world!

This repository contains code and methods to perform TM on the RecipeNLG dataset using a combined approach of LDA and BERT embeddings. Furthermore, the topics are visualised using dimensionality reduction techniques and clustered to identify groups of similar recipes.

## Overview

The RecipeNLG dataset contains various recipes that are pre-processed, tokenised, and used to perform TM. Two approaches to TM are integrated:

1. Traditional LDA (Latent Dirichlet Allocation)
2. BERT embeddings

After generating the topic distributions, the results are reduced to a 2D space using PCA, t-SNE, and UMAP for visualisation. The reduced data is then clustered using the K-Means algorithm, and the performance of the clustering is evaluated using the Silhouette Score.

## Dependencies

- `sklearn`
- `numpy`
- `pandas`
- `gensim`
- `ast`
- `transformers`
- `umap-learn`

## Usage
 - Load your dataset and pre-process it.
 - Tokenise the recipes for topic modeling using BERT Tokenizer.
 - Apply LDA and BERT-based topic modeling.
 - Visualise the topics using PCA, t-SNE, and UMAP.
 - Cluster the visualised topics using KMeans.
 - Evaluate clustering using the silhouette score.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License
MIT

## Setup and Installation

Clone the repository:
```bash
git clone [Your Repo URL]
nstall the required packages:
bash
Copy code:
pip install -r requirements.txt
(Note: You might want to create a requirements.txt file listing the dependencies.)
