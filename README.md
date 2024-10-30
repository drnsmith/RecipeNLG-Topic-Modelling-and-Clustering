# RecipeNLG Topic Modeling and Clustering

Explore the RecipeNLG dataset through advanced topic modeling, leveraging the power of both traditional Latent Dirichlet Allocation (LDA) and state-of-the-art BERT embeddings. This project aims to uncover and visualize unique culinary topics by combining modern NLP techniques with dimensionality reduction and clustering.

## Overview

This repository presents an analysis of the RecipeNLG dataset using a dual approach to topic modeling:

1. **Traditional LDA (Latent Dirichlet Allocation)**
2. **BERT Embeddings**

### Key Steps in the Analysis

- **Data Preprocessing and Tokenization:** Recipes are preprocessed and tokenized for topic modeling.
- **Topic Modeling with LDA and BERT:** Both approaches are applied to generate topic distributions.
- **Dimensionality Reduction for Visualization:** The topic distributions are reduced to 2D using PCA, t-SNE, and UMAP.
- **Clustering with K-Means:** The reduced data is clustered, and the performance is evaluated using the Silhouette Score.

## Dependencies

Ensure you have the following packages installed:
- `sklearn`
- `numpy`
- `pandas`
- `gensim`
- `ast`
- `transformers`
- `umap-learn`

## Setup and Installation

1. **Clone the repository**:
    ```bash
    git clone [Your Repo URL]
    ```

2. **Install the required packages**:
    Create a `requirements.txt` file or directly install the packages using:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To perform topic modelling and clustering with this project, follow these steps:

1. **Load and Preprocess the Dataset**: Prepare the RecipeNLG dataset for analysis by loading and preprocessing the data.
2. **Tokenize Recipes**: Tokenize the recipes for topic modeling using the BERT tokenizer.
3. **Apply Topic Modelling**:
   - Generate topic distributions using LDA and BERT-based topic modeling.
4. **Visualise Topics**: Use PCA, t-SNE, and UMAP to reduce dimensionality for easier visualization.
5. **Cluster Topics**: Apply KMeans clustering to the visualized data.
6. **Evaluate Clustering Performance**: Assess clustering using the silhouette score.

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss proposed modifications.

## License

This project is licensed under the MIT License.


## Decoding Culinary Narratives with Transformer Embeddings and Topic Modelling


## Introduction

In today’s data-rich world, the volume of textual information is expanding rapidly, and Natural Language Processing (NLP) techniques play a critical role in deriving meaningful insights. This study focuses on analyzing recipes, integrating **Bidirectional Encoder Representations from Transformers (BERT)** and **Latent Dirichlet Allocation (LDA)** to uncover the rich, thematic structure of culinary content.

This blog will walk you through the methodologies used to semantically cluster recipes and discover insights into culinary preferences and themes.

## Background: Topic Modeling and Semantic Similarity

**Topic Modeling** is an NLP technique used to identify hidden themes within large text datasets. Traditional topic modeling techniques, such as LDA, have been widely used to uncover thematic structures. However, with the advent of transformer-based models like BERT, we now have the capability to capture deeper semantic meanings and context.

In this study, we combined BERT embeddings with LDA to enhance the precision and granularity of topic modeling, especially within the context of **Recipe Analysis**.

## Methodology: Recipe Analysis Using BERT and LDA

Our approach follows a hybrid framework, using **BERT embeddings** to represent semantic similarity and **clustering algorithms** to discover thematic groupings. Here’s an overview of the process:

### 1. Dataset Selection and Pre-processing
   - The **RecipeNLG dataset**, a large collection of recipes, was chosen for this study. We focused on a subset, ensuring quality while keeping the dataset manageable.
   - Pre-processing involved tokenization, stop-word removal, and stemming, followed by embedding the recipes using the **BERT model**.

### 2. Embedding Generation with BERT
   - Using the `bert-base-uncased` model, each recipe’s text was transformed into a dense vector, capturing the context and nuances of ingredients and preparation methods.
   - We applied cosine similarity to these embeddings, calculating how semantically close recipes were to each other.

### 3. Topic Modeling with LDA and Dimensionality Reduction
   - LDA, known for its probabilistic approach to topic generation, was applied to the dataset to assign thematic structures to the recipe embeddings.
   - **Dimensionality reduction** techniques such as **PCA**, **t-SNE**, and **UMAP** were used to visualize clusters, enabling more effective clustering of recipes based on ingredients and preparation style.

### 4. Clustering and Evaluation
   - We used **k-Means Clustering** to group similar recipes and evaluated the quality of clusters using various metrics, including **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index**.

## Key Findings: Clustering Culinary Data

The results showed distinct clusters that correlated with specific culinary themes. Here are some insights from each cluster:

### Cluster Insights
1. **Dessert Recipes**:
   - Clustered by ingredients like chocolate, sugar, and butter, this group featured baking essentials, highlighting recipes for cookies, cakes, and pastries.

2. **Savory Main Dishes**:
   - With ingredients such as onion, garlic, and pepper, this cluster included roasted and braised dishes, rich in diverse meats and vegetables.

3. **Breakfast & Brunch Items**:
   - Eggs, cheese, and breakfast essentials characterized this group, suggesting popular brunch items such as omelets and hashes.

4. **Spicy and International Dishes**:
   - Keywords like chili, cumin, and curry highlighted a cluster focused on international cuisine, including Mexican and Indian dishes.

5. **Cocktails and Beverages**:
   - This group, defined by ingredients such as lime, rum, and tequila, offered insights into cocktail recipes and refreshing beverages.

The analysis underscores the power of combining BERT embeddings with LDA. While BERT captures semantic nuances, LDA helps categorize the data into coherent thematic clusters.

## Visualization of Clusters

Using **t-SNE** and **UMAP**, we visualized the clusters, revealing patterns and themes within the recipe data. The visualization provided an intuitive understanding of the diverse culinary themes, from appetizers to desserts.

## Conclusion

Our approach demonstrates the advantages of integrating transformer embeddings with traditional topic modeling for culinary data. The combination of BERT and LDA not only enhances the accuracy of topic identification but also provides a nuanced understanding of culinary themes. This method has potential applications in:
   - **Content Recommendation Systems**: Providing tailored recipe suggestions.
   - **Food Industry Research**: Identifying trends and consumer preferences.
   - **Culinary Blogging and Education**: Offering insights into culinary themes for educational purposes.

The study opens the door to further research into hybrid models, which can deepen the analysis of vast datasets beyond culinary applications, enabling refined recommendations and enriched insights across diverse domains.

---

### References

1. RecipeNLG Dataset: [https://recipenlg.cs.put.poznan.pl/dataset](https://recipenlg.cs.put.poznan.pl/dataset)
2. Transformer-based Embeddings and Applications in NLP
3. Clustering and Dimensionality Reduction in High-Dimensional Data Analysis

---

Stay tuned for more insights on NLP applications and further advancements in hybrid topic modeling techniques. For questions, feel free to reach out!


