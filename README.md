# **RecipeNLG Topic Modelling and Clustering**

## **Project Overview**
This project conducts an advanced analysis of the **RecipeNLG** dataset by employing a combination of **Latent Dirichlet Allocation (LDA)** and **BERT embeddings** for topic modelling. Through dimensionality reduction techniques and **K-Means clustering**, the aim is to discover and visualize distinct culinary topics, leveraging modern **Natural Language Processing (NLP)** methods to delve deep into the world of recipes.

---

## **Motivation**
Understanding the thematic structure of recipes can provide valuable insights into culinary trends, ingredient pairings, and cultural food practices. This project seeks to:

1. **Uncover Hidden Themes**: Identify underlying topics within a large corpus of recipes.
2. **Enhance Recipe Recommendations**: Improve the relevance of recipe suggestions based on thematic clustering.
3. **Contribute to NLP Research**: Explore the effectiveness of combining traditional and transformer-based models in topic modeling.

---

## **Dataset Description**
- **Source**: The dataset is sourced from the [RecipeNLG](https://recipenlg.cs.put.poznan.pl) project, which offers a large collection of cooking recipes for semi-structured text generation.
- **Structure**:
  - Each recipe includes:
    - **Title**: The name of the recipe.
    - **Ingredients**: A list of components required.
    - **Instructions**: Step-by-step preparation methods.
    - **Metadata**: Additional information such as cuisine type and cooking time.
- **Pre-processing**:
  - **Text Cleaning**: Removal of special characters, numbers, and extraneous whitespace.
  - **Tokenization**: Splitting text into meaningful tokens.
  - **Lemmatization**: Reducing words to their base or root form.

---

## **Methodologies**

### **1. Topic Modeling Approaches**
- **Latent Dirichlet Allocation (LDA)**:
  - A generative probabilistic model that identifies topics by clustering words that frequently co-occur across documents.
- **BERT Embeddings**:
  - Utilizes the **Bidirectional Encoder Representations from Transformers (BERT)** model to generate contextual embeddings for recipes, capturing semantic nuances.

### **2. Dimensionality Reduction Techniques**
- **Principal Component Analysis (PCA)**:
  - Reduces the dimensionality of data while preserving as much variance as possible.
- **t-Distributed Stochastic Neighbor Embedding (t-SNE)**:
  - Converts high-dimensional data into two or three dimensions for visualization, focusing on local data structure.
- **Uniform Manifold Approximation and Projection (UMAP)**:
  - Preserves both local and global data structure in the reduced dimensional space, facilitating effective clustering.

### **3. Clustering Algorithm**
- **K-Means Clustering**:
  - Partitions data into **K** clusters by minimizing the variance within each cluster.

### **4. Evaluation Metric**
- **Silhouette Score**:
  - Measures how similar an object is to its own cluster compared to other clusters, indicating the effectiveness of clustering.

---
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

### Contributing
Contributions are welcome! If you have ideas or improvements to share, please follow these steps:

1. **Fork the Repository:**
Create your own copy of the repository by clicking the "Fork" button at the top right of this page.

2. **Create a Feature Branch:**
Work on your changes in a dedicated branch.

```bash
git checkout -b feature/YourFeatureName
```
3. **Commit Your Changes:**
Write clear and concise commit messages explaining what you’ve done.

```bash
git commit -m "Add YourFeatureName"
```
4. **Push Your Changes:**
Push your feature branch to your forked repository.
```bash
git push origin feature/YourFeatureName
```
5. **Open a Pull Request:**
Submit your changes to the main repository by opening a pull request (PR). Ensure your PR description explains your changes clearly.

6. **Review and Feedback:**
I will review your PR and may suggest improvements before merging it into the main branch.

Thank you for your interest in contributing!

### License
Distributed under the MIT License. See `LICENSE` for more information.

### Acknowledgement
RecipeNLG Dataset: [https://recipenlg.cs.put.poznan.pl/dataset](https://recipenlg.cs.put.poznan.pl/dataset)

**Note**: This README provides a high-level overview. Detailed code can be found in the GitHub repository.



