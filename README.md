# NLP-Driven Recipe Clustering: Topic Modelling with BERT and LDA

## **Overview**
This project explores the **RecipeNLG** dataset using a combination of **Latent Dirichlet Allocation (LDA)** and **BERT embeddings** for advanced topic modelling. By employing dimensionality reduction techniques and **K-Means clustering**, the goal is to uncover and visualise distinct culinary themes, leveraging state-of-the-art **Natural Language Processing (NLP)** methods to analyse recipes.

---

## **Motivation**
Understanding the thematic structure of recipes provides valuable insights into culinary trends, ingredient pairings, and cultural food practices. The primary objectives of this project are to:

1. **Identify Hidden Themes**: Detect underlying topics within a large corpus of recipes.
2. **Improve Recipe Recommendations**: Enhance the relevance of recipe suggestions using thematic clustering.
3. **Advance NLP Research**: Evaluate the effectiveness of combining traditional and transformer-based models for topic modelling.

---

## **Dataset Description**

- **Source**: The dataset originates from the [RecipeNLG](https://recipenlg.cs.put.poznan.pl) project, which contains a vast collection of cooking recipes for semi-structured text generation.
- **Structure**:
  - Each recipe includes:
    - **Title**: The recipe name.
    - **Ingredients**: A list of required components.
    - **Instructions**: Step-by-step preparation methods.
    - **Metadata**: Additional details such as cuisine type and cooking time.
- **Pre-processing**:
  - **Text Cleaning**: Removing special characters, numbers, and extraneous whitespace.
  - **Tokenisation**: Splitting text into meaningful tokens.
  - **Lemmatisation**: Reducing words to their base or root form.

---

## **Methodologies**

### **1. Topic Modelling Techniques**

- **Latent Dirichlet Allocation (LDA)**:
  - A probabilistic model that identifies topics by clustering words that frequently co-occur across documents.
- **BERT Embeddings**:
  - Uses the **Bidirectional Encoder Representations from Transformers (BERT)** model to generate contextual embeddings, capturing semantic nuances in recipes.

### **2. Dimensionality Reduction Methods**

- **Principal Component Analysis (PCA)**:
  - Reduces data dimensionality while retaining maximum variance.
- **t-Distributed Stochastic Neighbour Embedding (t-SNE)**:
  - Transforms high-dimensional data into two or three dimensions for visualisation, focusing on local data structure.
- **Uniform Manifold Approximation and Projection (UMAP)**:
  - Balances the preservation of local and global data structure in reduced dimensionality, aiding clustering.

### **3. Clustering Algorithm**

- **K-Means Clustering**:
  - Divides data into **K** clusters by minimising intra-cluster variance.

### **4. Evaluation Metric**

- **Silhouette Score**:
  - Assesses how similar an object is to its cluster compared to other clusters, providing an effectiveness measure for clustering.

---

## **Dependencies**

Ensure you have the following packages installed:

- `scikit-learn`
- `numpy`
- `pandas`
- `gensim`
- `ast`
- `transformers`
- `umap-learn`

---

## **Setup and Installation**

1. **Clone the repository**:
    ```bash
    git clone [Your Repo URL]
    ```

2. **Install the required packages**:
    Create a `requirements.txt` file or directly install dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

---

## **Usage**

To perform topic modelling and clustering, follow these steps:

1. **Load and Pre-process the Dataset**: Prepare the RecipeNLG dataset by cleaning and processing the data.
2. **Tokenise Recipes**: Tokenise the recipes for topic modelling using the BERT tokenizer.
3. **Apply Topic Modelling**:
   - Generate topic distributions using LDA and BERT-based modelling.
4. **Visualise Topics**: Use PCA, t-SNE, and UMAP for dimensionality reduction and visualisation.
5. **Cluster Topics**: Apply K-Means clustering to the visualised data.
6. **Evaluate Clustering Performance**: Assess clustering using the silhouette score.

---

### **Contributing**

Contributions are welcome! To contribute, please follow these steps:

1. **Fork the Repository**: Create a personal copy of the repository by clicking the "Fork" button on GitHub.
2. **Create a Feature Branch**: Work on changes in a dedicated branch.
    ```bash
    git checkout -b feature/YourFeatureName
    ```
3. **Commit Your Changes**: Write clear and concise commit messages detailing your updates.
    ```bash
    git commit -m "Add YourFeatureName"
    ```
4. **Push Your Changes**: Push the feature branch to your forked repository.
    ```bash
    git push origin feature/YourFeatureName
    ```
5. **Open a Pull Request**: Submit changes to the main repository by opening a pull request (PR). Provide a detailed description of your updates.

---
## Repository History Cleaned

As part of preparing this repository for collaboration, its commit history has been cleaned. This action ensures a more streamlined project for contributors and removes outdated or redundant information in the history. 

The current state reflects the latest progress as of 24/01/2025.

For questions regarding prior work or additional details, please contact the author.

---


### **License**

This project is licensed under the MIT License. For more details, see the `LICENSE` file.

---

### **Acknowledgements**

Special thanks to the RecipeNLG dataset creators. You can access the dataset [here](https://recipenlg.cs.put.poznan.pl/dataset).





