from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Perform k-means clustering
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster_labels'] = kmeans.fit_predict(reduced_embeddings)

# Evaluate clustering quality
silhouette_avg = silhouette_score(reduced_embeddings, df['cluster_labels'])
print(f"Silhouette Score: {silhouette_avg}")

# Display cluster assignments
print(df[['preprocessed_full_recipe', 'cluster_labels']].head())
