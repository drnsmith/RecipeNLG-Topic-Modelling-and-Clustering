import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Reduce dimensionality to 2D using t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(reduced_embeddings)

# Plot clusters
plt.figure(figsize=(10, 8))
for label in set(df['cluster_labels']):
    cluster_points = tsne_embeddings[df['cluster_labels'] == label]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {label}")
plt.title("t-SNE Visualisation of Recipe Clusters")
plt.legend()
plt.show()
