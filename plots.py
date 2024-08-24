import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Visualization with PCA and t-SNE
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_encoded)

plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, cmap='coolwarm')
plt.title("PCA of Fraud Detection")
plt.xlabel("PCA1")
plt.ylabel("PCA2")

# t-SNE
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X_test_encoded)

plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_test, cmap='coolwarm')
plt.title("t-SNE of Fraud Detection")
plt.xlabel("t-SNE1")
plt.ylabel("t-SNE2")

plt.show()
