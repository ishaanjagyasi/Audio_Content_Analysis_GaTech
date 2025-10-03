import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

data = np.array(
    [
        [2.5, 2.325, 1.8],
        [0.5, 0.325, -0.2],
        [2.2, 2.120, -0.7],
        [1.9, 1.065, 2.1],
        [3.1, 2.735, -0.1],
        [2.3, 2.005, 1.5],
    ]
)

# Normalize the raw numpy array first (z-score per feature)
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# first pass to pandas and plot as-is using seaborn
df = pd.DataFrame(data, columns=["MFCC1", "Flux", "ZCR"])

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x="MFCC1", y="Flux", size="ZCR", sizes=(50, 200))
plt.title("Original Audio Features Visualization\n(ZCR represented by point size)")
plt.xlabel("MFCC1")
plt.ylabel("Flux")
plt.show()

df_normalized = pd.DataFrame(
    data_normalized, columns=["MFCC1_norm", "Flux_norm", "ZCR_norm"]
)


plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df_normalized, x="MFCC1_norm", y="Flux_norm", size="ZCR_norm", sizes=(50, 200)
)
plt.title(
    "Z-Score Normalized Audio Features Visualization\n(ZCR represented by point size)"
)
plt.xlabel("MFCC1 (normalized)")
plt.ylabel("Flux (normalized)")
plt.show()


cov_matrix = np.cov(data_normalized.T)
print("Covariance Matrix:")
print(cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)

# Normalize eigenvalues by their sum (proportion of variance explained)
eigenvalues_sum = np.sum(eigenvalues)
if eigenvalues_sum != 0:
    eigenvalues_normalized = eigenvalues / eigenvalues_sum
else:
    eigenvalues_normalized = eigenvalues

print("\nNormalized Eigenvalues (sum to 1):")
print(eigenvalues_normalized)

# Project z-scored data onto eigenvectors to obtain manual PC scores
# Note: np.linalg.eig returns eigenvectors as columns
scores_manual = data_normalized.dot(eigenvectors)
df_scores_manual = pd.DataFrame(
    scores_manual, columns=["PC1_raw", "PC2_raw", "PC3_raw"]
)

print("\nManual PC Scores (unsorted eigenvectors):")
print(df_scores_manual)

# Sort components by descending eigenvalues for conventional PC1, PC2, ... ordering
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvectors_sorted = eigenvectors[:, sorted_indices]
scores_manual_sorted = data_normalized.dot(eigenvectors_sorted)
df_scores_manual_sorted = pd.DataFrame(
    scores_manual_sorted, columns=["PC1", "PC2", "PC3"]
)

print("\nManual PC Scores (sorted by eigenvalues):")
print(df_scores_manual_sorted)

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_scores_manual_sorted, x="PC1", y="PC2")
plt.title("Manual PCA Projection: PC1 vs PC2 (from eigenvectors)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# PCA using scikit-learn
pca = PCA(
    n_components=3
)  # PCA already arranges the eigenvectors in descending order of eigenvalues so that the first eigenvector explains the most variance
pca.fit(data_normalized)

print("\nPCA Explained Variance Ratio (sklearn):")
print(pca.explained_variance_ratio_)

print("\nPCA Components (sklearn):")
print(pca.components_)

X_pca = pca.transform(data_normalized)
df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2", "PC3"])

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df_pca, x="PC1", y="PC2")
plt.title("PCA Projection (sklearn): PC1 vs PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()
