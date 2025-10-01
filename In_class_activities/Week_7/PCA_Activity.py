import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

# first pass to pandas and plot as-is using seaborn
df = pd.DataFrame(data, columns=["MFCC1", "Flux", "ZCR"])

plt.figure(figsize=(10, 8))
sns.scatterplot(data=df, x="MFCC1", y="Flux", size="ZCR", sizes=(50, 200))
plt.title("Original Audio Features Visualization\n(ZCR represented by point size)")
plt.xlabel("MFCC1")
plt.ylabel("Flux")
plt.show()


scaler = StandardScaler() #scikit-learn library for z score normalization
normalized_data = scaler.fit_transform(df)
df_normalized = pd.DataFrame(
    normalized_data, columns=["MFCC1_norm", "Flux_norm", "ZCR_norm"]
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


cov_matrix = np.cov(normalized_data.T)
print("Covariance Matrix:")
print(cov_matrix)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print("\nEigenvalues:")
print(eigenvalues)

print("\nEigenvectors:")
print(eigenvectors)