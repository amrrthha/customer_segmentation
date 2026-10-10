import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# 1. Create output folder
os.makedirs("outputs", exist_ok=True)

# 2. Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("\n--- First 5 Rows ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Select features
features = [
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].copy()

# Handle missing values in selected features
X = X.fillna(X.median())

# 4. Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 5. Find suitable K using Elbow Method
inertias = []
silhouette_scores = []
k_values = range(2, 11)

for k in k_values:
    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    inertias.append(model.inertia_)
    silhouette_scores.append(
        silhouette_score(X_scaled, labels)
    )

# Plot evaluation graphs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(
    list(k_values), inertias, marker="o"
)
axes[0].set_title("Elbow Method")
axes[0].set_xlabel("Number of Clusters (K)")
axes[0].set_ylabel("Inertia")

axes[1].plot(
    list(k_values),
    silhouette_scores,
    marker="o",
    color="green"
)
axes[1].set_title("Silhouette Score")
axes[1].set_xlabel("Number of Clusters (K)")
axes[1].set_ylabel("Silhouette Score")

plt.tight_layout()
plt.savefig("outputs/model_evaluation.png")
plt.show()

print("\n--- Silhouette Scores ---")
for k, score in zip(k_values, silhouette_scores):
    print(f"K = {k}: {score:.4f}")

# 6. Train final K-Means model
# Change K after examining the evaluation graphs.
K = 5

kmeans = KMeans(
    n_clusters=K,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# 7. Visualize customer clusters
centers = scaler.inverse_transform(
    kmeans.cluster_centers_
)

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=90
)

plt.scatter(
    centers[:, 0],
    centers[:, 1],
    color="red",
    marker="X",
    s=250,
    label="Centroids"
)

plt.title("Customer Segmentation Using K-Means")
plt.xlabel("Annual Income (thousands of dollars)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.tight_layout()

plt.savefig("outputs/customer_clusters.png")
plt.show()

# 8. Analyze each cluster
summary = df.groupby("Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending=("Spending Score (1-100)", "mean")
).round(2)

print("\n--- Customer Cluster Summary ---")
print(summary)

# 9. Save results
df.to_csv(
    "outputs/customer_segments.csv",
    index=False
)

summary.to_csv(
    "outputs/cluster_summary.csv"
)

print("\nProject completed successfully!")
print("Results saved inside the outputs folder.")