import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset from your local path
local_path = r"C:\Users\Owen\Downloads\Mall_Customers.csv"
data = pd.read_csv(local_path)

# Assume 'Annual Income' and 'Spending Score' are relevant features for clustering
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Choose the number of clusters (k)
k = 5
# Explicitly set the value of n_init
kmeans = KMeans(n_clusters=k, n_init=10)  # You can adjust the value as needed
kmeans.fit(X)

# Add the cluster labels to the dataset
data['Cluster'] = kmeans.labels_

# Visualize the clusters
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='rainbow')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.title('Customer Segmentation')
plt.show()
