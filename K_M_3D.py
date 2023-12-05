import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from random import randint
import os 

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# Importing the dataset
dataset = pd.read_csv(os.path.join(__location__, "wdbc.data"))
X = dataset.iloc[:, [2,5,7]].values


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

#choosing how many Centroids we are going to work with
#automating the elbow method kinda using the second derivative of wcss
selector = [wcss[i] - wcss[i+1] for i in range(len(wcss) - 1)]
selector_2 = [selector[i] - selector[i+1] for i in range(len(selector) - 1)]
temp = min(selector_2)
res = [i for i, j in enumerate(selector_2) if j == temp]

plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Training the possible Hierarchical Clustering models on the dataset
from random import randint
for i in range(0, len(res)):
    from sklearn.cluster import AgglomerativeClustering
    kmeans = KMeans(n_clusters = res[i], init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    for j in range(0, res[i]):
        ax.scatter(X[kmeans.labels_ == j, 0],
                   X[kmeans.labels_ == j, 1],
                   X[kmeans.labels_ == j, 2],
                   s=100, c='#%06X' %randint(0, 0xFFFFFF),
                   label='Cluster' + str(j+1))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
