# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# The path of the project
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))


# Importing the dataset
dataset = pd.read_csv(os.path.join(__location__, "Mall_customers.csv"))
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'), p=10 , truncate_mode = None)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()


### custom 
branches_heights = []
branches_width_height = []
res = []

# It contains the list of (x_coord_left, y_coord, y_coord, x_coord_right) of the branch
branches_width_height = [[round(dendrogram['dcoord'][i][0], 3),
                          round(dendrogram['dcoord'][i][1], 3),
                          round(dendrogram['dcoord'][i][2], 3),
                          round(dendrogram['dcoord'][i][3], 3)] for i in range(1, len(dendrogram['dcoord']))]

# It contains only the height of eachs branch
branches_heights = [element[1] for element in branches_width_height]

# Sorted
branches_heights_sorted = (sorted(branches_heights))

# Selector is a list that contains the differences between the ordered heights
selector = [branches_heights_sorted[i+1] - branches_heights_sorted[i] for i in range(len(branches_heights) - 1)]
print(selector)
# and we take the maximum of them
temp = max(selector)

# determine the list of the thresholds for making the clusters
ress = [i for i, j in enumerate(selector) if j == temp]


for j in range(0, len(ress)):
    flag = 0
    for i in range(0, len(branches_width_height)):

        if (branches_width_height[i][0] < branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][1] > branches_heights_sorted[ress[j]]*3/2) and not(branches_width_height[i][2] > branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][3] < branches_heights_sorted[ress[j]]*3/2):
            flag = flag+1

        elif (branches_width_height[i][2] > branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][3] < branches_heights_sorted[ress[j]]*3/2) and not(branches_width_height[i][0] < branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][1] > branches_heights_sorted[ress[j]]*3/2):
            flag = flag+1

        elif (branches_width_height[i][2] > branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][3] < branches_heights_sorted[ress[j]]*3/2) and (branches_width_height[i][0] < branches_heights_sorted[ress[j]]*3/2 and branches_width_height[i][1] > branches_heights_sorted[ress[j]]*3/2):
            flag = flag+2
    res.append(flag)


# Training the Hierarchical Clustering model on the dataset
for i in range(0,len(res)):
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(n_clusters = res[i], affinity = 'euclidean', linkage = 'ward')
    y_hc = hc.fit_predict(X)

    # Visualising the clusters
    plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
    plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
    plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
    plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
    plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
    plt.title('Clusters of customers')
    plt.xlabel('Annual Income (k$)')
    plt.ylabel('Spending Score (1-100)')
    plt.legend()
    plt.show()
