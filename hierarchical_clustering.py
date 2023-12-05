# Hierarchical Clustering

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv(r'')
X = dataset.iloc[:, [3, 4]].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'), p=10 , truncate_mode = None)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

#truncate function, rounding up numbers to avoid losing good no. of cluster choices
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

#automating the selection process of the optimal number of clusters using the dendrogram
#there could actualy be more than one optimal number of clusters
l=[]
l_1=[]
res=[]
for i in range(1, len(dendrogram['dcoord'])):
    l.append(truncate(dendrogram['dcoord'][i][1]))
    l_1.append([truncate(dendrogram['dcoord'][i][0]),truncate(dendrogram['dcoord'][i][1]),truncate(dendrogram['dcoord'][i][2]),truncate(dendrogram['dcoord'][i][3])])
l_sort=(sorted(l))
selector = [l_sort[i+1] - l_sort[i] for i in range(len(l) - 1)]
temp = max(selector)
ress = [i for i, j in enumerate(selector) if j == temp]

for j in range(0,len(ress)):
    flag=0
    for i in range(0,len(l_1)-1):
        if l_1[i][0] < l_sort[ress[j]]+1 and l_1[i][1] > l_sort[ress[j]]+1:
            flag=flag+1
            if l_1[i][2] > l_sort[ress[j]]+1 and l_1[i][3] < l_sort[ress[j]]+1:
                flag=flag+1
    res.append(flag)
# print(res)


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
