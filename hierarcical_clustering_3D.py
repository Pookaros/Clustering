import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from random import randint
import os

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def truncate(n:float, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

# returns the unique elements of a list in a list
def unique_elements(list:list) -> list:
    new_list = []
    for element in list:
        if element not in new_list: new_list.append(element)
    return new_list

# returs the number of inique elements in a list
def unique_count(list:list) -> int:
    return len(unique_elements(list))


# Importing the dataset
dataset = pd.read_csv(os.path.join(__location__, "wdbc.data"))
X = dataset.iloc[:, [2,5,7]].values
# print(X)

dendrogram = sch.dendrogram(sch.linkage(X, method='ward'), p=10, truncate_mode=None)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

cluster_no = unique_count(dendrogram['leaves_color_list'])
fig = plt.figure()
ax = fig.add_subplot(projection='3d')




# automating the selection process of the optimal number of clusters using the dendrogram
# there could actualy be more than one optimal number of clusters

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
ress = [i for i, j in enumerate(selector) if j == temp]
print(ress)


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
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters= cluster_no, 
                             metric='euclidean', 
                             linkage='ward')
y_hc = hc.fit_predict(X)
# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for j in range(0, cluster_no):
    ax.scatter(X[y_hc == j, 0], X[y_hc == j, 1], X[y_hc == j, 2], s=100,
            c='#%06X' % randint(0, 0xFFFFFF), label='Cluster' + str(j+1))
            
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
