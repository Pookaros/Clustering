import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as sch
from random import randint
# Importing the dataset
dataset = pd.read_csv(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\wdbc.data')
X = dataset.iloc[:, [2,5,7]].values
# print(X)


dendrogram = sch.dendrogram(sch.linkage(
    X, method='ward'), p=10, truncate_mode=None)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

# automating the selection process of the optimal number of clusters using the dendrogram
# there could actualy be more than one optimal number of clusters


l = []
l_1 = []

res = []
for i in range(1, len(dendrogram['dcoord'])):
    l.append(truncate(dendrogram['dcoord'][i][1], 3))
    l_1.append([truncate(dendrogram['dcoord'][i][0], 3), truncate(dendrogram['dcoord'][i][1], 3),
               truncate(dendrogram['dcoord'][i][2], 3), truncate(dendrogram['dcoord'][i][3], 3)])
l_sort = (sorted(l))
selector = [l_sort[i+1] - l_sort[i] for i in range(len(l) - 1)]
temp = max(selector)
ress = [i for i, j in enumerate(selector) if j == temp]
# print(ress)
# print(l_sort)
# print(l_sort[ress[0]])
# print(l)
# print(l_1)
# print(selector)
# print(temp)
# print(dendrogram['dcoord'])


for j in range(0, len(ress)):
    flag = 0
    for i in range(0, len(l_1)):
        if (l_1[i][0] < l_sort[ress[j]]*3/2 and l_1[i][1] > l_sort[ress[j]]*3/2) and not(l_1[i][2] > l_sort[ress[j]]*3/2 and l_1[i][3] < l_sort[ress[j]]*3/2):
            flag = flag+1
            # if flag != 0:
            #     print(flag)
        elif (l_1[i][2] > l_sort[ress[j]]*3/2 and l_1[i][3] < l_sort[ress[j]]*3/2) and not(l_1[i][0] < l_sort[ress[j]]*3/2 and l_1[i][1] > l_sort[ress[j]]*3/2):
            flag = flag+1
            # if flag != 0:
            #     print(flag)
        elif (l_1[i][2] > l_sort[ress[j]]*3/2 and l_1[i][3] < l_sort[ress[j]]*3/2) and (l_1[i][0] < l_sort[ress[j]]*3/2 and l_1[i][1] > l_sort[ress[j]]*3/2):
            flag = flag+2
            # if flag != 0:
            #     print(flag)
    res.append(flag)


# Training the Hierarchical Clustering model on the dataset

for i in range(0, len(res)):
    from sklearn.cluster import AgglomerativeClustering
    hc = AgglomerativeClustering(
        n_clusters=res[i], affinity='euclidean', linkage='ward')
    y_hc = hc.fit_predict(X)
    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for j in range(0, res[i]):
        ax.scatter(X[y_hc == j, 0], X[y_hc == j, 1], X[y_hc == j, 2], s=100,
                   c='#%06X' % randint(0, 0xFFFFFF), label='Cluster' + str(j+1))
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    # import mayavi.mlab as mylab
    #     mylab.points3d(X[y_hc == j, 0], X[y_hc == j, 1], X[y_hc == j, 2])
    #     mylab.show()
