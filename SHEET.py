import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from random import randint
import pickle
from PIL import Image
import imghdr
from keras.preprocessing.image import load_img




pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)
np.set_printoptions(threshold=None)
global namae

###Andrew's Algorithm for creating a convex hull for the cluster
def convex_hull(points):
    points = sorted(set(points))
    if len(points) <= 1:
        return points
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
#I leave the model there for future implemendations for now it does not affect anything
def image_datification(file, matrify, normalize):
    im = Image.open(file)
    if print(im.size) != (224,224): ### Resizing/Retyping file to png
        im = im.resize((224,224))
        im.save(file)
    if imghdr.what(file) != 'png':
        im_type = Image.open(file)
        namae = str(input("Name the converted folder"))
        im_type.save(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG/' + namae + '.png')
        im_png = Image.open(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG/' + namae + '.png')
    else:
        im_png = Image.open(file)

    ### load the image as a 224x224 array
    img = load_img(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG/' + namae + '.png', target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    ### reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels) (future features)
    # reshaped_img = img.reshape(1,224,224,3)
    ### prepare image for model
    # imgx = preprocess_input(reshaped_img)
    ### get the feature vector
    # features = model.predict(imgx, use_multiprocessing=True)
    if matrify == True:
        dataset_np = np.concatenate((img[0],img[1]))
        for i in range(2, 224):
            dataset_np = np.concatenate((dataset_np, img[i]))
        dataset = pd.DataFrame(dataset_np)
        return dataset
    else:
        return img
    if normalize == True:
        dataset = sc_x.fit_transform(img)
        return dataset
    else:
        return img

### Custom normalizer attempt
def scale(X, x_min, x_max):
    nom = (X-X.min())*(x_max-x_min)
    denom = X.max() - X.min()
    denom = denom + (denom == 0)
    return x_min + nom/denom

### Asimple truncator
def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier

dataset = image_datification(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\JPG\SmallBoatEnginefeat.jpg', matrify=True, normalize=False)
##### TESTING SITE
# print(np.array(load_img(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG\file2.png', target_size=(224,224))))

# img = load_img(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG\file2.png', target_size=(224,224))
# img = np.array(img)
# reshaped_img = img.reshape(1,224,224,3)
# print(img[110])
# print(img[223].size)
# print(image_datification(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\JPG\firstball.jpg', model).size)
# X = dataset.iloc[:,:].values
# print(X)
# print(X.size)
# from sklearn.preprocessing import Normalizer
# sc_x = Normalizer()
# dataset = sc_x.fit_transform(dataset)
# print(dataset)
# print(dataset_np)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(dataset)
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

### parameter testing
# print(X)
# print(ress)
# print(l_sort)
# print(l_sort[ress[0]])
# print(l)
# print(l_1)
# print(selector)
# print(temp)
# print(dendrogram['dcoord'])
# print(res)
# print(len(res))


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# Training the possible Hierarchical Clustering models on the dataset
from random import randint
if len(res) == 1:
    clusterAverages = np.ndarray((res[0],3))
    kmeans = KMeans(n_clusters = res[0], init = 'k-means++', random_state = 42)
    kmeans.fit(dataset)
    for j in range(0, res[0]):
            clusterAverages[j][0] = int(dataset[kmeans.labels_ == j].mean()[0]) #for the clustered image
            clusterAverages[j][1] = int(dataset[kmeans.labels_ == j].mean()[1])
            clusterAverages[j][2] = int(dataset[kmeans.labels_ == j].mean()[2])
            ax.scatter(dataset[kmeans.labels_ == j][0],
                       dataset[kmeans.labels_ == j][1],
                       dataset[kmeans.labels_ == j][2],
                       s=100, c='#%06X' %randint(0, 0xFFFFFF),
                       label='Cluster' + str(j+1))
    clusterAverages = scale(clusterAverages, 0, 1)
else:
    for i in range(0, len(res)-1):
        kmeans = KMeans(n_clusters = res[0], init = 'k-means++', random_state = 42)
        kmeans.fit(dataset)
        clusterAverages = np.ndarray((res[i],1,3))
        from sklearn.cluster import AgglomerativeClustering
        kmeans = KMeans(n_clusters = res[i], init = 'k-means++', random_state = 42)
        kmeans.fit(dataset)
        for j in range(0, res[i]):
            clusterAverages[j][0] = int(dataset[kmeans.labels_ == j].mean()[0]) #for the clustered image
            clusterAverages[j][1] = int(dataset[kmeans.labels_ == j].mean()[1])
            clusterAverages[j][2] = int(dataset[kmeans.labels_ == j].mean()[2])
            ax.scatter(dataset[kmeans.labels_ == j][0],
                       dataset[kmeans.labels_ == j][1],
                       dataset[kmeans.labels_ == j][2],
                       s=100, c='#%06X' %randint(0, 0xFFFFFF),
                       label='Cluster' + str(j+1))
    # clusterAverages = scale(clusterAverages, 0, 1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
imgCL = np.ndarray(([224,224,3]))
for r in range(0, 223):
    for c in range(0, 223):
        imgCL[r][c] = clusterAverages[kmeans.labels_[(224* r+c)]][0]

plt.imsave(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG/' + namae + 'CL.png', imgCL)

##### TESTING SITE
# print(clusterAverages[0][0][0])
# print(imgCL([223][223]))
# print(len(imgCL[0][1]))
# from collections import Counter
# Counter(kmeans.labels_).keys() # equals to list(set(words))
# Counter(kmeans.labels_).values() # counts the elements' frequency
# print(kmeans.labels_)
# print(kmeans.labels_)
# print(int(dataset[kmeans.labels_ == j].mean()[1]))
# print(namae)
# print(clusterAverages[kmeans.labels_[1]])
# print(dataset[kmeans.labels_ == 1].mean())
# print(kmeans.labels)
