from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import Model
import os
import numpy as np
import matplotlib.pyplot as plt
from random import randint
import pandas as pd
import pickle
from PIL import Image
import imghdr
import scipy.cluster.hierarchy as sch
import sys

np.set_printoptions(threshold=None)
model = VGG16()
model = Model(inputs = model.inputs, outputs = model.layers[-2].output)


#I leave the model there for future implemendations for now it does not affect anything
def image_datification(file, model, matrify, normalize):
    im = Image.open(file)
    ### Resizing/Retyping file to png
    if print(im.size) != (224,224):
        im = im.resize((224,224))
        im.save(file)
    if imghdr.what(file) != 'png':
        im_type = Image.open(file)
        im_type.save(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG\file1.png')
        im_png = Image.open(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG\file1.png')
    else:
        im_png = Image.open(file)

    ### load the image as a 224x224 array
    img = load_img(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG\file1.png', target_size=(224,224))
    # convert from 'PIL.Image.Image' to numpy array
    img = np.array(img)
    ### reshape the data for the model reshape(num_of_samples, dim 1, dim 2, channels) (future features)
    # reshaped_img = img.reshape(1,224,224,3)
    ### prepare image for model
    # imgx = preprocess_input(reshaped_img)
    # # get the feature vector
    # features = model.predict(imgx, use_multiprocessing=True)
    if matrify == True:
        dataset_np = np.concatenate((img[0],img[1]))
        for i in range(2, 223):
            dataset_np = np.concatenate((dataset_np,img[i]))
        dataset = pd.DataFrame(dataset_np)
        return dataset
    else:
        return img
    if normalize == True:
        dataset = sc_x.fit_transform(img)
        return dataset
    else:
        return img
img = load_img(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG\file1.png', target_size=(224,224))
img = np.array(img)
print(img[223])
print(dataset)
print(len(dataset))
print(dataset_np)

##### TESTING SITE
# print(np.array(load_img(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG\file1.png', target_size=(224,224))))
dataset = image_datification(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\JPG\firstball.jpg', model, matrify=True, normalize=False)
# img = load_img(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG\file1.png', target_size=(224,224))
# img = np.array(img)
# reshaped_img = img.reshape(1,224,224,3)
# print(img[110])
# print(img[223].size)
# print(image_datification(r'C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\JPG\firstball.jpg', model).size)

#we turn the image into a numpy array and then into a panda-one (star wars refference)

# X = dataset.iloc[:,:].values
# print(X)
# print(X.size)


### Custom normalizer attempt
# def scale(X, x_min, x_max):
#     nom = (X-X.min())*(x_max-x_min)
#     denom = X.max() - X.min()
#     denom = denom + (denom == 0)
#     return x_min + nom/denom
# scale(dataset, 0, 1)


### Lib Normalizer attempt
# from sklearn.preprocessing import Normalizer
# sc_x = Normalizer()
# dataset = sc_x.fit_transform(dataset)
# print(dataset)
# print(dataset_np)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')


def truncate(n, decimals=0):
    multiplier = 10 ** decimals
    return int(n * multiplier) / multiplier


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


# dendrogram = sch.dendrogram(sch.linkage(
#     X, method='ward'), p=10, truncate_mode=None)
# plt.title('Dendrogram')
# plt.xlabel('Customers')
# plt.ylabel('Euclidean distances')
# plt.show()
# automating the selection process of the optimal number of clusters using the dendrogram
# there could actualy be more than one optimal number of clusters


# l = []
# l_1 = []
# res = []
# for i in range(1, len(dendrogram['dcoord'])):
#     l.append(truncate(dendrogram['dcoord'][i][1], 3))
#     l_1.append([truncate(dendrogram['dcoord'][i][0], 3), truncate(dendrogram['dcoord'][i][1], 3),
#                truncate(dendrogram['dcoord'][i][2], 3), truncate(dendrogram['dcoord'][i][3], 3)])
# l_sort = (sorted(l))
# selector = [l_sort[i+1] - l_sort[i] for i in range(len(l) - 1)]
# temp = max(selector)
# ress = [i for i, j in enumerate(selector) if j == temp]

# parameter testing
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


# for j in range(0, len(ress)):
#     flag = 0
#     for i in range(0, len(l_1)):
#         if (l_1[i][0] < l_sort[ress[j]]*3/2 and l_1[i][1] > l_sort[ress[j]]*3/2) and not(l_1[i][2] > l_sort[ress[j]]*3/2 and l_1[i][3] < l_sort[ress[j]]*3/2):
#             flag = flag+1
#             # if flag != 0:
#             #     print(flag)
#         elif (l_1[i][2] > l_sort[ress[j]]*3/2 and l_1[i][3] < l_sort[ress[j]]*3/2) and not(l_1[i][0] < l_sort[ress[j]]*3/2 and l_1[i][1] > l_sort[ress[j]]*3/2):
#             flag = flag+1
#             # if flag != 0:
#             #     print(flag)
#         elif (l_1[i][2] > l_sort[ress[j]]*3/2 and l_1[i][3] < l_sort[ress[j]]*3/2) and (l_1[i][0] < l_sort[ress[j]]*3/2 and l_1[i][1] > l_sort[ress[j]]*3/2):
#             flag = flag+2
#             # if flag != 0:
#             #     print(flag)
#     res.append(flag)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# Training the possible Hierarchical Clustering models on the dataset
from random import randint
if len(res) == 1:
    for j in range(0, res[0]):
            clusterAverages = []
            clusterAverages.append(dataset[kmeans.labels_ == j].mean())  #for the clustered image
            print(clusterAverages)
            ax.scatter(dataset[kmeans.labels_ == j][0],
                       dataset[kmeans.labels_ == j][1],
                       dataset[kmeans.labels_ == j][2],
                       s=100, c='#%06X' %randint(0, 0xFFFFFF),
                       label='Cluster' + str(j+1))
else:
    for i in range(0, len(res)-1):
        from sklearn.cluster import AgglomerativeClustering
        kmeans = KMeans(n_clusters = res[i], init = 'k-means++', random_state = 42)
        kmeans.fit(dataset)
        i+1
        for j in range(0, res[i]):
                clusterAverages = []
                clusterAverages.append(dataset[kmeans.labels_ == j].mean())  #for the clustered image
                print(clusterAverages)
                ax.scatter(dataset[kmeans.labels_ == j][0],
                           dataset[kmeans.labels_ == j][1],
                           dataset[kmeans.labels_ == j][2],
                           s=100, c='#%06X' %randint(0, 0xFFFFFF),
                           label='Cluster' + str(j+1))
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
print(np.ones([2,2]))
imgCL = np.ndarray([224,224,3])
for r in range(0, 223):
    for c in range(0, 223):
        imgCL[r][c] = clusterAverages[kmeans.labels_[223*r+c]]
im = Image.fromarray(imgCL)
im.save(r"C:\Users\Pookaros\Desktop\george\ML DL AI\breast cancer\PNG\file1CL")
print(clusterAverages[kmeans.labels_[1]])
# print(dataset[kmeans.labels_ == 1].mean())
print(kmeans.labels_)
print(len(kmeans.labels_))
