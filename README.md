# Clustering
2-d Hierarcical/k-means++ Clustering with automated cluster selection.
Planning to make it 3-d in the future let's see how that goes.
Could also make it n-dimentional but n>3 will have no cluster graph only dendrogramm/elbow :(.
Most of the effort here as funny as it seems to be, was automating the no of cluster selections using dendrogramm/elbow diagrams.
Import your own datasets line:#9 and use only dimensions of them selectively line:#10.
UPDATE 28/10: Made it 3-d Hierarcical/k-means++ at last, there are more to implement though and a lot of code optimization to do, there were some problems with the dendrogram but I will manage. Also I am starting an image clustering programm based on the preevious 2. I am almost there, it's a contraption under construction :P. I also wanna note that there is a lot of spread code in comments cause I am saving it for potential use.
UPDATE 7/11: Well the image clustering is ready work with png jpg rgb files. I will try to reduce the noise of the images even more in the near future but for the time beign I would like to make it create convex hulls for the clusters. After that I think I can train it circling discrete objects in the picture.
