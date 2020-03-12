import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')

#Clustering -> Segmentation using Age and Spending Score
data1 = dataset[['Age', 'Spending Score (1-100)']].values

inertia = []
for i in range(1, 11):
    model = KMeans(n_clusters = i, init = 'k-means++', n_init = 10, max_iter = 300, tol = 0.0001, algorithm = 'elkan')
    model.fit(data1)
    inertia.append(model.inertia_)

#Selecting N Clusters based in Inertia -> Visualization
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()

#Found that n=4 is most effective
algorithm = (KMeans(n_clusters = 4 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(data1)
labels1 = algorithm.labels_
centroids1 = algorithm.cluster_centers_

#Predicting using 4 clusters on Age vs Spending_Score
h = 0.02
x_min, x_max = data1[:, 0].min() - 1, data1[:, 0].max() + 1
y_min, y_max = data1[:, 1].min() - 1, data1[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

#Visualizing above clusters
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z = Z.reshape(xx.shape)
plt.imshow(Z , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Age' ,y = 'Spending Score (1-100)' , data = dataset , c = labels1 , 
            s = 200 )
plt.scatter(x = centroids1[: , 0] , y =  centroids1[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Age')
plt.show()


#Clustering -> Segmentation using Annual Income and Spending Score
data2 = dataset[['Annual Income (k$)' , 'Spending Score (1-100)']].values
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
    algorithm.fit(data2)
    inertia.append(algorithm.inertia_)
    
#Selecting N Clusters based in Inertia -> Visualization
plt.figure(1 , figsize = (15 ,6))
plt.plot(np.arange(1 , 11) , inertia , 'o')
plt.plot(np.arange(1 , 11) , inertia , '-' , alpha = 0.5)
plt.xlabel('Number of Clusters') , plt.ylabel('Inertia')
plt.show()

#Found that n=5 is most effective
algorithm = (KMeans(n_clusters = 5 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') )
algorithm.fit(data2)
labels2 = algorithm.labels_
centroids2 = algorithm.cluster_centers_

#Predicting using 5 clusters on Annual Income vs Spending Score
h = 0.02
x_min, x_max = data2[:, 0].min() - 1, data2[:, 0].max() + 1
y_min, y_max = data2[:, 1].min() - 1, data2[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z2 = algorithm.predict(np.c_[xx.ravel(), yy.ravel()]) 

#Visualizing above clusters
plt.figure(1 , figsize = (15 , 7) )
plt.clf()
Z2 = Z2.reshape(xx.shape)
plt.imshow(Z2 , interpolation='nearest', 
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap = plt.cm.Pastel2, aspect = 'auto', origin='lower')

plt.scatter( x = 'Annual Income (k$)' ,y = 'Spending Score (1-100)' , data = dataset , c = labels2 , 
            s = 200 )
plt.scatter(x = centroids2[: , 0] , y =  centroids2[: , 1] , s = 300 , c = 'red' , alpha = 0.5)
plt.ylabel('Spending Score (1-100)') , plt.xlabel('Annual Income (k$)')
plt.show()


