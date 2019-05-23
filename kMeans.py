#Based on Peter Harrington, Machine Learning in Action
'''
Name: Andrew Abbott
User ID: aabbott
Python Version: 2.7
Usage: python kMeans.py

'''

import numpy as np
import matplotlib.pyplot as plt

'''
Data comes in like this:
0.xxx ...  0.yyyy
where the values are floating point numbers representing points in a space
my example is m x 2, but the code would work for n features
Read these into an m x 2 numpy matrix, where m is the number of points
'''
def loadData(file_name):
    with open(file_name) as fin:
        rows = (line.strip().split('\t') for line in fin)
        dataMat = [map(float,row) for row in rows]
    return np.mat(dataMat)

'''
Construct a k x n matrix of randomly generated points as the
initial centroids. The points have to be in the range of the points
in the data set
'''
def randCent(dataMat,k):
    numCol = np.shape(dataMat)[1]  #notice the number of cols is not fixed.
    centroids = np.mat(np.zeros((k,numCol))) #kxnumCol matrix of zeros
    for col in range(numCol):
        minCol = np.min(dataMat[:,col]) #minimum from each column
        maxCol = np.max(dataMat[:,col]) #maximum from each column
        rangeCol = float(maxCol - minCol)
        centroids[:,col] = minCol + rangeCol * np.random.rand(k,1)
    return centroids

'''
Compute the Euclidean distance between two points
Each point is vector, composed of n values, idicating a point in n space 
'''
def distEucl(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB,2)))

def kMeans(dataMat, k, distMeas=distEucl, createCent=randCent):
    m = np.shape(dataMat)[0]  #how many items in data set
    
    #create an mX2 natrix filled with zeros
    #each row stores centroi information for a point
    #col 0 stores the centroid index to which the point belongs
    #col 1 stores the distance from the point to the centroid
    clusterAssment = np.mat(np.zeros((m,2)))

    #create k randomly placed centroids
    centroids = createCent(dataMat,k)  
    
    clusterChange = True
    while clusterChange:
        clusterChange = False
        
        for i in range(m):
            curDistance = 100000 #initialize to large number 
            curLoc = 0
            for x in range(k):
                eucDis = distEucl(dataMat[i,:],centroids[x,:])
                if eucDis < curDistance:
                    curDistance = eucDis
                    curLoc = x
            if clusterAssment[i,0] != curLoc:
                clusterChange = True
                clusterAssment[i,:] = curLoc, curDistance
                
        #calculate the mean for each assigned mean 
        for i in range(k): #for each centroid
            #nonzero returns the indicies of the given condition, in this case when the assigned centroid is equal to the the current i value
            #then the list of indicies is returned and used to with in the data matirx to isolate the coridantes associatd with a ceraitn centroid
            pointsInCluster = dataMat[np.nonzero(clusterAssment[:, 0] == i)[0]]
            #then the mean of the iosdlated coridates are found and assignd to the new centroid 
            centroids[i, :] = np.mean(pointsInCluster, axis = 1) 
            
    return centroids, iter #is the number of iterations required

def plot_results(dataMat, centroids, k, iterations):
    #your code goes here.  The trick is to transfrom the incoming matrices 
    #to lists
    dataLst = dataMat.tolist()
    x_point = [point[0] for point in dataLst]
    y_point = [point[1] for point in dataLst]
    centLst = centroids.tolist()
    x_cent = [cent[0] for cent in centLst]
    y_cent = [cent[1] for cent in centLst]

    plt.scatter(x_point,y_point,color='r')
    plt.scatter(x_cent,y_cent, color = 'g', marker = 'o')
   
    plt.title("k-Means")
    #plt.text(3, 2, "Number of Iterations: " + str(iterations))
    plt.show()
    '''
    On the same scatter plot, plot the points and the centroids
    The centroid points should be a different shape and color than the data
    points
    '''

def main():
    k = 4
    dataMat = loadData("testSet.txt")

    centroids, iterations = kMeans(dataMat, k, distEucl, randCent)
    
    plot_results(dataMat, centroids, k, iterations)

    
main()

