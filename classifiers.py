import numpy as np
from collections import Counter

def euclidian_distance(x1, x2):
    distance = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2

    return np.sqrt(distance)

def mahalanobis_distance(x1, x2, C):
    return np.transpose(x1 - x2) @ C @ (x1 - x2)

def get_centroids(X, y):

    n1 = list(y).count(0) # Class 1
    n2 = list(y).count(1) # CLass 2
    M = np.array([np.zeros(X.shape[1]), np.zeros(X.shape[1])])

    for i in range(X.shape[0]):
        if y[i] == 0:
            M[0] = M[0] + X[i]
        elif y[i] == 1:
            M[1] = M[1] + X[i]
 
    M[0] = M[0] / n1
    M[1] = M[1] / n2
    return M

# Nearest Neighbour Classifier
def KNN(x, X, y, k=3):

    distances = []
    for idx in range(X.shape[0]):
        dist = euclidian_distance(x, X[idx])
        distances.append([dist, y[idx]])

    labels = [i[1] for i in sorted(distances, key=lambda d: d[0])[:k]] 
    pred = Counter(labels).most_common(1)[0][0]           
    return pred

# Minimum Distance from the Centroid Classifier
def MDC(x, X, y):
    M = get_centroids(X, y)
    y = np.array([0, 1])
    return KNN(x, M, y, k=1)

# Quadratic Classifier
def QC(x, X, y):
    C = np.cov(X.T)
    C_inv = np.linalg.pinv(C)
    M = get_centroids(X, y)
    pred = None
    min_dist = np.inf
    for label in range(M.shape[0]):
        dist = mahalanobis_distance(x, M[label], C_inv)
        if dist < min_dist:
            min_dist = dist
            pred = label
    return pred