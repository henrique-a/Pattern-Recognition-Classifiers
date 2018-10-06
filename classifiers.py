import numpy as np

def euclidian_distance(x1, x2):
    distance  = 0
    for i in range(len(x1)):
        distance += (x1[i] - x2[i])**2

    return np.sqrt(distance)

def mahalanobis_distance(x1, x2, C):
    return np.transpose(x1 - x2) @ C @ (x1 - x2)

def get_centroids(X, y):

    n1 = list(y).count(0) # Class 1
    n2 = list(y).count(1) # CLass 2
    m = np.array([np.zeros(X.shape[1]), np.zeros(X.shape[1])])

    for i in range(X.shape[0]):
        if y[i] == 0:
            m[0] = m[0] + X[i]
        elif y[i] == 1:
            m[1] = m[1] + X[i]
 
    m[0] = m[0] / n1
    m[1] = m[1] / n2

    return m

# Nearest Neighbour Classifier
def NN(x, X):
    i = 0
    min_dist = np.inf
    min_i = 0
    for i in range(X.shape[0]):
        dist = euclidian_distance(x, X[i])
        if dist < min_dist:
            min_dist = dist
            min_i = i 
            
    return min_i

# Minimum Distance from the Centroid Classifier
def MDC(x, X, y):
    m = get_centroids(X, y)
    return NN(x, m)

# Quadratic Classifier
def QC(x, X, y):
    C = X.cov().values
    C_inv = np.linalg.pinv(C)
    m = get_centroids(X.values, y)
    min_i = 0
    min_dist = np.inf
    for i in range(m.shape[0]):
        dist = mahalanobis_distance(x, m[i], C_inv)
        if dist < min_dist:
            min_dist = dist
            min_i = i
    return min_i