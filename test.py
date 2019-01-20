import pandas as pd
import numpy as np
from numpy.random import shuffle
from sklearn.metrics import confusion_matrix
import enum 
import classifiers

class Classifier(enum.Enum):
    KNN = 'KNN'
    MDC = 'MDC'
    QC = 'QC'

def partition(X, y, percent):

    n = len(y)
    nTrain = int(percent*n)
    
    idx = np.arange(0, n, 1) # Return evenly spaced values within a given interval
    shuffle(idx)
    
    idx_train = idx[0:nTrain]
    idx_test = idx[nTrain:]
    
    X_train = X[idx_train, :]
    y_train = y[idx_train]
    
    X_test  = X[idx_test,:]
    y_test  = y[idx_test]
    
    return (X_train, X_test, y_train, y_test)

def test_classifier(classifier, X, y):

    print("###### Testing " + classifier.value + " ######")
    worst_hit_rate = np.inf
    best_hit_rate = 0
    best_y_pred = np.array(0)
    worst_y_pred = np.array(0)
    best_y_test = np.array(0)
    worst_y_test = np.array(0)

    for j in range(5):

        print("*** Round {}:".format(j+1))
        X_train, X_test, y_train, y_test = partition(X, y, 0.7)
        n_e = list(y_test).count(0)
        n_p = list(y_test).count(1)
        y_pred = np.empty([len(y_test)])
        
        true_positives_p = 0 # True positives class "p"
        true_positives_e = 0 # True positives class "e"
        true_positives = 0

        for i in range(len(X_test)):

            if classifier is Classifier.KNN:
                y_pred[i] = classifiers.KNN(X_test[i], X_train, y_train)
            elif classifier is Classifier.MDC:
                y_pred[i] = classifiers.MDC(X_test[i], X_train, y_train) 
            elif classifier is Classifier.QC:
                y_pred[i] = classifiers.QC(X_test[i], X_train, y_train)

            if y_pred[i] == y_test[i]:
                true_positives += 1
                if y_pred[i] == 0:
                    true_positives_e += 1
                else:
                    true_positives_p += 1

        print("True positives: {} from {} samples".format(true_positives, len(y_test)))
        print("True positives e: {} from {} samples".format(true_positives_e, n_e))
        print("True positives p: {} from {} samples".format(true_positives_p, n_p))
        
        hit_rate = true_positives / len(y_test)
        hit_rate_e = true_positives_e / n_e
        hit_rate_p = true_positives_p / n_p

        print("Hit rate: {}".format(hit_rate))
        print("Hit rate e: {}".format(hit_rate_e))
        print("Hit rate p: {}".format(hit_rate_p))

        if hit_rate > best_hit_rate:
            best_y_pred = y_pred
            best_y_test = y_test
            best_hit_rate = hit_rate

        if hit_rate < worst_hit_rate:
            worst_y_pred = y_pred
            worst_y_test = y_test
            worst_hit_rate = hit_rate

    print("Best result: {}".format(best_hit_rate ))
    print("Confusion matrix best result: ")
    cnf_matrix_best = confusion_matrix(best_y_test, best_y_pred)
    print(cnf_matrix_best)

    print("Worst result: {}".format(worst_hit_rate))
    print("Confusion matrix worst result: ")
    cnf_matrix_worst = confusion_matrix(worst_y_test, worst_y_pred)
    print(cnf_matrix_worst)


def main():

    df = pd.read_csv('data/resample_numeric.csv')
    X = df.drop(columns=['id', 'Class']).values
    y = df['Class'].values

    test_classifier(Classifier.KNN, X, y)
    test_classifier(Classifier.MDC, X, y)
    test_classifier(Classifier.QC, X, y)

if __name__ == '__main__':
    main()