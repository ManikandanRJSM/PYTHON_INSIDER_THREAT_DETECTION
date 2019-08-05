from __future__ import division
import math
import operator
import copy
import csv
import time
import random
import csv, os, sys
import numpy as np
from SVM import SVM
from c45 import freq, info, infox, gain
from GA import GeneticAlgorithm,FE1
from measuremen import minkowski_Similarity,nth_root
from ANN import Featureselection,getNeighbors,getAccuracy,loadDataset,getResponse,sensitivity,specificity,euclideanDistance
import numpy as np
import matplotlib.pyplot as plt
filepath = os.path.dirname(os.path.abspath(__file__))
class csvdata():
    def __init__(self, classifier):
        self.rows = []
        self.attributes = []
        self.attribute_types = []
        self.classifier = classifier
        self.class_col_index = None
        
def readData(filename, header=True):
    data, header = [], None
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        if header:
            header = spamreader.next()
        for row in spamreader:
            data.append(row)
    return (np.array(data), np.array(header))

def processeqence(dataset):
    #C4.5 Filtering 
    c45=freq(dataset)
    print (c45[1])
    FE1(c45[1])
   
def svmcall():
    # Load data
    filename='2009-12.csv'
    C=1.0
    kernel_type='linear'
    epsilon=0.001
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)
    data = data.astype(float)
     
    # Split data
    X, y = data[:,0:-1], data[:,-1].astype(int)

    # Initialize model
    model = SVM()

    # Fit model
    support_vectors, iterations = model.fit(X, y)

    # Support vector count
    sv_count = support_vectors.shape[0]

    # Make prediction
    y_hat = model.predict(X)
    ###print("Support vector count: %d" % (sv_count))
   

def mainprocess():
    dataset = csvdata("")
    training_set = csvdata("")
    test_set = csvdata("")
    trainingSet=[]
    testSet=[]
    split = 0.1
    # 11 Attributes
    print ("id,Clump Thickness ,elial Cell Size,Bare Nuclei,Bland Chromatin,Normal Nucleoli,Mitoses,classs")
    with open('2009-12.csv', 'rt',encoding='utf-8') as f:
        original_file = f.read()
    rowsplit_data = original_file.splitlines()
    
    dataset.rows = [rows.split(',') for rows in rowsplit_data]
    
    dataset.attributes = dataset.rows.pop(0)
    loadDataset('2009-12.csv', split, trainingSet, testSet)
    print ('Train set: ' + repr(len(trainingSet)))
    print ('Test set: ' + repr(len(testSet)))
      # print  dataset.rows
    for row in  rowsplit_data:
       print (row)

    dataset.class_col_index = range(len(dataset.attributes))[-1]
   
    ds = np.unique(rowsplit_data).tolist()
    processeqence(ds)
    #===================feature selection======================================
    fl,b,m,fe=Featureselection(ds);
    print ("ranking")
    print ("================================================================================")
    rank = sorted(fe)
    print (rank)
    print ()
    print ("Benign :",b)
    print ("Malignant:",m)
   # svmcall()
      
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
        result = getResponse(neighbors)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    sens = sensitivity(testSet, predictions)
    sp = specificity(testSet, predictions)
    print ('sensitivity: ', sens)
    print ('specificity: ', sp)
    print ('Accuracy: ', accuracy)

    #x = [0,sens]
    #plt.plot(x)
    #plt.ylabel('sensitivity')
    #plt.xlabel('')
    #plt.show()

    #x = [0,sp]
    #plt.plot(x)
    #plt.ylabel('specificity')
    #plt.xlabel('')
    #plt.show()

    
    #x = [0,accuracy]
    #plt.plot(x)
    #plt.ylabel('Accuracy')
    #plt.xlabel('')
    #plt.show()

 
       
if __name__ == "__main__":
    mainprocess()
    
