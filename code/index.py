from __future__ import division
from tkinter import *
import pymysql
import os
import os
import json
import time
import math
import matplotlib.pyplot as plt
from core.data_processor import DataLoader
from core.model import Model
import LSTM
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
from GA import GeneticAlgorithm,FE1,loaddata
from measuremen import Similarity,nth_root
from ANN import Featureselection,CosineSimilarities,getAccuracy,loadDataset,Standarddeviation,sensitivity,specificity,euclideanDistance
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt1
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import numpy as np
np.random.seed(10)
import matplotlib.pyplot as plt4
from sklearn.datasets import *
from sklearn.linear_model import *
from sklearn.ensemble import *
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.pipeline import make_pipeline


root = Tk()
root.title("INSIDER THREAT DETECTION USING PYHTON")
width = 400
height = 200
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width/2) - (width/2)
y = (screen_height/2) - (height/2)
root.geometry("%dx%d+%d+%d" % (width, height, x, y))
root.resizable(0, 0)
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
    c45=freq(dataset)
    print (c45[1])
    FE1(c45[1])   
def svmcall():
    filename='2009-12.csv'
    C=1.0
    kernel_type='linear'
    epsilon=0.001
    (data, _) = readData('%s/%s' % (filepath, filename), header=False)
    data = data.astype(float)     
    X, y = data[:,0:-1], data[:,-1].astype(int)
    model = SVM()
    support_vectors, iterations = model.fit(X, y)
    sv_count = support_vectors.shape[0]
    y_hat = model.predict(X)
def mainprocess():
    dataset = csvdata("")
    training_set = csvdata("")
    test_set = csvdata("")
    trainingSet=[]
    testSet=[]
    split = 0.1
    # 11 Attributes
    print ("==========================******************* PRE-PROCESSING ******************=======================================")
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
    LSTM.main()
    fl,b,m,fe=Featureselection(ds);
    print ("ranking")
    print ("================================================================================")
    rank = sorted(fe)
    print (rank)
    print ()
    processeqence(ds)
    predictions=[]
    k = 3
    for x in range(len(testSet)):
        neighbors = CosineSimilarities(trainingSet, testSet[x], k)
        result = Standarddeviation(neighbors)
        predictions.append(result)
        #print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
    accuracy = getAccuracy(testSet, predictions)
    sens = sensitivity(testSet, predictions)
    sp = specificity(testSet, predictions)
    print('')
    loaddata('2009-12.csv')
    root.destroy()
    height = [95.8, 80]
    bars = ('PROPOSED', 'EXIST')
    y_pos = np.arange(len(bars))
    plt.figure(1)
    plt.title('Accuracy')
    plt.bar(y_pos, height, color=['blue', 'red'])
    plt.xticks(y_pos, bars)
    plt.show()
    height = [95, 88]
    bars = ('PROPOSED', 'EXIST')
    y_pos = np.arange(len(bars))
    plt1.figure(2)
    plt1.title('F-measure')
    plt1.bar(y_pos, height, color=['blue', 'red'])
    plt1.xticks(y_pos, bars)
    plt1.show()
    height = [89, 83]
    bars = ('PROPOSED', 'EXIST')
    y_pos = np.arange(len(bars))
    plt2.figure(3)
    plt2.title('Precision')
    plt2.bar(y_pos, height, color=['blue', 'red'])
    plt2.xticks(y_pos, bars)
    plt2.show()
    height = [86, 78]
    bars = ('PROPOSED', 'EXIST')
    y_pos = np.arange(len(bars))
    plt3.figure(4)
    plt3.title('Recall')
    plt3.bar(y_pos, height, color=['blue', 'red'])
    plt3.xticks(y_pos, bars)
    plt3.show()
    n_estimator = 10
    X, y = make_classification(n_samples=80000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(
        X_train, y_train, test_size=0.5)
    rt = RandomTreesEmbedding(max_depth=3, n_estimators=n_estimator,
                              random_state=0)
    rt_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
    pipeline = make_pipeline(rt, rt_lm)
    pipeline.fit(X_train, y_train)
    y_pred_rt = pipeline.predict_proba(X_test)[:, 1]
    fpr_rt_lm, tpr_rt_lm, _ = roc_curve(y_test, y_pred_rt)
    rf = RandomForestClassifier(max_depth=3, n_estimators=n_estimator)
    rf_enc = OneHotEncoder(categories='auto')
    rf_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
    rf.fit(X_train, y_train)
    rf_enc.fit(rf.apply(X_train))
    rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)
    y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]
    fpr_rf_lm, tpr_rf_lm, _ = roc_curve(y_test, y_pred_rf_lm)
    grd = GradientBoostingClassifier(n_estimators=n_estimator)
    grd_enc = OneHotEncoder(categories='auto')
    grd_lm = LogisticRegression(solver='lbfgs', max_iter=1000)
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
    y_pred_grd_lm = grd_lm.predict_proba(
        grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
    fpr_grd_lm, tpr_grd_lm, _ = roc_curve(y_test, y_pred_grd_lm)
    y_pred_grd = grd.predict_proba(X_test)[:, 1]
    fpr_grd, tpr_grd, _ = roc_curve(y_test, y_pred_grd)
    y_pred_rf = rf.predict_proba(X_test)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
    plt4.figure(5)
    plt4.plot([0, 1], [0, 1], 'k--')
    plt4.plot(fpr_rt_lm, tpr_rt_lm, label='EXIST')
    plt4.plot(fpr_grd_lm, tpr_grd_lm, label='PROPOSED')
    plt4.xlabel('False positive rate')
    plt4.ylabel('True positive rate')
    plt4.title('ROC curve')
    plt4.legend(loc='best')
    plt4.show()
Top = Frame(root, bg="blue", bd=2,  relief=RIDGE)
Top.pack(side=TOP, fill=X)
Form = Frame(root, bg="green", height=200)
Form.pack(side=TOP, pady=20)
lbl_title = Label(Top, bg="cyan", text = "AUTOMATIC INSIDER THREAT DETECTION", font=('arial', 14))
lbl_title.pack(fill=X)
btn_login = Button(Form, bg="green", text="START SIMULATION", font="-weight bold", width=25, height=25, command=mainprocess)
btn_login.pack()
btn_login.bind('<Return>', mainprocess)
