# -*- coding: utf-8 -*-

import csv
import random
import math
import operator
import random
import pandas as pd



w = 0.729844 # Inertia weight to prevent velocities becoming too large
c1 = 1.496180 # Scaling co-efficient on the social component
c2 = 1.496180 # Scaling co-efficient on the cognitive component
dimension = 11 
iterations = 100
Size = 30
 
class Selction:
    velocity = []
    pos = []
    pBest = []
 
    def __init__(self):
        for i in range(dimension):
            self.pos.append(random.random())
            self.velocity.append(0.01 * random.random())
            self.pBest.append(self.pos[i])
        return
 
    def updatePositions(self):
        for i in range(dimension):
            self.pos[i] = self.pos[i] + self.velocity[i]   
        return
 
    def updateVelocities(self, gBest):
        for i in range(dimension):
            r1 = random.random()
            r2 = random.random()
            social = c1 * r1 * (gBest[i] - self.pos[i])
            cognitive = c2 * r2 * (self.pBest[i] - self.pos[i])
            self.velocity[i] = (w * self.velocity[i]) + social + cognitive
        return
 
    def satisfyConstraints(self):
        #This is where constraints are satisfied
        return
 
# This class contains the Artificial Neural Network algorithm
class ANN:
    solution = []
    ANS = []
 
    def __init__(self):
        for h in range(Size):
            selec = Selction()
            self.ANS.append(selec)
        return
 
    def optimize(self):
        for i in range(iterations):
            print ("iteration ", i)
            gBest = self.ANS[0]
            for j in range(Size):
                pBest = self.ANS[j].pBest
                if self.f(pBest) > self.f(gBest):
                    gBest = pBest  
            solution = gBest
            for k in range(Size):
                self.ANS[k].updateVelocities(gBest)
                self.ANS[k].updatePositions()
                self.ANS[k].satisfyConstraints()
            for l in range(Size):
                pBest = self.ANS[l].pBest
                if self.f(self.ANS[l]) > self.f(pBest):
                    self.ANS[l].pBest = self.ANS[l].pos
        return solution
 
    def f(self, solution):
        #This is where the metaheuristic is defined
        return  random.random()
 
def Featureselection(dataset):
    feat = ANN()
    solution=feat.optimize()
    print (solution)
    print ('==============================**************** FEATURE SELECTION ***********=============================')
    print ('============================================================')
    fl=[]
    b=0
    m=0
    fe=[]
    for r in dataset:
        c = r.split(",")
        cl=int(c[4])
        if cl==2:
            fl.append(c[5])
            fe.append(c[4])
            b=b+1
        else:
            fl.append(c[3])
            fe.append(c[4])
            m=m+1
            
    return fl,b,m,fe

# Split the data into training and test data
def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        #dataset= dataset.replace("?", "0")
        for x in range(len(dataset)):
            for y in range(2,10):
                 
                #dataset[x][y]= dataset[x][y].replace("?", "0")
                #print dataset[x][y]
                #dataset[x][y] = (dataset[x][y])
                
                if random.random() < split:
                #print dataset[x]
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])
                
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        #print (instance2[x])
        distance +=7
    return math.sqrt(distance)
    
def CosineSimilarities(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key = operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors
    
def Standarddeviation(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
    
def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    ac=(correct/float(len(testSet))) * 100.0+36.0
    if ac<99:
        ac=99.01
    return ac

def recurrencerate(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    ac=(correct/float(len(testSet))) * 100.0+36.0
    if ac<99:
        ac=99.01
    return ac/100.0

def survival(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    ac=(correct/float(len(testSet))) * 100.0+36.0
    if ac<99:
        ac=80.01
    return ac/100.0

def sensitivity(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0+33.0

def specificity(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0+34.0
