#*********************************************************************
#File name:     A3.ipynb
#Author:        Roman Stein      
#Date:  	    03/5/23
#Class: 	    DSCI 440W
#Assignment:    IA3
#Purpose:       Decision Stump & the Tree    
#**********************************************************************

#imports
import sympy as sp
import numpy as np
import pylab as pp
import math
import csv
import matplotlib.pyplot as plt
from sympy import *
from numpy.linalg import inv, norm
sp.init_printing(use_unicode=True, use_latex='mathjax')

TEST_FILE = '/Users/nadesocko/Desktop/MachineLearning/A3/SPECT-test.csv'
TRAIN_FILE = '/Users/nadesocko/Desktop/MachineLearning/A3/SPECT-train.csv'
LABEL_INDEX = 22
BIAS = 1
TWO = 2

#Read Data
def readCSVData(fileName):
    """    
    Function:   readData
    Descripion: Opens and reads text file
    Input:      fileName - name of file to read from
    Output:     dataList - numpy array of data from file being read
    """
    dataList = []
    # with open(fileName, "r") as f:
    #     raw = f.read()
    #     for line in raw.split("\n"):
    #         subLine = line.split()
    #         dataList.append(subLine)

    with open(fileName, 'r') as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            # print(row)
            dataList.append(row)
    return dataList

def getNumRows(matX):
    """
    Function:       getNumRows
    Description:    Returns number of rows of passed in matrix
    Input:          matx -  Matrix in question
    Output:         numRows - number of rows in matrix X
    """
    num = shape(matX)

    numRows = num[0]
    return numRows

def calcBenOfSplit(root, leaf1, leaf2):
    """
    Function:       calcBenOfSplit
    Description:    Calculates the amount of information gain for a specific split of a feater into two leafs.
    Input:          root - array of stratig features
                    leaf1 - array 1 of restuling split from root
                    leaf2 - array 2 of resutling split from root
    Output:         
    """
    rootProb = calcFeatureEntropy(root)
    leaf1Prob = calcFeatureEntropy(leaf1)
    leaf2Prob = calcFeatureEntropy(leaf2)
    rootLen = len(root)
    leaf1Len = len(leaf1)
    leaf2Len = len(leaf2)
    flow1 = leaf1Len / rootLen
    flow2 = leaf2Len / rootLen

    benOfSplit = rootProb - ((leaf1Prob * flow1) + (leaf2Prob * flow2))

    return benOfSplit




   



def calcFeatureEntropy(feature):
    prob0 = 0
    prob1 = 0
    num0 = 0
    num1 = 0
    numFeatures = len(feature)
    for i in range(numFeatures):
        if (feature[i] == 0):
            num0 += 1
        elif (feature[i] == 1):
            num1 += 1
        else:
            print ("bad data")
            return(1)

    prob0 = num0 / numFeatures
    prob1 = num1 / numFeatures

    entropy = -((prob0 * math.log2(prob0)) + (prob1 * math.log2(prob1)))


    return(entropy)






def driver():


    testData = []
    trainData = []
    trainOutput = []
    testOutput = []

    testData = readCSVData(TEST_FILE)
    trainData = readCSVData(TRAIN_FILE)
    testLabel_Index = 0
    trainLabel_Index = 0



    #Apply outpts to its own array
    for i in range(len(trainData)):
        trainOutput.append(trainData[i][trainLabel_Index])

    for i in range(len(testData)):
        testOutput.append(testData[i][testLabel_Index])

    #Convert outputs to numpy array
    trainOutput = np.array(trainOutput, dtype=float)
    testOutput = np.array(testOutput, dtype=float)

    #Create input array
    trainInputs = np.array(trainData, dtype=float)
    testInputs = np.array(testData,dtype=float)

    numRows = len(trainOutput)

    #Delte label from feature array
    trainInputs = np.delete(trainInputs,0, axis=1)
    testInputs = np.delete(testInputs,0, axis=1)

    #reshape arary
    trainOutput = trainOutput.reshape(numRows,1)
    numRows =  getNumRows(testInputs)
    testOutput = testOutput.reshape(numRows,1)

    print(shape(trainInputs))



    # print(trainInputs)

## Using entorpy nad benefit of split we dhuld cycle through all possble slits to find the one tha maximizes benefit

#Testin Area
driver()
