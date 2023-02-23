#*********************************************************************
#File name:  A2.ipynb
#Author:     Roman Stein      
#Date:  	    02/20/23
#Class: 	    DSCI 440W
#Assignment: IA1
#Purpose:    Impliment Linear regression and        
#**********************************************************************

#imports
import sympy as sp
import numpy as np
import pylab as pp
from sympy import *
sp.init_printing(use_unicode=True, use_latex='mathjax')

TEST_FILE = './A2/housing_test.txt'
TRAINING_FILE = './A2/housing_train.txt'
PRICE_INDEX = 13
BIAS = 1
TWO = 2

allData = []
trainPrice = []
testPrice = []
testData = []

def findW(matX, vecY):
    """
    Function:   findW
    Descripion: Creates matrix W using closed form solution of linear regression
    Input:      matx - matrix of inputs
                vecY - vector of outputs
    Output:     numRows - number of rows in matrix X
    """
    # matW = ((matX.T * matX)**-1) * (matX.T * vecY)
    vecW = np.dot(np.linalg.inv(np.dot(matX.T,matX)), np.dot(matX.T,vecY))
    return vecW

def genPreds(matX, vecW):
    """
    Function:       genPreds 
    Description:    Genrates predicted values given inputs X and weights W
    Input:  matX -  Matrix of inputs
            vecW -  Column vector of weights

    Output: matPreds - matrix of predicted values
    """ 
    vecPred = np.dot(matX,vecW)
    return vecPred

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

def genSSE(vecPred,vecActual):
    """
    Function:   genSSE
    Descripion: Determines the SSE given matrix X,Y,W
    Input:      vecPred - vector or predicted values
                vecActual - vector of actaul values
    Output:    SSE - Sum of Squared Errors
    """

    sum = 0
    numRows = getNumRows(vecPred)
    for i in range(numRows):
        error = (vecPred[i] - vecActual[i])** TWO
        sum = sum + error
    return (sum)

def findSSE(matX, vecY, vecW):
    """
    Function:   findSSE
    Descripion: Determines the SSE given matrix X,Y,W
    Input:     matX - matrix of values
               vecY - training data
               vecW - vector of weights
    Output:    SSE - Sum of Squared Errors of linear regression
    """
    sum = 0
    matSSE = (np.dot(matX,vecW)-vecY)**TWO
    numRows = getNumRows(matSSE)
    for i in range(numRows):
        sum = sum + matSSE[i]
    return (sum)

#Read Data
def readData(fileName):
    dataList = []
    with open(fileName, "r") as f:
        raw = f.read()
        for line in raw.split("\n"):
            subLine = line.split()
            dataList.append(subLine)
    return dataList

def addBiasnDelLast(array):
    numInputs = len(array[BIAS])
    numRows = len(array)

    array = np.insert(array,0, 1, axis = 1)
    array = np.delete(array,(numInputs), axis=1)

    array = array.reshape(numRows,numInputs)
    return array


# Read data into lists
testData = readData(TEST_FILE)
trainData = readData(TRAINING_FILE)

# Get last ow into its own list
for i in range(len(trainData)):
    trainPrice.append(trainData[i][PRICE_INDEX])

for i in range(len(testData)):
    testPrice.append(testData[i][PRICE_INDEX])

#convert lists to np.arrays
trainPrice = np.array(trainPrice, dtype=float)
testPrice = np.array(testPrice, dtype=float)

trainInputs = np.array(trainData, dtype=float)
testInputs = np.array(testData,dtype=float)

# Add bias column to fornt and delet last col
trainInputs = addBiasnDelLast(trainInputs)
testInputs = addBiasnDelLast(testInputs)

#reshape arary
numRows = getNumRows(trainPrice)
trainPrice = trainPrice.reshape(numRows,1)
numRows =  getNumRows(testInputs)
testPrice = testPrice.reshape(numRows,1)


trainWeights = findW(trainInputs, trainPrice)
print("Optimal Weights: ")
print(trainWeights)

trainPred = genPreds(trainInputs,trainWeights)
trainSSE = genSSE(trainPred,trainPrice)
testPred = genPreds(testInputs, trainWeights)
testSSE = genSSE(testPred, testPrice)
print("Training SSE: ")
print(trainSSE)

print("Test SSE")
print(testSSE)






