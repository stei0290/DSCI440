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

allData = []
housePrice = []

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
    matSSE = (np.dot(matX,vecW)-vecY)**EVEN
    numRows = getNumRows(matSSE)
    for i in range(numRows):
        sum = sum + matSSE[i]
    return (sum)


#Read Data
with open(TRAINING_FILE, "r") as f:
   raw = f.read()
   for line in raw.split("\n"):
    subLine = line.split()
    allData.append(subLine)


for i in range(len(allData)):
    housePrice.append(allData[i][PRICE_INDEX])

numInputs = len(allData[BIAS])
numRows = len(allData)
prices = np.array(housePrice, dtype=float)
inputs = np.array(allData, dtype=float)
inputs = np.insert(inputs,0, 1, axis = 1)
inputs = np.delete(inputs,(numInputs), axis=1)

print(inputs.shape)
inputs = inputs.reshape(numRows,numInputs)
print(inputs.shape)
print("prices")
print(prices.shape)
prices = prices.reshape(numRows,1)
print(prices.shape)
vecW = findW(inputs, prices)
print("Optimal Weights: ")
print(vecW)








