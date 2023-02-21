#*********************************************************************
#File name:  A1.ipynb
#Author:     Roman Stein      
#Date:  	    02/20/23
#Class: 	    DSCI 440W
#Assignment: IA1
#Purpose:    Impliment Linear regressio with the closed-form solution       
#**********************************************************************

#imports
import sympy as sp
import pandas as pd
import numpy as np
import pylab as pp
import matplotlib.pyplot as plt
from sympy import *
sp.init_printing(use_unicode=True, use_latex='mathjax')

#Constants and variables
FILENAME = './A1/cf.txt'
cel = []
far = []


#***********************************************************************
#Function:   getNumRows
#Descripion: Returns the number of rows in the passed in matrix
#Input:      matx - matrix in question
#Output:     numRows - number of rows in matrix X
#***********************************************************************
def getNumRows(matX):
    """
    Function: getNumRows
    Description: Returns number of rows of passed in matrix
    """
    num = shape(matX)
    numRows = num[0]
    return numRows

#******************************************************************************
#Function:   findW
#Descripion: Creates matrix W using closed form solution of linear regression
#Input:      matx - matrix in question
#Output:     numRows - number of rows in matrix X
#******************************************************************************
# W =X.T * X)**-1 * X.T*Y
def findW(matX, vecY):
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
    print(vecW.shape)
    print(vecW.T.shape)
    vecPred = np.dot(matX,vecW)
    return vecPred




#******************************************************************************
#Function:   findSSE
#Descripion: Determines the SSE given matrix X,Y,W
#Input:     matX - matrix of values
#           vecY - training data
#           vecW - vector of weights
#Output:    SSE - Sum of Squared Errors of linear regression
#******************************************************************************
def findSSE(matX, vecY, vecW):

    SSE = (np.dot(matX,vecW)-vecY)**2

    return (SSE)

#*****
# Start Driver
#*****
def  driver():
    TRAINING_DATA = 1
    ODD = 1
    EVEN = 2
    HUNDRED = 100
    DIVIDER = 0.01
    myData = np.loadtxt(FILENAME, delimiter=' ')
    numRows = myData.size

    celCounter = 0
    farCounter = 0

    # Read in data to list
    for i in range(numRows):
        if (i % EVEN == 0):
            cel.append(myData[i])
            celCounter = celCounter + ODD
        else:
            far.append(myData[i])
            farCounter = farCounter + ODD



    # ## Coonvert list to np.array
    celArr = np.array(cel)
    numbRows = celArr.size
    farArr = np.array(far)
    ## Add bias to cel array
    biasCol = np.ones((numbRows,ODD))
    biasCol = biasCol.reshape(numbRows,ODD)
    celArr = celArr[:,np.newaxis]
    celBias = np.append(celArr, biasCol,axis=ODD)
    #celBias = np.append(celArr, np.ones((numbRows,1))) 

    ## Reshape Arrays
    celBias = celBias.reshape(numbRows,EVEN)
    farArr = farArr.reshape(numbRows,ODD)
 
    vecW = findW(celBias,farArr)
    print("Optimal Weights")
    print(vecW)

    vecPred = genPreds(celBias,vecW)
    SSE = findSSE(celBias,farArr,vecW)

    pp.figure(TRAINING_DATA)
    pp.plot(celArr,farArr, '.r', label='Training Data')
    pp.plot(celArr, vecPred,'.b', label='Predicted Values')
    x_hlp = np.arange(-ODD,HUNDRED,DIVIDER)
    y_hlp = vecW[1] + vecW[0]*x_hlp
    pp.plot(x_hlp, y_hlp, 'm',label='Linear Fit')
    pp.xlabel('x (C)')
    pp.ylabel('y(F)')
    pp.legend()
    pp.show()
driver()
