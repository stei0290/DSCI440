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
import numpy as np
import pylab as pp
from sympy import *
sp.init_printing(use_unicode=True, use_latex='mathjax')

#Constants and variables
FILENAME = './A1/cf.txt'
TRAINING_DATA = 1
ODD = 1
EVEN = 2
HUNDRED = 100
DIVIDER = 0.01
INPUT = 33

cel = []
far = []


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

def findW(matX, vecY):
    """
    Function:   findW
    Descripion: Creates matrix W using closed form solution of linear regression
    Input:      matx - matrix in question
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

def  driver():
    """
    Function:   driver
    Descripion: General fucntion for determing SSE uring normal equations
    Input:      None
    Output:     None
    """

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
    print("Sum of Squared Errors")
    print(SSE)
    out = vecW[ODD] + vecW[0] * INPUT
    print("Given an input of 33 *C")
    print("Ouput of: ", str(out))
    pp.figure(TRAINING_DATA)
    pp.plot(celArr,farArr, '.r', label='Training Data')
    pp.plot(celArr, vecPred,'.b', label='Predicted Values')
    x_hlp = np.arange(-ODD,HUNDRED,DIVIDER)
    y_hlp = vecW[ODD] + vecW[0]*x_hlp
    pp.plot(x_hlp, y_hlp, 'm',label='Linear Fit')
    pp.xlabel('x (C)')
    pp.ylabel('y(F)')
    pp.legend()
    pp.show()

driver()
