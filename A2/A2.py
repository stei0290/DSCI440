#*********************************************************************
#File name:     A2.ipynb
#Author:        Roman Stein      
#Date:  	    03/5/23
#Class: 	    DSCI 440W
#Assignment:    IA2
#Purpose:       Impliment Linear regression and regularization     
#**********************************************************************

#imports
import sympy as sp
import numpy as np
import pylab as pp
import matplotlib.pyplot as plt
from sympy import *
from numpy.linalg import inv, norm
sp.init_printing(use_unicode=True, use_latex='mathjax')

TEST_FILE = './A2/housing_test.txt'
TRAINING_FILE = './A2/housing_train.txt'
PRICE_INDEX = 13
BIAS = 1
TWO = 2

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

def findWReg(matX, vecY, lamb):
    """
    Function:   findWReg
    Descripion: Creates matrix W of weights using regulization term
    Input:      matx - matrix of inputs
                vecY - vector of outputs
                lamb - rugularization term
    Output:     vecW - vecor of weights
    """
    numRows = getNumRows(matX)
    # print("numrows")
    # print(numRows)
    col = shape(matX)
    # print("Cols")
    numCols = col[1]
    # print(numCols)
    idenMat = np.eye(numCols)
    op1 = np.dot(lamb,idenMat)
    op2 = np.dot(matX.T,matX)
    op3 = inv((op1 + op2))
    op4 = np.dot(matX.T,vecY)
    vecW = np.dot(op3,op4)
    return vecW

    # vecW = np.dot((((np.dot(lamb,idenMat) + np.dot(matX.T,matX)) ** -1),np.dot(matX,vecY)))

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

def findSSEReg(matX, vecY, vecW, lamb):
    """
    Function:   findSSEvar
    Descripion: Determines the SSE given matrix X,Y,W utilizing a regularization term
    Input:      vecPred - vector or predicted values
                vecActual - vector of actaul values
                lamb - regularization term
    Output:     lamb - reuglarization temrn used in calculation
                sum - Sum of Squared errors
    """
    sum = 0
    op1 = np.dot(matX,vecW)
    op2 = (vecY - op1) ** 2

    # op3 = np.dot(vecW.T,vecW)
    # op4 = np.dot(lamb,op3) **2
    op4 = euclidNorm(vecW)
    op6 = np.dot(lamb,op4)
    op5 = op2 + op6
    numRows = getNumRows(op5)

    for i in range(numRows):
        sum = sum + op5[i]

    return (sum)

def euclidNorm(vecW):
    """
    Function:   euclidNorm
    Descripion: Determines the Euclidiean norm of a vector
    Input:      vecW - Vector to find norm
    Output:     op3 - L2 norm of vecW
    """
    op5 = np.dot(vecW.T,vecW)
    # op3 = np.sum(vecW)
    #op3 = norm(vecW)
    op3 = np.sum(op5)

    return(op3)

# Unused fucntion to test that findSSEReg was working properly
# def findSSEReg2(matX, vecY, vecW, lamb):
     
#     sum = 0
#     numRows = getNumRows(matX)

#     for i in range(numRows):
#         op1 = (vecY[i] - np.dot(vecW.T,matX[i])) ** 2
#         op2 = np.dot(vecW.T,vecW)
#         op3 = np.dot(lamb,op2)
#         op4 = op1 + op3
#         sum = sum + op4
#     return sum

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
    matSSE = (vecY - np.dot(matX,vecW))**TWO
    numRows = getNumRows(matSSE)
    for i in range(numRows):
        sum = sum + matSSE[i]
    return (sum)

#Read Data
def readData(fileName):
    """    
    Function:   readData
    Descripion: Opens and reads text file
    Input:      fileName - name of file to read from
    Output:     dataList - numpy array of data from file being read
    """
    dataList = []
    with open(fileName, "r") as f:
        raw = f.read()
        for line in raw.split("\n"):
            subLine = line.split()
            dataList.append(subLine)
    return dataList


def addBiasDelLast(array):
    """
    Function:   addBiasDelLast
    Descripion: adds bas column to front of array and deletes the last column in the array
    Input:      array - numpy array to be operated on
    Output:     array - copy of origonal array with modifications
    """
    numInputs = len(array[BIAS])
    numRows = len(array)

    array = np.insert(array,0, 1, axis = 1)
    array = np.delete(array,(numInputs), axis=1)

    array = array.reshape(numRows,numInputs)
    return array

def driver():
    """
    Function:   driver
    Descripion: Handles general manpu;ation for program
    Input:      none
    Output:     none
    """
    trainOutput = []
    testOutput = []
    testData = []

    # Read data into lists
    testData = readData(TEST_FILE)
    trainData = readData(TRAINING_FILE)

    # Get last ow into its own list
    for i in range(len(trainData)):
        trainOutput.append(trainData[i][PRICE_INDEX])

    for i in range(len(testData)):
        testOutput.append(testData[i][PRICE_INDEX])

    #convert lists to np.arrays
    trainOutput = np.array(trainOutput, dtype=float)
    testOutput = np.array(testOutput, dtype=float)

    trainInputs = np.array(trainData, dtype=float)
    testInputs = np.array(testData,dtype=float)

    # Add bias column to fornt and delet last col
    trainInputs = addBiasDelLast(trainInputs)
    testInputs = addBiasDelLast(testInputs)

    #reshape arary
    numRows = getNumRows(trainOutput)
    trainOutput = trainOutput.reshape(numRows,1)
    numRows =  getNumRows(testInputs)
    testOutput = testOutput.reshape(numRows,1)


    trainWeights = findW(trainInputs, trainOutput)
    print("Optimal Weights: ")
    print(trainWeights)

    trainPred = genPreds(trainInputs,trainWeights)
    trainSSE = genSSE(trainPred,trainOutput)
    testPred = genPreds(testInputs, trainWeights)
    testSSE = genSSE(testPred, testOutput)
    print("Training SSE: ")
    print(trainSSE)

    print("Test SSE")
    print(testSSE)

    # part 4
    # lambs = [0.000001,0.00001,0.0001, 0.001, 0.01, 0.1, 0, 0.0005]
    lambs = np.linspace(0.000000001,10, 10000)
    lambs = np.array(lambs)
    regularizedTrainingWeights = []
    aTrainSSE = []
    aTestSSE = []
    euclidNorms = []
    smallTrainSSE = 100000
    minLamb = -1

    print('#Regularized Model#')
    for  i in range(len(lambs)):
        weights = findWReg(trainInputs,trainOutput,lambs[i])
        norm = euclidNorm(weights)
        regularizedTrainingWeights.append(weights)
        train = findSSEReg(trainInputs,trainOutput,regularizedTrainingWeights[i],lambs[i])
        test = findSSEReg(testInputs, testOutput, regularizedTrainingWeights[i], lambs[i])
        euclidNorms.append(norm)

        aTrainSSE.append(train)
        if (train <= smallTrainSSE):
            smallTrainSSE = train
            minLamb = lambs[i]
            smallTestSSE = test
        aTestSSE.append(test)

    print("Smallest Traing SSE")
    print(smallTrainSSE)
    print('Smallest Test SSE')
    print(smallTestSSE)
    print("Optimal Lambda")
    print(minLamb)
    euclidNorms = np.array(euclidNorms)
    aTrainSSE = np.array(aTrainSSE)
    aTestSSE = np.array(aTestSSE)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(lambs, aTrainSSE, label="Lambda vs Training SSE")
    plt.plot(lambs, aTestSSE,label="Lambdas vs Test SSE")
    plt.xlabel("Lambdas")
    plt.ylabel("SSE")
    plt.title("Lambdas and SSE")
    plt.legend()

    plt.figure(2)
    plt.subplot(211)
    plt.plot(lambs, euclidNorms, label="Lambdas vs Euclidean Norms")
    plt.xlabel("Lambdas")
    plt.ylabel("Euclidean 2 norms")
    plt.title("Lambdas and Euclidean Norms")
    plt.legend()



    plt.show()

driver()








