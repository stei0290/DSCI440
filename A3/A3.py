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



def driver():


    testData = []
    trainData = []
    trainOutput = []
    testOutput = []

    testData = readCSVData(TEST_FILE)
    trainData = readCSVData(TRAIN_FILE)
    testLen = len(testData[0])
    trainlen = len(trainData[0])
    testLabel_Index = testLen - BIAS
    trainLabel_Index = trainlen - BIAS

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

    # Add bias column to front and delete last last column
    trainInputs = addBiasDelLast(trainInputs)
    testInputs = addBiasDelLast(testInputs)

    #reshape arary
    numRows = getNumRows(trainOutput)
    trainOutput = trainOutput.reshape(numRows,1)
    numRows =  getNumRows(testInputs)
    testOutput = testOutput.reshape(numRows,1)


    # print(trainInputs)


driver()