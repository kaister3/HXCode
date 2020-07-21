import pandas as pd
from matplotlib.font_manager import FontProperties
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(fileName):
    xArr = []
    yArr = []
    dataset = pd.read_csv(fileName)
    X = dataset.iloc[:, 0].values
    Y = dataset.iloc[:, 1].values
    return X, Y

def lwlr(testPoint, X, Y, k = 1.0):
    xMat = np.mat(X)
    yMat = np.mat(Y).T
    m = np.shape(xMat)[0]
    # print('m =', m)
    weight = np.mat(np.eye((1)))
    diffMat = testPoint - xMat
    # print(diffMat)
    weight[0, 0] = np.exp(diffMat * diffMat.T / (-2 * k * k))
    # print(weight[0,0])
    xTx = xMat.T * (weight * xMat)
    if np.linalg.det(xTx) == 0.0:
        # print("奇异矩阵，无法求解")
        return
    ws = xTx.I * (xMat.T * (weight * yMat))
    return testPoint * ws

def lwlrTest(testArr, X, Y, k = 1.0):
    m = np.shape(testArr)[0]
    # print("m = ", m)
    yHat = np.zeros(m)
    
    for i in range(m):
        yHat[i] = lwlr(testArr[i], X, Y, k)
    return yHat

def plotlwlrRegression():
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    X, Y = loadDataSet('Salary_Data.csv')
    # print (X)
    # print (Y)
    yHat_1 = lwlrTest(X, X, Y, 1.0)
    yHat_1 = lwlrTest(X, X, Y, 0.01)
    yHat_1 = lwlrTest(X, X, Y, 0.005)

    xMat = np.mat(X)
    yMat = np.mat(Y)
    srtInd = xMat[:, 0].argsort(0)
    fig, axs = plt.subplots(nrows=3, ncols=1,sharex=False, sharey=False, figsize=(10,8))                                        

    plt.show()

if __name__ == "__main__":
    plotlwlrRegression()