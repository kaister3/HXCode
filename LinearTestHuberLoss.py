'''
术语介绍：
=================
高斯核
局部加权：对每个样本点调用一次数据集，分配不同的权重计算回归系数

'''

# -*- coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
 
def loadDataSet(fileName):
    """
    函数说明:加载数据
    Parameters:
        fileName - 文件名
    Returns:
        xArr - x数据集
        yArr - y数据集
    Website:
        https://www.cuijiahua.com/
    Modify:
        2017-11-12
    """
 
    numFeat = len(open(fileName).readline().split('\t')) - 1
    #数据的列数，三列分别为1.0，x轴，y轴
    xArr = []
    yArr = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr =[]
        curLine = line.strip().split('\t')
        #去除首尾空格，按制表符分隔
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
            #读出字符串,float（）将其变为数字
            #linear存放每列的前两个
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
        #y轴数据存入y轴 
    return xArr, yArr
 
def plotDataSet():
    """
    函数说明:绘制数据集
    Parameters:
        无
    Returns:
        无
    Website:
        https://www.cuijiahua.com/
    Modify:
        2017-11-12
    """
    xArr, yArr = loadDataSet('ex0.txt')
    # 加载数据集
    n = len(xArr)
    # 数据个数
    xcord = []
    ycord = []
    # 初始化样本点
    for i in range(n):                                                   
        xcord.append(xArr[i][1])
        # 存入x轴值
        ycord.append(yArr[i])
        # 存入y轴值
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # 画布分割成1行1列，在第一个子图添加subplot
    ax.scatter(xcord, ycord, s = 20, c = 'blue',alpha = .5)                  #绘制样本点

    plt.title('DataSet')                                                     #绘制title
    plt.xlabel('X')
    plt.show()

def standRegres(xArr, yArr):
    """
    函数说明:平方误差公式计算回归系数w
    Parameters:
        xArr - x数据集
        yArr - y数据集
    Returns:
        ws - 回归系数
    Website:
        https://www.cuijiahua.com/
    Modify:
        2017-11-12
    """
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 将x，y转换成矩阵
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("奇异矩阵，无法求解")
        return
    ws = xTx.I * (xMat.T * yMat)
    # 又称 θ(theta)
    return ws

def standRegresWithHuber(theta, xArr, yArr, delta = 3):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 将x，y转换成矩阵
    diffmat = xMat - (theta * yMat)

def plotRegression():
    """
    函数说明:绘制回归曲线和数据点
    Parameters:
        无
    Returns:
        无
    Website:
        https://www.cuijiahua.com/
    Modify:
        2017-11-12
    """
    # xAAr, yAAr = loadDataSet('.\\abalone\\abalone.txt')
    xAAr, yAAr = loadDataSet('ex0.txt')
    ws = standRegres(xAAr, yAAr)
    xMat = np.mat(xAAr)
    yMat = np.mat(yAAr)
    xCopy = xMat.copy()
    xCopy.sort(0)
    yHat = xCopy * ws
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xCopy[:,1], yHat, c = 'red')
    ax.scatter(xMat[:,1].flatten().A[0], yMat.flatten().A[0], s = 20, c = 'blue', alpha = 0.3)
    plt.title('Dataset')
    plt.xlabel('X')
    plt.show()

if __name__ == '__main__':
    # 线性回归
    plotRegression()