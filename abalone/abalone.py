# -*- coding:utf-8 -*-
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

def loadDataSet(filename):
    '''
    :description
        读取数据集
    :input
        filename        文件名
    :return
        xArr            X轴数据集
        yArr            Y轴数据集
    :param
        lineArr         单行读取出的数组中的数字
        curLine         每行读出的粗数据
    :modify
        2020-07-16
    '''
    numFeat = len(open(filename).readline().split('\t')) - 1
    xArr = []
    yArr = []
    fp = open(filename)
    for line in fp.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat):
            lineArr.append(float(curLine[i]))
        xArr.append(lineArr)
        yArr.append(float(curLine[-1]))
        # 只有最后一列是 y 轴的值
    return xArr, yArr

def lwlr(testpoint, xArr, yArr, k = 1.0):
    '''
    :description
        使用局部加权线性回归计算回归系数w
    :input
        testpoint   测试点
        xArr        x轴数据集
        yArr        y轴数据集
        k           高斯核
    :param
        xMat        x数据矩阵
        yMat        y数据矩阵
        m           x数据矩阵的行数
        weights     局部权重矩阵
        diffMat     中间变量—公式里的分子
    :return
        ws          回归系数 * 测试点
    :modify
        2020-07-16
    '''
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    m = np.shape(xMat)[0]
    weights = np.mat(np.eye(m))
    for j in range(m):
        diffMat = testpoint - xMat[j, :]
        weights[j, j] = np.exp(diffMat * diffMat.T/(-2 * k * k))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) == 0.0:
        print("奇异矩阵，无法求解")
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testpoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    '''
    :description
        局部加权线性回归测试
    :input
        testArr         测试数据集
        xArr            x数据集
        yArr            y数据集
        k               高斯核
    :param

    :return
        yHat            预测值
    :modify
        2020-07-16
    '''
    m = np.shape(testArr)[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def standRegres(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T * xMat
    if np.linalg.det(xTx) == 0.0:
        print("奇异矩阵，无法求解")
        return
    ws = xTx.I * (xMat.T * yMat)
    return ws

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr)**2).sum()

if __name__ == "__main__":
    abX, abY = loadDataSet('abalone.txt')
    print('训练集与测试集相同:局部加权线性回归,核k的大小对预测的影响:')
    yHat01 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[0:99], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:',rssError(abY[0:99], yHat01.T))
    print('k=1  时,误差大小为:',rssError(abY[0:99], yHat1.T))
    print('k=10 时,误差大小为:',rssError(abY[0:99], yHat10.T))
    print('')
    print('训练集与测试集不同:局部加权线性回归,核k的大小是越小越好吗？更换数据集,测试结果如下:')
    yHat01 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 0.1)
    yHat1 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 1)
    yHat10 = lwlrTest(abX[100:199], abX[0:99], abY[0:99], 10)
    print('k=0.1时,误差大小为:',rssError(abY[100:199], yHat01.T))
    print('k=1  时,误差大小为:',rssError(abY[100:199], yHat1.T))
    print('k=10 时,误差大小为:',rssError(abY[100:199], yHat10.T))
    print('')
    print('训练集与测试集不同:简单的线性归回与k=1时的局部加权线性回归对比:')
    print('k=1时,误差大小为:', rssError(abY[100:199], yHat1.T))
    ws = standRegres(abX[0:99], abY[0:99])
    yHat = np.mat(abX[100:199]) * ws
    print('简单的线性回归误差大小:', rssError(abY[100:199], yHat.T.A))