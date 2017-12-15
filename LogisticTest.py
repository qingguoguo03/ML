import numpy as np


def loadDataSet(dataPath=r'C:\datasets\books\machinelearninginaction\Ch05\testSet.txt'):
    dataLi = []
    label = []
    with open(dataPath, 'r') as file:
        for line in file.readlines():
            lineArr = line.strip().split()
            dataLi.append([1.0, float(lineArr[0]), float(lineArr[1])])
            label.append(int(lineArr[2]))
        file.close()
    print('数据集总共有： ', len(dataLi), ' 样本 ')
    return dataLi, label


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels): # batch size 用矩阵计算比较方便
    weightsDynamic = []
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))  # 因为下面的dataMat是矩阵 所以不能写成n了
    weightsDynamic.append(weights)
    for i in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights = weights + alpha * dataMat.transpose() * error  # 这里用的就是logistic里面的likelihood最大
        weightsDynamic.append(weights)
    return weights, weightsDynamic

def plotBestFit(weights, dataArrIn, classLabels):
    import matplotlib.pyplot as plt
    if type(weights) == np.matrix:
        weights = weights.getA()
    dataArr = np.array(dataArrIn)
    n = np.shape(dataArr)[0]
    xcord1 = [];
    ycord1 = [];
    xcord2 = [];
    ycord2 = [];
    for i in range(n):
        if classLabels[i] == 1:
            xcord1.append(dataArr[i, 1])
            ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1])
            ycord2.append(dataArr[i, 2])
    figure = plt.figure()
    ax = figure.add_subplot(111)  # 分割几行几列 取第几个
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')  # #在坐标系中画点，前两个参数为点坐标，marker=s为方形，s=30为大小,c=red为坐标颜色
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def plotWeights(weightsDynamic):
    import matplotlib.pyplot as plt
    iterNum = len(weightsDynamic)
    ycord0 = []
    ycord1 = []
    ycord2 = []
    for weight in weightsDynamic:
        ycord0.append(weight[0])
        ycord1.append(weight[1])
        ycord2.append(weight[2])
    figure = plt.figure()
    ax0 = figure.add_subplot(311)
    # new_ticks = np.linspace(0, 10, 10)
    # ax0.set_yticks(new_ticks)
    ax1 = figure.add_subplot(312)
    # new_ticks = np.linspace(0, 1, 10)
    # ax1.set_yticks(new_ticks)
    ax2 = figure.add_subplot(313)
    # new_ticks = np.linspace(1, -1, 10)
    # ax2.set_yticks(new_ticks)
    x = np.arange(0,iterNum,1)
    ax0.plot(x, ycord0)
    ax1.plot(x, ycord1)
    ax2.plot(x, ycord2)
    # plt.xticks(np.linspace(0, 500, 25, endpoint=True))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

def stoGradAscent1(dataArrIn, classLabels, numIter=150):
    weightsDynamic = []
    dataArr = np.array(dataArrIn)
    m, n = np.shape(dataArr)
    weights = np.ones(n)
    weightsDynamic.append(weights)
    for j in range(numIter):
        dataIndex = list(range(m))
        # alpha = 0.01
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01 # 设置动态变化
            randIndex = int(np.random.uniform(0, len(dataIndex)))  # numpy.random.uniform(low,high,size) 功能:从一个均匀分布[low,high)中随机采样,注意定义域是左闭右开
            index = dataIndex[randIndex]
            randIndex, index  = index, randIndex
            h = sigmoid(sum(dataArr[randIndex] * weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataArr[randIndex] # 这里不能用 += 不然是直线 这和python的指针引用有关
            weightsDynamic.append(weights)
            del(dataIndex[index])  # 一直都觉得这里的随机 书上是错的 但是网上都没人说 好奇怪 这为改后抽样
    return weights, weightsDynamic


if __name__ == '__main__':

    # dataLi, label = loadDataSet()
    # # weights, weightsDynamic = gradAscent(dataLi, label)
    # # print('weight: ', weights)
    # weights, weightsDynamic = stoGradAscent1(dataLi, label, 200)
    # plotBestFit(weights, dataLi, label)
    # plotWeights(weightsDynamic)
    a = 5
    print(a)