import numpy as np

def loadDataSet(dataPath=r'E:\deep learning\papers\machinelearninginaction\Ch05\testSet.txt'):
    dataMat = []
    label = []
    with open(dataPath, 'r') as file:
        for line in file.readlines():
            lineArr = line.strip().split()
            dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
            label.append(int(lineArr[2]))
        file.close()
    return dataMat, label


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    dataMat = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    m, n = np.shape(dataMat)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n,1)) # 因为下面的dataMat是矩阵 所以不能写成n了
    print('weights: ', weights)
    for i in range(maxCycles):
        h = sigmoid(dataMat * weights)
        error = labelMat - h
        weights = weights + alpha * dataMat.transpose() * error  # 这里用的就是logistic里面的likelihood最大
    return weights


def stoGradAscent(dataMatrix, classLabels):
    m, n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones((n)) #注意这里一定不是(n,1)以为下面dataMatrix的维数和上一个函数不一样了
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        print('alpha * error * dataMatrix[i]: ',alpha * error * dataMatrix[i])
        weights = weights + alpha * error * dataMatrix[i]
        print('weights2: ', weights2 + alpha * error * dataMatrix[i])
        print('add weights: ', weights)
        break
    return weights


def plotBestFit(weights, dataMat, labelMat):
    import matplotlib.pyplot as plt
    if type(weights) == np.matrix:
        weights = weights.getA()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = [];
    ycord1 = [];
    xcord2 = [];
    ycord2 = [];
    for i in range(n):
        if labelMat[i] == 1:
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

def stoGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    print(weights)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):

            alpha = 4/(1.0+j+i)+0.01
            randIndex = int(np.random.uniform(0,len(dataIndex))) # numpy.random.uniform(low,high,size) 功能:从一个均匀分布[low,high)中随机采样,注意定义域是左闭右开
            # index = dataIndex[randIndex]
            h = sigmoid(sum(dataMatrix[i]*weights))
            error = classLabels[randIndex] - h
            weights += alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])

    return weights

dataMat, label = loadDataSet()
# weights = gradAscent(dataMat, label)
# weights = stoGradAscent(np.array(dataMat), label)
weights = stoGradAscent1(np.array(dataMat), label, )
print(weights)
plotBestFit(weights, dataMat, label)
# 明天需要看下logistic的求导公式怎么退出
