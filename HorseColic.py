import requests
import os
import pandas as pd
import numpy as np
from LogisticTest import stoGradAscent1, sigmoid

URL_1 = 'http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.data'
URL_2 = 'http://archive.ics.uci.edu/ml/machine-learning-databases/horse-colic/horse-colic.test'


def get_data_from_web(url, save_filepath):
    text = requests.get(url).text
    with open(save_filepath, 'w') as file:
        file.write(text)
        file.close()
    print('已存储在本地目录：', os.path.dirname(save_filepath))
    return text


def get_data_from_file(filepath):
    dataSet = pd.read_table(filepath, header=None, delim_whitespace=True)
    columns = list(dataSet.columns)
    if dataSet[23].dtype != 'int64':  # 说明缺失数据
        dataSet = dataSet[dataSet[23] != '?']  # 去掉缺失label的数据
    for column in columns:
        if dataSet[column].dtype != 'int64':
            dataSet[column][dataSet[column] == '?'] = '0'  # 缺失值赋值为0
            try:
                dataSet[column] = dataSet[column].astype(np.int64)  # 强制转换数据类型
            except ValueError:
                dataSet[column] = dataSet[column].astype(np.float)
    # 根据对比 貌似删除了第三列以及第24列以后的数据，第24列是分类的label “已经死亡”、“已经安乐死”合并成为“未能存活”
    dataSet[23][dataSet[23] != 1] = 0
    columns = list(set(columns) - set([2, 24, 25, 26, 27]))
    dataSet[columns].to_csv(os.path.dirname(filepath) + os.sep + 'prep' + os.path.split(filepath)[-1].split('_')[-1],
                            index=False)
    print('预处理数据完成，保存在目录 ', os.path.dirname(filepath))


def load_dataset(filepath):
    dataSet = pd.read_csv(filepath)
    columns = list(dataSet.columns)
    labelArr = dataSet[columns[-1]].get_values()
    dataArr = dataSet[columns[:-1]].get_values()
    return dataArr, labelArr


def classifyLabel(inX, weights):
    prob = sigmoid(sum(inX * weights))
    if prob > 0.5:
        return 1
    else:
        return 0

def colicTest():
    filepath1 = r'C:\datasets\books\machinelearninginaction\Ch05\prepdata.txt'
    filepath2 = r'C:\datasets\books\machinelearninginaction\Ch05\preptest.txt'
    dataArr, labelArr = load_dataset(filepath1)
    weights, _ = stoGradAscent1(dataArr, labelArr, numIter=200)
    dataArr, labelArr = load_dataset(filepath2)
    errorCnt = 0
    m, _ = np.shape(dataArr)
    # 使用sigmiod函数分类的时候可能由于数字过小出现溢出警告, 别人的解法是使用bigfloat或忽略警告
    for i in range(m):
        if classifyLabel(dataArr[i],weights) != labelArr[i]:
            errorCnt += 1
    error_rate = errorCnt / m
    print('the error rate of this test is: {:.2f}'.format(error_rate))
    return error_rate

def multiTest(numIter):  # 多次测试算平均错误率  这和我们的十折交叉验证啥的不一样
    error_sum = 0.0
    for i in range(numIter):
        error_sum += colicTest()
    print('%d次测试之后的平均错误率是 %.2f' % (numIter, error_sum/numIter))

if __name__ == '__main__':
    # colicTest()
    multiTest(10)