import requests
import os
import pandas as pd
import numpy as np
# from LogisticTest import stoGradAscent1

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
    label_index = columns[-1]
    if dataSet[label_index].dtype !=  'int64': # 说明缺失数据
        dataSet = dataSet[dataSet[label_index] != '?'] # 去掉缺失label的数据
    for column in columns:
        if dataSet[column].dtype != 'int64':
            dataSet[column][dataSet[column] == '?'] = '0' # 缺失值赋值为0
            try:
                dataSet[column] = dataSet[column].astype(np.int64)  # 强制转换数据类型
            except ValueError :
                dataSet[column] = dataSet[column].astype(np.float)
    # 根据对比 貌似删除了第三列
    del(columns[2])
    dataSet[columns].to_csv(os.path.dirname(filepath)+os.sep+'prep'+os.path.split(filepath)[-1].split('_')[-1], index=False)
    print('预处理数据完成，保存在目录 ',os.path.dirname(filepath))

def load_dataset(filepath):
    dataSet = pd.read_csv(filepath)
    columns = list(dataSet.columns)
    label = list(dataSet[columns[-1]])
    dataLi = list(dataSet[columns[:-1]])
    return dataLi, label





if __name__ == '__main__':
    filepath1 = r'C:\datasets\books\machinelearninginaction\Ch05\prepdata.txt'
    filepath1 = r'C:\datasets\books\machinelearninginaction\Ch05\preptest.txt'

    dataLi, label = load_dataset(filepath1)
    print(dataLi)
    print(label)