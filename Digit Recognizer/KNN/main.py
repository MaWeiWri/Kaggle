# -*- coding:utf-8 -*-
'''
Created on 2014年9月9日

@author: 威
'''
import numpy as np
import random

import KNN

# corss validation中使用不同的k，得到不同的分类准确度
K_RANGE = range(1,100,10)
# 分类test数据时使用的k
K = 5
# 训练数据文件
INPUT_FILE = '../train.csv'
# 官方提供的knn benchmark
VALIDATION_FILE_1 = '../knn_benchmark.csv'
# 官方提供的random forest benchmark
VALIDATION_FILE_2 = '../rf_benchmark.csv'
# 需要被分类的数据
TEST_FILE = '../test.csv'
# 分类结果输出路径
RESULT_FILE ='../KNNResult.csv'

def loadTrainingData(input_file):
    '''
    读取训练数据集，返回两个列表
    第一个列表为训练数据的feature， m*n维
    第二个列表为训练数据的label
    '''
    # 打开文件
    fp = open(input_file)
    # 用于存放训练数据的feature
    data = []
    # 用于存放训练数据的label
    label = []
    # 是否是第一行，由于第一行是字段名，所以要跳过
    isFirstLine = True
    for line in fp:
        # 去掉换行符
        line = line.strip()
        # 如果是第一行则跳过
        if isFirstLine:
            isFirstLine = False
            continue
        # 第一个字段是label
        isLabel = True
        # 用于存放一行记录的feature
        temp  = []
        features = line.split(",")
        for f in features:
            # 第一个字段是label，加入到label中
            if isLabel:
                label.append(f)
                isLabel = False
            # 后面的字段是feature，把他们放入到数组中去
            else:
                temp.append(int(f))
        # 将一行记录的所有feature放入到data这个二维列表中
        data.append(temp)
    # 返回两个值
    # 第一个值为训练数据集中的feature值，每一行代表一张图片的feature
    # 第二个值为训练数据集中的label值，每一行代表一张图片所属的类
    return data,label

def loadBenchmarkData(input_file):
    '''
    读取benchmark文件
    返回一个列表，第i个元素代表knn benchmark对第i幅图片的分类
    '''
    # 打开文件
    fp = open(input_file)
    # 用于存放数据
    data = []
    # 是否是第一行，由于第一行是字段名，所以要跳过
    isFirstLine = True
    for line in fp:
        # 去掉换行符
        line = line.strip()
        # 如果是第一行则跳过
        if isFirstLine:
            isFirstLine = False
            continue
        # 第二行开始为benchmark分类的图片id与结果
        infos = line.split(",")
        # 将结果按顺序加入到列表中
        data.append(infos[1])
    # 返回已经分类好的结果
    return data

def loadTestData(input_file):
    '''
    读取Test文件
    '''
    # 打开文件
    fp = open(input_file)
    # 用于存放数据
    data = []
    # 是否是第一行，由于第一行是字段名，所以要跳过
    isFirstLine = True
    for line in fp:
        # 去掉换行符
        line = line.strip()
        # 如果是第一行则跳过
        if isFirstLine:
            isFirstLine = False
            continue
        # 用于存放一行记录的feature
        temp  = []
        features = line.split(",")
        # 把一行记录的feature放入到列表中去
        for f in features:
            temp.append(int(f))
        # 将一行记录的所有feature放入到data这个二维列表中
        data.append(temp)
    # 返回需要被分类数据的feature
    return data


def crossValidation(data,label,percent = 0.95):
    '''
    使用训练数据集的一部分作交叉验证
    默认以95%的数据作为训练数据，留5%的数据作交叉验证
    两部分的数据没有交集，都是从原始训练数据中随机抽样得到
    '''
    # 获取事物总数
    m = len(data)
    # 获取training set 的数量
    numOfTrain = int(m*percent)
    # 生成所有事物的index
    indexRange = range(m)
    
    # 对index进行采样,作为我们的training set
    trainIndex = random.sample(indexRange,numOfTrain)
    # 剩下的作为cross validation set用于交叉验证
    crossValidationIndex = set(indexRange).difference(set(trainIndex))
    # 将其转换成列表
    crossValidationIndex = list(crossValidationIndex)
    
    # 用于存放训练数据的feature
    trainingData = []
    # 用于存放训练数据的label
    trainingLabel = []
    for i in trainIndex:
        trainingData.append(data[i])
        trainingLabel.append(label[i])
        
    # 用于存放验证数据的feature
    crossValidationData = []
    # 用于存放验证数据的label
    crossValidationLabel = []
    for i in crossValidationIndex:
        crossValidationData.append(data[i])
        crossValidationLabel.append(label[i])
    # 创建一个字典，保存不同k情况下的分类错误率
    errorRate = {}
    # 获取验证数据集的数量
    crossValidationDataSize = len(crossValidationData)
    print '交叉验证训练集:' + str(m-crossValidationDataSize)
    print '交叉验证验证集:' + str(crossValidationDataSize)
    # 使用不同的k做crossValidation
    for k in K_RANGE:
        knn = KNN.KNN(np.array(trainingData),trainingLabel,k,False)
        # count为计数器，记录正确分类的事务数
        count = 0.0
        # 对验证数据进行分类
        for i in range(crossValidationDataSize):
            result = knn.classify(np.array(crossValidationData[i]))
            # 如果分类正确，则count+1
            if result == crossValidationLabel[i]:
                count +=1
        # 计算分类错误率，并建起放入到字典中
        errorRate[k] = 1-float(count/crossValidationDataSize)
        print 'K=' + str(k) +'时，分类准确率为' + str(errorRate[k])
    # 返回交叉验证结果
    print errorRate
    
    

    
    

if __name__ == '__main__':
    
    print 'Now loading training data !'
    train_set,train_label = loadTrainingData(INPUT_FILE)
    print 'Training data loaded !'
    
#     print 'Now loading knn_benchmark data !'
#     validation_set_1 = loadBenchmarkData(VALIDATION_FILE_1)
#     print 'Knn_benchmark data loaded !'
    
#     print 'Now loading rf_benchmark data !'
#     validation_set_2 = loadBenchmarkData(VALIDATION_FILE_2)
#     print 'rf_benchmark data loaded !'
    
    print 'Now loading test data !'
    test_set = loadTestData(TEST_FILE)
    print 'Test data loaded !'

    # 交叉验证
    crossValidation(np.array(train_set),train_label)

    # 使用预设的k进行knn分类
    knn = KNN.KNN(np.array(train_set),train_label,K,False)
    # 获取待分类的数据数量
    m = len(test_set)
    # 用于存放最后分类的结果
    totalResult=[] 
    # 对每一个数据进行分类
    for i in range(m):
        result = knn.classify(np.array(test_set[i]))
        print str(i)+','+str(result)
        # 将分类结果加入到totalResult列表中，最后需要使用它输出结果
        totalResult.append(str(i+1)+','+str(result) + '\n')
    # 将分类结果写入到文件中去
    resultFile = open(RESULT_FILE,'w')
    resultFile.writelines(totalResult)
            
    

    
    
    
    
    
    
    
