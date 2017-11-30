# coding=utf-8
import pandas as pd
import numpy as np
from decimal import *
from pandas import DataFrame


# np.set_printoptions(threshold=np.inf)


def Train(X_train, Y_train, feature):
    global class_num, label
    class_num = 2  # 分类数目
    label = [2, 4]  # 分类标签
    feature_len = 9  # 特征长度

    prior_probability = np.zeros(( class_num))  # 初始化先验概率
    print(prior_probability.shape)
    conditional_probability = np.zeros((class_num, feature_len + 1, feature_len))  # 初始化条件概率,
    print(conditional_probability.shape)

    conditional_num = np.zeros((class_num, feature_len + 1, feature_len))  # 需要记住的数有2类，9组特征，每一组10个值，是一个2*10*9

    positive_count = 0  # 统计正类
    negative_count = 0  # 统计负类
    for i in range(len(Y_train)):
        if Y_train[i] == 2:
            positive_count += 1
        else:
            negative_count += 1
    print("benign", positive_count)
    print("malignant", negative_count)
    prior_probability[0] = positive_count / float(len(Y_train))  # 求得正类的先验概率
    prior_probability[1] = negative_count / float(len(Y_train))  # 求得负类的先验概率

    print(prior_probability)

    # conditional_probability是一个

    # 分为两个类别
    for i in range(class_num):
        # 对特征按行遍历
        for j in range(feature_len + 1):  # 总共10行，10个取值（1,2,3,4,5,6，7,8,9,10）
            # 遍历数据集，并依次做判断
            for k in range(len(Y_train)):
                if Y_train[k] == label[i]:  # 相同类别
                    if X_train[k][0] == feature[j][0]:  # 判断i类第j行（特征1取定值j）个数
                        conditional_num[i][j][0] += 1
                    if X_train[k][1] == feature[j][1]:  # 判断i类第j行（特征2取定值j）个数
                        conditional_num[i][j][1] += 1
                    if X_train[k][2] == feature[j][2]:  # 判断i类第j行（特征3取定值j）个数
                        conditional_num[i][j][2] += 1
                    if X_train[k][3] == feature[j][3]:  # 判断i类第j行（特征4取定值j）个数
                        conditional_num[i][j][3] += 1
                    if X_train[k][4] == feature[j][4]:  # 判断i类第j行（特征5取定值j）个数
                        conditional_num[i][j][4] += 1
                    if X_train[k][5] == feature[j][5]:  # 判断i类第j行（特征6取定值j）个数
                        conditional_num[i][j][5] += 1
                    if X_train[k][6] == feature[j][6]:  # 判断i类第j行（特征7取定值j）个数
                        conditional_num[i][j][6] += 1
                    if X_train[k][7] == feature[j][7]:  # 判断i类第j行（特征8取定值j）个数
                        conditional_num[i][j][7] += 1
                    if X_train[k][8] == feature[j][8]:  # 判断i类第j行（特征9取定值j）个数
                        conditional_num[i][j][8] += 1
    conditional_num = conditional_num.astype(np.int32)
    print(conditional_num)



    
    class_label_num = [positive_count, negative_count]  # 存放各类型的数目
    for i in range(class_num):
        for j in range(feature_len + 1):
            conditional_probability[i][j][0] = conditional_num[i][j][0] / float(class_label_num[i])  # 求得i类j行第1个特征的条件概率
            conditional_probability[i][j][1] = conditional_num[i][j][1] / float(class_label_num[i])  # 求得i类j行第2个特征的条件概率
            conditional_probability[i][j][2] = conditional_num[i][j][2] / float(class_label_num[i])  # 求得i类j行第3个特征的条件概率
            conditional_probability[i][j][3] = conditional_num[i][j][3] / float(class_label_num[i])  # 求得i类j行第4个特征的条件概率
            conditional_probability[i][j][4] = conditional_num[i][j][4] / float(class_label_num[i])  # 求得i类j行第5个特征的条件概率
            conditional_probability[i][j][5] = conditional_num[i][j][5] / float(class_label_num[i])  # 求得i类j行第6个特征的条件概率
            conditional_probability[i][j][6] = conditional_num[i][j][6] / float(class_label_num[i])  # 求得i类j行第7个特征的条件概率
            conditional_probability[i][j][7] = conditional_num[i][j][7] / float(class_label_num[i])  # 求得i类j行第8个特征的条件概率
            conditional_probability[i][j][8] = conditional_num[i][j][8] / float(class_label_num[i])  # 求得i类j行第9个特征的条件概率
    print(conditional_probability)
    return prior_probability, conditional_probability


# 给定数据进行分类


def Predict(testset, prior_probability, conditional_probability, feature):
    result = np.zeros((len(label)))    #存放估计为不同标签的概率
    feature_len = 9  # 特征长度
    for i in range(class_num):
        for j in range(feature_len + 1):
            if feature[j][0] == testset[0]:
                P0 = conditional_probability[i][j][0]
            if feature[j][1] == testset[1]:
                P1 = conditional_probability[i][j][1]
            if feature[j][2] == testset[2]:
                P2 = conditional_probability[i][j][2]
            if feature[j][3] == testset[3]:
                P3 = conditional_probability[i][j][3]
            if feature[j][4] == testset[4]:
                P4 = conditional_probability[i][j][4]
            if feature[j][5] == testset[5]:
                P5 = conditional_probability[i][j][5]
            if feature[j][6] == testset[6]:
                P6 = conditional_probability[i][j][6]
            if feature[j][7] == testset[7]:
                P7 = conditional_probability[i][j][7]
            if feature[j][8] == testset[8]:
                P8 = conditional_probability[i][j][8]
        result[i] = P0 * P1 * P2 * P3 * P4 * P5 * P6 * P7 * P8 * prior_probability[i]
    print(result)
    k = 0
    if(result[0] > result[1]):
        k = label[0]
    else :
        k = label[1]
    return k

def main():
    print(getcontext())
    getcontext().prec = 50
    f_r = open('Breastdata.txt', "r")
    f_r = f_r.read().split("\n")  # f_r中每一行文本内容就是list中的一个元素
    for i in f_r:
        if "?" in i:
            f_r.remove(i)

    f_o = open("remove.txt", "w")
    for i in f_r:
        f_o.write(i + "\n")

    f_o.close()

    data = np.loadtxt(open('remove.txt'), dtype=np.int32, delimiter=',')
    # sp = data.shape

    X_train = data[:, 1:10]
    Y_train = data[:, 10]
    print(X_train)
    print(Y_train)
    feature_len = 9

    feature = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
               [2, 2, 2, 2, 2, 2, 2, 2, 2],
               [3, 3, 3, 3, 3, 3, 3, 3, 3],
               [4, 4, 4, 4, 4, 4, 4, 4, 4],
               [5, 5, 5, 5, 5, 5, 5, 5, 5],
               [6, 6, 6, 6, 6, 6, 6, 6, 6],
               [7, 7, 7, 7, 7, 7, 7, 7, 7],
               [8, 8, 8, 8, 8, 8, 8, 8, 8],
               [9, 9, 9, 9, 9, 9, 9, 9, 9],
               [10, 10, 10, 10, 10, 10, 10, 10, 10]
               ]

    testset = [2, 1, 2, 1, 2, 1, 3, 1, 1]

    prior_probability, conditional_probability = Train(X_train, Y_train, feature)
    result = Predict(testset, prior_probability, conditional_probability, feature)
    print(result)


if __name__ == '__main__':
    main()


