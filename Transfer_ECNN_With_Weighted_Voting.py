import collections
import sys
import time

import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.io
import tensorflow as tf
from keras.utils import np_utils
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import (SGD, Adadelta, Adagrad, Adam, Adamax,
                                         Nadam, RMSprop)


def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[3]))  # 化为（145*145,200）
    # 保留特征30，copy=true:原始数据不变 whiten白化
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    # 化为145,145,30(30为输入值)
    newX = np.reshape(
        newX, (X.shape[0], X.shape[1], X.shape[2], numComponents))
    return newX, pca


def build_model(n_channel):
    model = models.Sequential()

    model.add(layers.Conv2D(64, (13, 13), input_shape=(100, 247, n_channel)))
    # model.add(layers.BatchNormalization())
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))  # 测试5%所加，非10%最优

    model.add(layers.Conv2D(128, (9, 9)))
    model.add(layers.Activation("relu"))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (5, 5), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))  # GlobalAveragePooling2D()

    model.add(layers.Flatten())
    model.add(layers.Dense(12, activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop', metrics=['accuracy'])  # rmsprop
    # model.summary()
    return model


def splitTrainTestSet(X, y, classnum=1, testRatio=0.90):  # 每类选取(1-n)%,n%作为训练、测试数据
    #     X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                test_size=testRatio, random_state=345, stratify=y)
    ss = StratifiedShuffleSplit(n_splits=classnum, test_size=testRatio,
                                train_size=1 - testRatio)

    for train_index, test_index in ss.split(X, y):
        # print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

    return X_train, X_test, y_train, y_test  # 此法生成16组train-test单只返回一组且不随机。。。


def transfer_source_model(xtrain, xtest, ytrain, xvali, yvali, batch_size, epochs):
    model = build_model(n_channel=1)  # xx=X_train.shape[2]
    start_orig = time.time()
    history = model.fit(xtrain[:, :, :, [2]], ytrain, batch_size=batch_size,
                        epochs=epochs, validation_data=(xvali[:, :, :, [2]], yvali))
    print(history.history)
    val_acc = history.history['val_accuracy']
    val_acc = val_acc[-1]
    print("源(模值)模型val_acc:", val_acc)
    pred = model.predict_classes(xtest[:, :, :, [2]])
    test11_acc = metrics.accuracy_score(y_test11, pred)
    AA = metrics.precision_score(y_test11, pred, average=None)
    AA = np.mean(AA)
    print("源(模值)模型的测试OA：", test11_acc)
    print("源(模值)模型的测试AA：", AA)
    end_orig = time.time()
    orig_running_time = end_orig - start_orig
    print("源模型运行总时间:", orig_running_time)
    # model.save('.\\transfer_simul_measur_source_model.h5')
    # print("transfer_simul_measur_source_model.h5保存完毕")
    return model, pred, val_acc, test11_acc, AA


def weight_calculation(acc):
    acc_mean = np.mean(acc)
    print("各个分类器验证精度val_acc_all:", acc, "shape:", acc.shape)
    print("各个分类器验证精度的均值:", acc_mean)
    index = np.where(acc <= acc_mean)
    acc[index] = 0
    w = acc/sum(acc)
    return w


# 创建一个txt文件，文件名为mytxtfile
def text_create(name):
    desktop_path = "C:\\Users\\rossetta\\Desktop\\"
    # 新创建的txt文件的存放路径
    full_path = desktop_path + name + '.txt'  # 也可以创建一个.doc的word文档
    file = open(full_path, 'w')


##-------------------------------开始执行---------------------------------##
# place=[[2],[0],[1],[0,1],[0,2],[1,2],[0,1,2],[3],[0,3],[1,3],[2,3],[0,1,3],[0,2,3],[1,2,3],[0,1,2,3]]
# place=[[2],[0],[1],[0,1],[0,2],[1,2],[0,1,2]]#在自己电脑验证WECNN-TL节约时间
# place=[[2],[0],[1],[0,1],[0,2],[1,2],[0,1,2],[2,3],[1,2,3],[0,1,2,3]]#在自己电脑验证WECNN学习节约时间
# place=[[2],[0]]
# Num_of_algorithm= int(input("1：CNN；2：ECNN，3：ECNN-EFSs;请输入算法："))
# Num_of_repeated_experiments= int(input("请输入重复实验次数："))
# Num_of_repeated_experiments= 5
testRatio = 0.97
validationRatio = 0.02
weighted_vote = 0  # weighted_vote=1为执行加权投票;weighted_vote=0为普通多数投票，
# 如不进行加权投票，不利用验证精度优化模型、不影响测试的独立就行了，只是略微多了验证时间
transfer = 1  # transfer=1为执行迁移学习，transfer=0为不执行迁移学习
source_batch_size = 8
source_epochs = 20  # 源模型以及不迁移学习模型的训练批次大小和轮数
transfer_batch_size = 8
transfer_epochs = 7  # 源模型以及不迁移学习模型的训练批次大小和轮数
# 设置源模型冻结层数，未冻结的可以训练0,1,2,3;4,5,6;7,8,9;10,11,12,13因此冻0（不冻结）、1、7、10层均可
num_frozen_layers = 1
if transfer == 1:
    # place = [[2], [0], [1], [0, 1], [0, 2], [1, 2], [0, 1, 2], [3], [0, 3], [1, 3], [2, 3], [0, 1, 3], [0, 2, 3],
    #          [1, 2, 3], [0, 1, 2, 3]]
    # place = [[2], [0], [1], [0, 1], [0, 2], [1, 2], [0, 1, 2]]  # 在自己电脑验证WECNN-TL节约时间
    place = [[2], [0, 1],  [0, 2], [1, 2], [
        0, 1, 2], [0], [1]]  # 在自己电脑验证PCA节约时间
    Num_of_repeated_experiments = 3  # 内存要求
elif transfer == 0:
    if weighted_vote == 0:

        place = [[2], [0], [1], [0, 1], [0, 2], [1, 2], [0, 1, 2], [3], [0, 3], [1, 3], [2, 3], [0, 1, 3], [0, 2, 3],
                 [1, 2, 3], [0, 1, 2, 3]]
        Num_of_repeated_experiments = 2
    elif weighted_vote == 1:
        # place = [[2], [0], [1], [0, 1], [0, 2], [1, 2], [0, 1, 2], [3], [0, 3], [1, 3], [2, 3], [0, 1, 3], [0, 2, 3],
        #          [1, 2, 3], [0, 1, 2, 3]]
        place = [[2], [0], [1], [0, 1], [0, 2], [1, 2], [0, 1, 2],
                 [2, 3], [1, 2, 3], [0, 1, 2, 3]]  # 在自己电脑验证WECNN学习节约时间
        Num_of_repeated_experiments = 3
print("Place:", place)
nclassifiers = len(place)  # 分类器个数与特征组合数相同

##-----------------------------------加载仿真+实测数据集-------------------------------------##
# tf_data.shape: (8570, 100, 247, 4)，gt_label.shape: (8570, 1)
print("load数据")
# 实测+仿真数据：
tf_data = h5py.File('data/tf_data.mat')
tf_data = np.transpose(tf_data['tf_data'])

gt_label = h5py.File(
    'data/gt_label')
gt_label = np.transpose(gt_label['gt_label'])
# 仿真数据：
# tf_data=h5py.File('F:/deep_learning_for_active_jamming_2020.11.16/实虚模相4通道_时域归一化_频域归一化_3.16/tf_data.mat')
# tf_data=np.transpose(tf_data['tf_data'])
#
# gt_label=h5py.File('F:/deep_learning_for_active_jamming_2020.11.16/实虚模相4通道_时域归一化_频域归一化_3.16/gt_label.mat')
# gt_label=np.transpose(gt_label['gt_label'])
# gt_label=gt_label.reshape(6000,1)

print("tf_data.shape:", tf_data.shape)
print("gt_label.shape:", gt_label.shape)
print("load数据完毕")


'''
# 划分训练、测试集合,太耗时，测试代码时放在重复实验循环外
X_train, X_test, y_train, y_test = splitTrainTestSet(tf_data, gt_label, 1, testRatio)
print('划分训练与测试+验证集合的大小：', X_train.shape, X_test.shape)
# Ratio = validationRatio / testRatio#0.02/0.98~0.02/0.92近似于0.02
X_test, X_vali, y_test, y_vali = splitTrainTestSet(X_test, y_test, 1,validationRatio)
print('划分测试和验证样本、标签的大小：', X_test.shape, y_test.shape, X_vali.shape, y_vali.shape)

y_test11 = y_test
y_train = np_utils.to_categorical(y_train)
# y_train_original= np_utils.to_categorical(y_train_original)
y_test = np_utils.to_categorical(y_test)
y_vali=np_utils.to_categorical(y_vali)
'''
# place中[2]放第一个，迁移学习中直接保存其结果，集成模型直接用？

#--------------以下为预留给重复实验的空矩阵，存入每次实验结果----------------#
time_all = np.array([])
acc_all_all = np.array([])
AA_all_all = np.array([])
average_each_classier_oa_all = np.array([])
Overall_accuracy_all = np.array([])
average_accuracy_all = np.array([])
acc_for_each_class_all = np.array([])
recall_for_each_class_all = np.array([])  # 每类recall
average_recall_all = np.array([])     # 集成Recall
f1_for_each_class_all = np.array([])  # 每类F1
average_f1_all = np.array([])          # 集成F1
kappa_all = np.array([])

# 开始实验运行模型
for m in range(Num_of_repeated_experiments):  # 重复实验取结果均值
    print('第', m + 1, '次重复实验开始：')

    # 划分训练、测试集合,太耗时，测试代码时放在重复实验循环外
    X_train, X_test, y_train, y_test = splitTrainTestSet(
        tf_data, gt_label, 1, testRatio)
    print('划分训练与测试+验证集合的大小：', X_train.shape, X_test.shape)
    # Ratio = validationRatio / testRatio#0.02/0.98~0.02/0.92近似于0.02
    X_test, X_vali, y_test, y_vali = splitTrainTestSet(
        X_test, y_test, 1, validationRatio)
    print('划分测试和验证样本、标签的大小：', X_test.shape,
          y_test.shape, X_vali.shape, y_vali.shape)
    # X_train = tf.cast(X_train, tf.float16)

    y_test11 = y_test
    y_train = np_utils.to_categorical(y_train)
    # y_train_original= np_utils.to_categorical(y_train_original)
    y_test = np_utils.to_categorical(y_test)
    y_vali = np_utils.to_categorical(y_vali)

    start = time.time()
    pre_all = np.zeros((len(y_test),))  # 存放各个分类器的（测试）预测标签
    acc_all = np.array([0])  # 存放各个分类器的测试精度
    AA_all = np.array([])
    val_acc_all = np.array([])  # 存放各个分类器的验证精度（用以求分类器权值W）
    # 迁移学习的源模型，用模值数据（通道2）训练，place中第一个就是模值，所以直接使用源模型结果，
    # 而不再进行模值->模值的迁移学习
    for i in range(nclassifiers):  # 每个子CNN的模型调用、训练并测试
        print('第', m + 1, '次重复实验中')

        if i == 0:  # 仅在第一个分类器处判断
            if transfer == 0:  # 不迁移
                print("transfer==0，不做迁移学习")
            elif transfer == 1:  # 迁移
                print("transfer==1，做迁移学习，开始训练源模型")
                transfer_model, pred, val_acc, test11_acc, AA = transfer_source_model(xtrain=X_train, xtest=X_test,
                                                                                      ytrain=y_train, xvali=X_vali, yvali=y_vali,
                                                                                      batch_size=source_batch_size, epochs=source_epochs)
                print("迁移学习源模型训练完毕！")
                pre_all = np.append(pre_all, pred, axis=0)
                acc_all = np.append(acc_all, test11_acc)
                AA_all = np.append(AA_all, AA)
                val_acc_all = np.append(val_acc_all, val_acc)
                continue

        print(1+i, '号CNN', '\n')
        # 为每个基CNN选取相应通道组合
        X_train1 = X_train[:, :, :, place[i]]
        X_test1 = X_test[:, :, :, place[i]]
        X_vali1 = X_vali[:, :, :, place[i]]

        if transfer == 1:
            X_test1 = tf.cast(X_test1, tf.float16)  # 只能如此,wecnn中运行不了太大
            for j in range(num_frozen_layers):  # 冻结对应层
                transfer_model.layers[j].trainable = False

            if i == 1:
                transfer_model.summary()

            if X_train1.shape[3] != 1:

                X_train1 = np.sum(X_train1, axis=3, keepdims=True)  # SUM
                X_test1 = np.sum(X_test1, axis=3, keepdims=True)
                X_vali1 = np.sum(X_vali1, axis=3, keepdims=True)
            history = transfer_model.fit(X_train1, y_train, batch_size=transfer_batch_size,
                                         epochs=transfer_epochs, validation_data=(X_vali1, y_vali))
            print(history.history)
            val_acc = history.history['val_accuracy']
            val_acc = val_acc[-1]
            pred = transfer_model.predict_classes(X_test1)
            test11_acc = metrics.accuracy_score(y_test11, pred)
            AA = np.mean(metrics.precision_score(y_test11, pred, average=None))
            print("第", i+1,  "个分类器的测试OA：", test11_acc)
            print("第", i + 1, "个分类器的测试AA：", AA)
        elif transfer == 0:
            model = build_model(n_channel=len(place[i]))
            history = model.fit(X_train1, y_train, batch_size=source_batch_size,
                                epochs=source_epochs, validation_data=(X_vali1, y_vali))  # 10%时15轮就可以了好像
            print(history.history)
            val_acc = history.history['val_accuracy']
            val_acc = val_acc[-1]

            pred = model.predict_classes(X_test1)
            test11_acc = metrics.accuracy_score(y_test11, pred)
            AA = np.mean(metrics.precision_score(y_test11, pred, average=None))
            print("第", i + 1, "个分类器的测试OA：", test11_acc)
            print("第", i + 1, "个分类器的测试AA：", AA)

        pre_all = np.append(pre_all, pred, axis=0)
        acc_all = np.append(acc_all, test11_acc)
        val_acc_all = np.append(val_acc_all, val_acc)
        AA_all = np.append(AA_all, AA)

    pre_all = pre_all.reshape(nclassifiers + 1, len(y_test))
    pre_all = np.delete(pre_all, 0, axis=0)
    acc_all = np.delete(acc_all, 0, axis=0)  # 每个分类器的总体精度OA

    ################# ################# ################# ################# ################
    ################# majority_vote or weighted voting################# #################
    pre_all = pre_all.astype(np.uint8)
    print('pre_all', pre_all, pre_all.shape, type(pre_all), pre_all.dtype)
    print('acc_all', acc_all, acc_all.shape, type(acc_all))
    num = len(y_test11)  # 测试样本数目
    pred = np.array([])
    if weighted_vote == 1:
        print('weighted_vote=1,加权投票算法')
        weight = weight_calculation(acc=val_acc_all)
        print("各个分类器的权值：", weight)
        for k in range(num):
            # pre_all中每行是一个分类器对所有测试标签的预测，
            bb = np.bincount(pre_all[:, k], weights=weight)
            # 每列是一个样本在各个分类器下的预测结果，统计每个样本各个分类器的预测情况
            cc = bb.argmax(axis=0)  # 返回出现次数最多的标签的索引
            pred = np.append(pred, cc)
        print('加权投票集成预测结果pred:', '\n', pred, pred.shape)
    elif weighted_vote == 0:
        for k in range(num):
            bb = np.bincount(pre_all[:, k])  # pre_all中每行是一个分类器对所有测试标签的预测，
            # 每列是一个样本在各个分类器下的预测结果，统计每个样本各个分类器的预测情况
            cc = bb.argmax(axis=0)  # 返回出现次数最多的标签的索引
            pred = np.append(pred, cc)
        print('多数投票集成预测结果pred:', '\n', pred, pred.shape)
    end = time.time()
    running_time = end - start
    print("运行时间:", running_time)
    # 结果评价指标
    average_each_classier_oa = np.mean(acc_all)  # 每个分类器OA的均值
    Overall_accuracy = metrics.accuracy_score(y_test11, pred)  # 集成OA
    acc_for_each_class = metrics.precision_score(
        y_test11, pred, average=None)  # 每类的精度
    average_accuracy = np.mean(acc_for_each_class)  # 集成AA
    recall_for_each_class = metrics.recall_score(
        y_test11, pred, average=None)  # 每类recall
    average_recall = np.mean(recall_for_each_class)  # 集成Recall
    f1_for_each_class = metrics.f1_score(y_test11, pred, average=None)  # 每类F1
    average_f1 = np.mean(f1_for_each_class)  # 集成F1
    kappa = metrics.cohen_kappa_score(y_test11, pred)  # 集成Kappa
    print("第", m + 1, "次实验", "时间:", '\n', running_time)
    print("第", m+1, "次实验", "每个分类器的OA:", '\n', acc_all)
    print("第", m+1, "次实验", "各个分类器OA的均值:", '\n', average_each_classier_oa)
    print("第", m+1, "次实验", "每个分类器的AA:", '\n', AA_all)
    print("第", m+1, "次实验", "各个分类器AA的均值:", '\n', np.mean(AA_all))
    print("第", m+1, "次实验", "集成OA:", '\n', Overall_accuracy)
    print("第", m+1, "次实验", "集成每类精度:", '\n', acc_for_each_class)
    print("第", m+1, "次实验", "集成AA:", '\n', average_accuracy)
    print("第", m+1, "次实验", "每类Recall:", '\n', recall_for_each_class)
    print("第", m+1, "次实验", "集成Recall:", '\n', average_recall)
    print("第", m+1, "次实验", "每类F1:", '\n', f1_for_each_class)
    print("第", m+1, "次实验", "集成F1:", '\n', average_f1)
    print("第", m+1, "次实验", "集成Kappa", '\n', kappa)
    # 存入重复实验的结果
    time_all = np.append(time_all, running_time)
    acc_all_all = np.append(acc_all_all, acc_all)
    AA_all_all = np.append(AA_all_all, AA_all)
    average_each_classier_oa_all = np.append(
        average_each_classier_oa_all, average_each_classier_oa)
    Overall_accuracy_all = np.append(Overall_accuracy_all, Overall_accuracy)
    average_accuracy_all = np.append(average_accuracy_all, average_accuracy)
    acc_for_each_class_all = np.append(
        acc_for_each_class_all, acc_for_each_class)
    # recall_for_each_class_all=np.append(recall_for_each_class_all,recall_for_each_class)
    average_recall_all = np.append(average_recall_all, average_recall)
    # f1_for_each_class_all=np.append(f1_for_each_class_all,f1_for_each_class)
    average_f1_all = np.append(average_f1_all, average_f1)
    kappa_all = np.append(kappa_all, kappa)
    # np.save("pred_pu_cnn.npy", pred)
    # np.save("pred_ip_ecnn.npy", pred)
    # np.save("pred_ip_ecnn_efs.npy", pred)

# 计算重复实验的均值和方差
average_each_classier_oa_all_mean = np.mean(
    average_each_classier_oa_all)  # 重复实验各个分类器OA均值的均值
average_each_classier_oa_all_std = np.std(
    average_each_classier_oa_all)  # 重复实验各个分类器OA均值的均值
time_all_mean = np.mean(time_all)  # 多次实验用时的均值
acc_for_each_class_all = np.reshape(
    acc_for_each_class_all, (Num_of_repeated_experiments, -1))
acc_for_each_class_all_mean = np.mean(
    acc_for_each_class_all, axis=0)  # 集成后每类精度AA的均值
acc_for_each_class_all_std = np.std(
    acc_for_each_class_all, axis=0)  # 集成后每类精度AA的标准差
average_accuracy_all_mean = np.mean(average_accuracy_all)  # 多次实验集成AA的均值
average_accuracy_all_std = np.std(average_accuracy_all)  # 多次实验集成AA的标准差
Overall_accuracy_all_mean = np.mean(Overall_accuracy_all)  # 多次实验集成OA的均值
Overall_accuracy_all_std = np.std(Overall_accuracy_all)  # 多次实验集成OA的标准差
average_recall_all_mean = np.mean(average_recall_all)  # 多次实验集成recall的均值
average_recall_all_std = np.std(average_recall_all)  # 多次实验集成recall的标准差
average_f1_all_mean = np.mean(average_f1_all)  # 多次实验集成f1的均值
average_f1_all_std = np.std(average_f1_all)  # 多次实验集成f1的标准差
kappa_all_mean = np.mean(kappa_all)  # 多次实验集成kappa的均值
kappa_all_std = np.std(kappa_all)  # 多次实验集成kappa的标准差

acc_all_all = acc_all_all.reshape(
    Num_of_repeated_experiments, nclassifiers)  # 每个分类器重复实验的均值精度
acc_all_all_mean = acc_all_all.mean(axis=0)
acc_all_all_std = acc_all_all.std(axis=0)
AA_all_all = AA_all_all.reshape(
    Num_of_repeated_experiments, nclassifiers)  # 每个分类器重复实验的均值精度
AA_all_all_mean = AA_all_all.mean(axis=0)
AA_all_all_std = acc_all_all.std(axis=0)
filename = 'log'
text_create(filename)
output = sys.stdout
outputfile = open(
    "C:\\Users\\dell\\Desktop\\" + filename + '.txt', 'w')
sys.stdout = outputfile
# 将显示重复实验结果改为显示其均值±标准差
print("实验场景：干扰分类")
print("nclassifiers =", nclassifiers, file=outputfile)
print("测试样本占:", testRatio, file=outputfile)
print("验证样本占:", validationRatio, file=outputfile)
print("算法：集成CNN")
print("重复实验次数：", Num_of_repeated_experiments, file=outputfile)
print("重复实验时间均值:", '\n', time_all_mean, file=outputfile)
print("重复实验每个分类器的OA(仅看看)：", '\n', acc_all_all, file=outputfile)
print("重复实验各个分类器OA的均值(仅看看):", '\n', acc_all_all_mean, file=outputfile)
print("重复实验各个分类器OA的方差(仅看看):", '\n', acc_all_all_std, file=outputfile)
print("重复实验各个分类器OA的均值的均值(仅看看):", '\n',
      average_each_classier_oa_all_mean, file=outputfile)
print("重复实验各个分类器OA的均值的方差(仅看看):", '\n',
      average_each_classier_oa_all_std, file=outputfile)

print("重复实验每个分类器的AA(仅看看)：", '\n', AA_all_all, file=outputfile)
print("重复实验各个分类器AA的均值(仅看看):", '\n', AA_all_all_mean, file=outputfile)
print("重复实验各个分类器AA的方差(仅看看):", '\n', AA_all_all_std, file=outputfile)
print("重复实验各个分类器AA均值的均值(仅看看):", '\n', np.mean(AA_all_all_mean), file=outputfile)
print("重复实验各个分类器AA均值的方差(仅看看):", '\n', np.std(AA_all_all_mean), file=outputfile)

print("重复实验集成OA的均值:", '\n', Overall_accuracy_all_mean, file=outputfile)
print("重复实验集成OA的标准差:", '\n', Overall_accuracy_all_std, file=outputfile)
print("重复实验集成AA的均值:", '\n', average_accuracy_all_mean, file=outputfile)
print("重复实验集成AA的标准差:", '\n', average_accuracy_all_std, file=outputfile)
print("重复实验集成每类精度AA的均值:", '\n', acc_for_each_class_all_mean, file=outputfile)
print("重复实验集成每类精度AA的标准差:", '\n', acc_for_each_class_all_std, file=outputfile)
print("重复实验集成Recall的均值:", '\n', average_recall_all_mean, file=outputfile)
print("重复实验集成Recall的标准差:", '\n', average_recall_all_std, file=outputfile)
print("重复实验集成f1的均值:", '\n', average_f1_all_mean, file=outputfile)
print("重复实验集成f1的标准差:", '\n', average_f1_all_std, file=outputfile)
print("重复实验集成kappa的均值:", '\n', kappa_all_mean, file=outputfile)
print("重复实验集成kappa的标准差:", '\n', kappa_all_std, file=outputfile)
outputfile.close()  # close后才能看到写入的数据
