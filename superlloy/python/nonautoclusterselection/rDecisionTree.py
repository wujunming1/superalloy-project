#coding=utf8
import os
import time
import numpy as np
import pandas as pd
import sys

from sklearn.impute import SimpleImputer
from sklearn import tree
from sklearn.metrics import explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import pymongo
import matplotlib.pyplot as plt

#读取数据文件后构建文件的类
class Bunch(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __getstate__(self):
        return self.__dict__

#读取文件操作
def load_data(filename):
    print(filename)
    data_file = pd.read_excel(filename)

    #对于分类的数据集，直接取出最后一列标签向量
    # data_file.drop([data_file.columns.values[-1]], axis = 1, inplace = True)

    data_file = pd.get_dummies(data_file) #处理字符串

    imr = SimpleImputer(missing_values=np.nan, strategy='mean') #处理空缺值
    imr = imr.fit(data_file)
    data = imr.transform(data_file.values)
    n_samples = data.shape[0]

    feature_names = data_file.columns.values
    n_features = feature_names.shape[0]


    return Bunch(sample=n_samples, features=n_features, data=data,
                 feature_names=feature_names)

##DecisionTree
def DecisionTree(data,depth):
    x = data[:,:-1]
    y = data[:,-1]
    x_train, x_test, y_train, y_test = train_test_split (x, y, test_size = 0.3, random_state = 0)
    model = tree.DecisionTreeRegressor(max_depth=depth)
    model.fit(x_train, y_train)
    score = model.score(x_test, y_test)
    pred = model.predict(x_test)
    # pred = np.rint(pred)
    print(score)
    xy=[list(x) for x in zip(pred,y_test)]
    # print(pred)
    # y_pre=model.predict(x_test)
    # m=explained_variance_score(y_test, y_pre)
    # print(m)
    # print(model.score(x_test,y_test))
    return model,pred,y_test,score,xy

#PCA降维
#通过PCA降维将数据维度降成二维数据，从而在坐标系中显示数据，完成可视化功能
def visualization(data):
    print('Starting PCA')
    pca=PCA(n_components=2)
    X=pca.fit_transform(data)
    # print(X)
    print('End PCA')
    return X

#链接MongoDB数据库
def Connectdatabase():
    conn=pymongo.MongoClient(host="localhost",port=27017)
    db=conn.MongoDB_Data
    return db

#插入数据操作
def insert(file_name,file_path,alg,depth,username):
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    user_name = username
    file_name = file_name
    algorithm_name = alg
    data = data_file.data
    feature_names = data_file.feature_names
    feature_count = data_file.features
    xy = visualization(data)
    depth=int(depth)
    try:
        model, label, y_test, score, xy = DecisionTree(data,depth)
    except Exception as err:
        print(err)

    try:
        joblib.dump(model, 'E:\\automachine\\superlloy\\cluster\\cluster_model\\' +
                    file_name.rstrip('.xls') + '_' + alg + '_model.m')
    except Exception as err:
        print(err)

    # print(date,user_name,file_name,algorithm_name,data,result_label,score,feature_names,feature_count,parameter_list,xy)

    print('Insert DataBase:')
    db = Connectdatabase()
    db.NonAutoClusterModel.insert({
        "date": date,
        "user_name": user_name,
        "file_name": file_name,
        "algorithm_name": algorithm_name,
        "data": data.astype(np.float64).tolist(),
        "result_label": label.astype(np.float64).tolist(),
        "label_true":y_test.astype(np.float64).tolist(),
        "score": score,
        "feature_names": feature_names.astype('object').tolist(),
        "feature_count": feature_count,
        "xy": xy

    })
    print('End Insert')

if __name__ == '__main__':
    # file_name = 'auto-test.xlsx'
    # file='E:\\automachine\\superlloy\\data\\piki\\boston.xls'
    # data_file = load_data(file)
    # model=DecisionTree(data_file.data,20)
    # alg='linear'
    # name='piki'
    # # insert(file_name,file,alg,name)
    length_argv = len(sys.argv)
    print(length_argv)

    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)

    data_file = load_data(parameterlist[1])
    insert(parameterlist[0], parameterlist[1], parameterlist[2], parameterlist[3], parameterlist[4])