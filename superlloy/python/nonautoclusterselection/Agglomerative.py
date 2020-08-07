#coding=utf8
import os
import time
import numpy as np
import pandas as pd
import sys

from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import AffinityPropagation

from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
from hyperopt import fmin,tpe,hp,partial
from sklearn.externals import joblib
from sklearn.decomposition import PCA
import pymongo

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

#评估标准，采取的是轮廓系数评价指标
def sil_score(data,labels):
    try:
        score = silhouette_score(data,labels)
    except Exception as err:
        # print(err)
        #报错返回 -1
        return -1
    else:
        return score

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

#agglomerative
def agglomerative(data,n_clusters,linkage,affinity):
    # linkage_list = ['complete', 'average', 'single']
    # affinity_list = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    # ag = AgglomerativeClustering(n_clusters = int(args['n_clusters']), affinity = args['affinity'], linkage = args['linkage'])

    db = AgglomerativeClustering(n_clusters = n_clusters, affinity = affinity, linkage = linkage)
    db.fit(data)
    pred = db.fit_predict(data)
    score = sil_score(data,pred)
    print(score)
    return db,pred,score

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
def insert(file_name,file_path,alg,n_clusters,affinity,linkage,username):
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    user_name = username
    file_name = file_name
    algorithm_name = alg
    data = data_file.data
    feature_names = data_file.feature_names
    feature_count = data_file.features
    xy = visualization(data)

    print(n_clusters)
    print('linkage',linkage)
    print('affinity',affinity)
    n_clusters = int(float(n_clusters))

    # try:
    #     model, labels, score = agglomerative(data, n_clusters, linkage, affinity)
    # except Exception as err:
    #     print(err)

    model, labels, score = agglomerative(data, n_clusters, linkage, affinity)



    try:
        joblib.dump(model, 'D:\\superlloy\\cluster\\cluster_model\\' +
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
        "result_label": labels.astype(np.float64).tolist(),
        "score": score,
        "feature_names": feature_names.astype('object').tolist(),
        "feature_count": feature_count,
        "xy": xy.astype(np.float64).tolist()

    })
    print('End Insert')

if __name__ == '__main__':
    #python .py file_name0 file_path1 agglomerative2 n_clusters3,affinity4,linkage5  username6
    # linkage_list = ['complete', 'average', 'single']
    # affinity_list = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    # file_name = 'auto-test.xlsx'
    # file_path = '/Users/buming/Documents/Super_Alloy/SuperAlloy_System/superalloy/data/piki/auto-test.xlsx'
    # # file_path = '/Users/buming/Documents/Super_Alloy/DataSet/datasets/train_done/cmc.xls'
    # alg = 'kmeans'
    # min_samples = 2
    # username = 'piki'
    #
    # data_file = load_data(file_path)
    # # global data_file
    #
    # insert(file_name,file_path,alg,min_samples,username)
    # print(1)
    length_argv=len(sys.argv)
    print(length_argv)

    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)

    data_file = load_data(parameterlist[1])
    insert(parameterlist[0], parameterlist[1], parameterlist[2], parameterlist[3],parameterlist[5],parameterlist[4],parameterlist[6])

