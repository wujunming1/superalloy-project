#coding=utf8
#coding=utf8
#coding=utf8
#coding=utf8
import os
import time
import numpy as np
import pandas as pd

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
import sys

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

#选择算法操作，返回最好的模型参数，预测标签，得分情况，最好的模型（用于存储）
def select_algorithm(data,algorithm):
    pre_label = []
    sil_score = 0.0
    best = {}
    try:
        if algorithm == 'dbscan':
            best,pre_label, sil_score,model = dbscan(data)
        elif algorithm == 'kmeans':
            best,pre_label, sil_score,model = kmeans(data)
        elif algorithm == 'meanshift':
            best,pre_label, sil_score,model = meanshift(data)
        elif algorithm == 'agglomerative':
            best,pre_label, sil_score,model = agglomerative(data)
        elif algorithm == 'ward':
            best,pre_label, sil_score,model = ward(data)
        elif algorithm == 'birch':
            best,pre_label, sil_score,model = birch(data)
        elif algorithm == 'affinity':
            best,pre_label, sil_score,model = affinity(data)
    except Exception as err:
        return {},[],err,[]
    else:
        return best,pre_label, sil_score,model

#聚类算法
#DBSCAN
def hyper_dbscan(args):
    global data_file
    db = DBSCAN(eps = args["eps"], min_samples = int(args["min_samples"]), metric = args["metric"],n_jobs=-1)
    pred = db.fit_predict(data_file.data)
    temp = sil_score(data_file.data,pred)
    return -temp

def dbscan(data):
    metric_list = ['euclidean','manhattan','chebyshev']
    #对维度较大的数据，
    if data.shape[0] < 30:
        space = {
            "eps": hp.uniform("eps", 0, 2),
            "min_samples": hp.choice("min_samples", range(2, data.shape[0]-1)),
            "metric": hp.choice("metric", metric_list)
        }
    else:
        space = {
            "eps": hp.uniform("eps", 0, 2),
            "min_samples": hp.choice("min_samples", range(2, 30)),
            "metric": hp.choice("metric", metric_list)
        }
    algo = partial(tpe.suggest,n_startup_jobs = 10)
    best = fmin(hyper_dbscan, space, algo = algo, max_evals = 50)
    model = DBSCAN(eps = best["eps"], min_samples = int(best["min_samples"]+2), metric = metric_list[best["metric"]])
    return best,model.fit_predict(data),sil_score(data,model.fit_predict(data)),model.fit(data)

#KMeans
def hyper_kmeans(args):
    global data_file
    km = KMeans(n_clusters = int(args["n_iter"]),init = 'k-means++',n_init = 10,max_iter = 300,random_state = 0,n_jobs=-1)
    km.fit(data_file.data)
    pred = km.predict(data_file.data)
    temp = sil_score(data_file.data,pred)
    return -temp

def kmeans(data):
    if data.shape[0] < 30:
        space = {
            "n_iter": hp.choice("n_iter", range(1, data.shape[0]))
        }
    else:
        space = {
            "n_iter": hp.choice("n_iter", range(1, 30))
        }
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(hyper_kmeans, space, algo=algo, max_evals = 50)
    model = KMeans(n_clusters = int(best["n_iter"]+1),init = 'k-means++',n_init = 10,max_iter = 300,random_state = 0)
    model.fit(data)
    return best,model.predict(data),sil_score(data,model.predict(data)),model

#MeanShift
def hyper_meanshift(args):
    global data_file
    ms = MeanShift(bandwidth = args['bandwidth'],min_bin_freq = int(args['min_bin_freq']),n_jobs=-1)
    pred = ms.fit_predict(data_file.data)
    temp = sil_score(data_file.data,pred)
    return -temp

def meanshift(data):
    bandwidth = estimate_bandwidth(data)
    if(bandwidth - bandwidth/2) <0 and (bandwidth + bandwidth/2) >0:
        space = {
            'bandwidth': hp.uniform('bandwidth', 0, bandwidth + bandwidth / 2),
            'min_bin_freq': hp.choice('min_bin_freq', range(1, 30))
        }
    elif (bandwidth + bandwidth/2) <=0 :
        space = {
            'bandwidth': hp.uniform('bandwidth', 0.1, 1.5),
            'min_bin_freq': hp.choice('min_bin_freq', range(1, 30))
        }
    else:
        space = {
            'bandwidth': hp.uniform('bandwidth', bandwidth - bandwidth / 2, bandwidth + bandwidth / 2),
            'min_bin_freq': hp.choice('min_bin_freq', range(1, 30))
        }
    algo = partial(tpe.suggest,n_startup_jobs = 10)
    if data.shape[0] <1000:
        best = fmin(hyper_meanshift, space, algo=algo, max_evals=100)
    else:
        best = fmin(hyper_meanshift, space, algo=algo, max_evals=30)
    model = MeanShift(bandwidth = best['bandwidth'], min_bin_freq = int(best['min_bin_freq']+1))
    return best,model.fit_predict(data),sil_score(data,model.fit_predict(data)),model.fit(data)

#Agglomerative 三种连接方式：min、max、avg
def hyper_agglomerative(args):
    global data_file
    ag = AgglomerativeClustering(n_clusters = int(args['n_clusters']), affinity = args['affinity'], linkage = args['linkage'])
    pred = ag.fit_predict(data_file.data)
    temp = sil_score(data_file.data,pred)
    return -temp

def agglomerative(data):
    linkage_list = ['complete', 'average', 'single']
    affinity_list = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    if data.shape[0] < 30:
        space = {
            'n_clusters': hp.choice('n_clusters', range(1, data.shape[0])),
            'affinity': hp.choice('affinity', affinity_list),
            'linkage': hp.choice('linkage', linkage_list)
        }
    else:
        space = {
            'n_clusters': hp.choice('n_clusters', range(1, 30)),
            'affinity': hp.choice('affinity', affinity_list),
            'linkage': hp.choice('linkage', linkage_list)
        }
    algo = partial(tpe.suggest,n_startup_jobs = 10)
    best = fmin(hyper_agglomerative, space, algo = algo, max_evals =50)
    model = AgglomerativeClustering(n_clusters = int(best['n_clusters']+1), affinity = affinity_list[best['affinity']], linkage = linkage_list[best['linkage']])
    return best,model.fit_predict(data),sil_score(data,model.fit_predict(data)),model.fit(data)

#ward 使用Agglomerative中的ward方法
def hyper_ward(args):
    global data_file
    wd = AgglomerativeClustering(n_clusters=int(args['n_clusters']), affinity='euclidean', linkage='ward')
    pred = wd.fit_predict(data_file.data)
    temp = sil_score(data_file.data, pred)
    return -temp

def ward(data):
    if data.shape[0] < 30:
        space = {
            'n_clusters': hp.choice('n_clusters', range(1, data.shape[0])),
        }
    else:
        space = {
            'n_clusters': hp.choice('n_clusters', range(1, 30)),
        }
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(hyper_ward, space, algo=algo, max_evals=50)
    model = AgglomerativeClustering(n_clusters=int(best['n_clusters'] + 1), affinity='euclidean',
                                    linkage='ward')
    return best,model.fit_predict(data), sil_score(data, model.fit_predict(data)),model.fit(data)

#Birch
def hyper_birch(args):
    global data_file
    bir = Birch(threshold=args['threshold'], branching_factor=int(args['branching_factor']))
    pred = bir.fit_predict(data_file.data)
    temp = sil_score(data_file.data, pred)
    return -temp

def birch(data):
    space = {
        'threshold': hp.uniform('threshold', 0, 1),
        'branching_factor': hp.choice('branching_factor', range(25,75)),
    }
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(hyper_birch, space, algo=algo, max_evals=50)
    model = Birch(threshold=best['threshold'], branching_factor=int(best['branching_factor'] + 25))
    return best,model.fit_predict(data), sil_score(data, model.fit_predict(data)),model.fit(data)

# Affinity propagation
def hyper_affinity(args):
    global data_file
    ap = AffinityPropagation(damping = args['damping'])
    pred = ap.fit_predict(data_file.data)
    temp = sil_score(data_file.data, pred)
    return -temp

def affinity(data):
    space = {
        'damping': hp.uniform('damping',0.5, 0.99)
    }
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(hyper_affinity, space, algo=algo, max_evals=30)
    model = AffinityPropagation(damping = best['damping'])
    return best,model.fit_predict(data), sil_score(data, model.fit_predict(data)),model.fit(data)

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
def insert(file_name,file_path,alg,username):
    algorithm_list = ['dbscan', 'kmeans', 'meanshift', 'agglomerative', 'ward', 'birch', 'affinity']

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    user_name = username
    file_name = file_name
    algorithm_name = alg
    data = data_file.data
    feature_names = data_file.feature_names
    feature_count = data_file.features
    xy = visualization(data)

    start = time.time()
    parameter_list, label, score, model = select_algorithm(data, alg)
    print(label)
    end = time.time()

    index_temp = algorithm_list.index(alg)

    df_regression = pd.DataFrame(index=[file_name], columns=algorithm_list)
    df_regression_best = pd.DataFrame(index=[file_name], columns=algorithm_list)
    df_regression_label = pd.DataFrame(index=[file_name], columns=algorithm_list)
    df_regression_time = pd.DataFrame(index=[file_name], columns=algorithm_list)
    df_regression.values[0][index_temp] = score
    df_regression_best.values[0][index_temp] = parameter_list
    df_regression_label.values[0][index_temp] = label.astype(np.int).tolist()
    df_regression_time.values[0][index_temp] = end - start

    df_regression.to_csv(
        'D:\\superlloy\\automl\\cluster\\selection\\cluster_result\\' + file_name.rstrip(
            '.xls') + '_' + alg + '_result.csv')
    df_regression_best.to_csv(
        'D:\\superlloy\\automl\\cluster\\selection\\cluster_best\\' + file_name.rstrip(
            '.xls') + '_' + alg + '_best.csv')
    df_regression_label.to_csv(
        'D:\\superlloy\\automl\\cluster\\selection\\cluster_label\\' + file_name.rstrip(
            '.xls') + '_' + alg + '_label.csv')
    df_regression_time.to_csv(
        'D:\\superlloy\\automl\\cluster\\selection\\cluster_time\\' + file_name.rstrip(
            '.xls') + '_' + alg + '_time.csv')

    try:
        joblib.dump(model, 'D:\\superlloy\\automl\\cluster\\selection\\cluster_model\\' +
                    file_name.rstrip('.xls') + '_' + alg + '_model.m')
    except Exception as err:
        print(err)

    # print(date,user_name,file_name,algorithm_name,data,result_label,score,feature_names,feature_count,parameter_list,xy)

    print('Insert DataBase:')
    db = Connectdatabase()
    db.ClusterModel.insert({
        "date": date,
        "user_name": user_name,
        "file_name": file_name,
        "algorithm_name": algorithm_name,
        "data": data.astype(np.float64).tolist(),
        "result_label": label.astype(np.float64).tolist(),
        "score": score,
        "feature_names": feature_names.astype('object').tolist(),
        "feature_count": feature_count,
        # "parameter_list": parameter_list,
        "xy": xy.astype(np.float64).tolist()

    })
    print('End Insert')

if __name__ == '__main__':
    #python .py file_name0 file_path1 alg2 username3
    # file_name = 'auto-test.xlsx'
    # # file_path = '/Users/buming/Documents/Super_Alloy/SuperAlloy_System/superalloy/data/piki/auto-test.xlsx'
    # file_path = '/Users/buming/Documents/Super_Alloy/DataSet/datasets/train_done/cmc.xls'
    # alg = 'kmeans'
    # username = 'piki'
    #
    # data_file = load_data(file_path)
    # # global data_file
    #
    # insert(file_name,file_path,alg,username)
    # print(1)
    length_argv=len(sys.argv)
    print(length_argv)

    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)

    data_file = load_data(parameterlist[1])
    insert(parameterlist[0], parameterlist[1], parameterlist[2], parameterlist[3])

