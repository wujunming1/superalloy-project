import csv
import os
import sys
import time
import pymongo
import pickle
from bson.binary import Binary
import json
import pandas as pd
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import estimate_bandwidth

from sklearn.metrics import silhouette_samples
from sklearn.metrics import adjusted_rand_score
# from sklearn.metrics import calinski_harabaz_score

from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor

# from sklearn.grid_search import GridSearchCV

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

#链接MongoDB数据库
def Connectdatabase():
    conn=pymongo.MongoClient(host="localhost",port=27017)
    db=conn.MongoDB_Data
    return db

#读取文档操作。返回值：sample-数据个数；feature-特征个数；data-数据值；target-最后一列值-该iris数据集中表示花的种类；
#                  feature_names-数据特征的名称；
def load_data(filename):
    with open(filename) as f:
        data_file=csv.reader(f)
        data=[]
        target=[]
        temp=next(data_file)
        feature_names=np.array(temp)
        for i, d in enumerate(data_file):
            temp = []
            for j in d:
                if j == '':
                    j = 0
                temp.append(j)
            data.append(np.asarray(temp[:-1], dtype=np.float))
            target.append(np.asarray(d[-1], dtype=np.float))
        data = np.array(data)
        target = np.array(target)
        n_samples = data.shape[0]
        n_features = data.shape[1]
    return Bunch(sample=n_samples, features=n_features, data=data, target=target,
                 feature_names=feature_names)



#其中一种聚类质量的定量分析方法——轮廓分析法，用于度量簇中样本聚集的密集程度
#内聚度a：某一样本x与簇内其他点之间的平均距离
#分离度b：样本与其最近簇中所有点之间的平均距离看作是与下一最近簇的分离度
#轮廓系数是（b-a）/max{b，a}
#该函数画出了轮廓系数图返回轮廓系数均值，这里暂时主要是用于评价DBSCAN的聚类质量
def silhouette(data,labels):
    cluster_labels=np.unique(labels)
    n_cluster=cluster_labels.shape[0]
    silhouette_vals=silhouette_samples(data,labels,metric='euclidean')
    y_ax_lower,y_ax_upper=0,0
    yticks=[]
    for i,c in enumerate(cluster_labels):
        c_silhouette_vals=silhouette_vals[labels==c]
        c_silhouette_vals.sort()
        y_ax_upper += len(c_silhouette_vals)
        color=cm.jet(i/n_cluster)
        plt.barh(range(y_ax_lower,y_ax_upper),
                 c_silhouette_vals,
                 height=1.0,
                 edgecolor='none',
                 color=color)
        yticks.append((y_ax_lower+y_ax_upper)/2)
        y_ax_lower+=len(c_silhouette_vals)
    silhouette_avg=np.mean(silhouette_vals)
    plt.axvline(silhouette_avg,
                color="red",
                linestyle="--")
    plt.yticks(yticks,cluster_labels+1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette coefficient')
    plt.show()
    return silhouette_avg

#该函数使用的是Adjusted Rand index 指标评价聚类的好坏
#主要是根据真实类别与预测类别用一定方法分析得到预测的正确值
#详情见sklearn.cluster中的最后一节的第一个方法
def accuracy(labels,right_labels):
    accuracy=adjusted_rand_score(labels,right_labels)
    return accuracy

#返回descan预测的值，由自己输入半径和簇类最小数目
def dbscan_labels(data,eps,min_samples):
    model=DBSCAN(eps=eps,min_samples=min_samples)
    labels=model.fit_predict(data)
    return labels

#该函数是为了计算dbscan算法性能指标最好的半径值，指定的k值是簇类最小个体数目
#详细原理见http://shiyanjun.cn/archives/1288.html
#k值使用的是经验值4，目前觉得还没有必要对k值进行最佳值判断
def select_best_eps(data,k):
    distance=[]
    k_distance=[]
    row,col=data.shape[0],data.shape[1]
    for i in range(row):
        dis = []
        for j in range(row):
            dist=np.linalg.norm(data[i]-data[j])
            dis.append(dist)
        dis.sort()
        k_distance.append(dis[k])
        distance.append(dis)
    k_distance=np.array(k_distance)
    k_distance.sort()
    n_list=list(range(k_distance.shape[0]))
    delta=[]
    for i in range(k_distance.shape[0]):
        if(i==0):
            temp=0
        else:
            temp=k_distance[i]-k_distance[i-1]
        delta.append(temp)
    delta_distance=dict(zip(n_list,delta))
    delta_distance=sorted(delta_distance.items(),key=lambda d:d[1],reverse=True)
    return k_distance[delta_distance[0][0]]

#dbscan的函数
#显示两种性能指标，其中adjusted rand index目前是用于所有聚类算法的性能评价，average silhouette目前只是用于dbscan
def dbscan(data,right_labels):
    best_eps=select_best_eps(data,4)
    labels=dbscan_labels(data,best_eps,4)
    adjusted_rand_score=accuracy(labels,right_labels)
    return labels,adjusted_rand_score

#KMEANS、KMODES、KPROTOTYPE
#Kmeans:度量的是数值型数据，使用的是欧氏距离
#Kmodes:度量的是非数值属性数据，比较的是属性之间相同数目的大小，不同D加1，越大越不相似。更新modes使用最代表的属性
#Kprototype:使用混合属性，D=P1+a*P2，P1 kmeans，P2 kmodes

#kmeans函数，直接调用sklearn的函数，由于k的值是由专家决定，所以这里不用函数选取最佳的k值
#如有需要，可以使用肘方法确定簇的最佳数量。
def kmeans(data,k,right_labels):
    model=KMeans(n_clusters=k,init='k-means++')
    labels=model.fit_predict(data)
    adjusted_rand_score = accuracy(labels, right_labels)
    return labels,adjusted_rand_score

#kmodes函数,使用的是kmodes库中的函数
#GitHub链接：https://github.com/nicodv/kmodes
#参数有n_clusters、max_iter、cat_dissim:function used for categorical variabels default:matching_dissim matching,找的是不同的
#init:Huang、Cao、random,default:Cao,初始化方法，具体目前不了解
#n_init,verbose
def kmodes(data,k,right_labels):
    model=KModes(n_clusters=k)
    labels=model.fit_predict(data)
    adjusted_rand_score=accuracy(labels,right_labels)
    return labels, adjusted_rand_score

#MeanShift函数,使用的是sklearn库
#评判标准是簇类点到中心点的向量之和为零
#bandwidth高斯核函数的带宽
#seeds初始化质心得到的方式，bin_seeding在没有设置seeds时使用，none表示全部点都为质心
#min_bin_freq设置最少质心数目
#cluster_all，True表示所有点都会被聚类，False表示孤立点类标签为-1
#n_jobs多线程，-1所有cpu，1不使用，-2只有一块不被使用，-3有两块
def meanshift(data,k,right_labels):
    bandwidth=estimate_bandwidth(data)
    model=MeanShift(bandwidth=bandwidth,bin_seeding=True,min_bin_freq=k,n_jobs=-1)
    labels=model.fit_predict(data)
    adjusted_rand_score = accuracy(labels, right_labels)
    return labels, adjusted_rand_score

#agglomerativeClustering的三种模式
#两类之间某两个样本之间的距离
#ward_linkage ward minimizes the variance of the clusters being merged.离差平方和
#complete_linkage最大距离
#average_linkage平均距离
def agglomerative(data,k,link,right_labels):
    if link not in ['ward','complete','average']:
        link='ward'
    model=AgglomerativeClustering(n_clusters=k,affinity='euclidean',linkage=link)
    labels = model.fit_predict(data)
    adjusted_rand_score = accuracy(labels, right_labels)
    return labels, adjusted_rand_score

#birch聚类
# def birch(data,th,br,right_labels):
def birch(data,k,right_labels):
    model=Birch(n_clusters=k)
    # model=Birch(threshold=th,branching_factor=br,n_clusters=None)
    labels=model.fit_predict(data)
    adjusted_rand_score = accuracy(labels, right_labels)
    return labels,adjusted_rand_score

#SpectralClustering谱聚类
#n_clusters:簇的个数;affinity:相似矩阵建立方法，nearest_neighbors，precomputed:自定义，全连接法，自定义核函数，默认rbf高斯核函数;
#gamma:核函数参数，默认1.0，gamma
#degree:核函数参数，d
#coef0:核函数参数，r
def spectralclustering(data,k,right_labels):
    best_gamma_score={}
    for index,gamma in enumerate((0.01,0.1,1,10)):
        model=SpectralClustering(n_clusters=k,gamma=gamma)
        labels=model.fit_predict(data)
        accuracy_gamma=accuracy(labels,right_labels)
        best_gamma_score[gamma]=accuracy_gamma
    best_gamma_score=sorted(best_gamma_score.items(),key=lambda d:d[1],reverse=True)
    model_best = SpectralClustering(n_clusters=k, gamma=best_gamma_score[0][0])
    labels = model_best.fit_predict(data)
    adjusted_rand_score=accuracy(labels,right_labels)
    return labels, adjusted_rand_score

#通过PCA降维将数据维度降成二维数据，从而在坐标系中显示数据，完成可视化功能
def visualization(data):
    print('Starting PCA')
    pca=PCA(n_components=2)
    # pca.fit(data)
    X=pca.fit_transform(data)
    print(X)
    print('End PCA')
    return X

#利用LOF算法去检测是否有离群点，返回lof_label，false则没有离群点，True则有离群点
def LOF(data):
    model=LocalOutlierFactor()
    labels=model.fit_predict(data)
    lof_label=False
    for i in range(len(labels)):
        if(labels[i]==-1):
            lof_label=True
            break
    return lof_label

#算法汇总选择
#策略，K，数值型混合型标记，可解释标记，数据文件名编码，样本数目，特征数目，数据，分类标签，特征的名字，用户名

#非自动化算法

def NonAuto_Cluster(Algorithm,K,filename,sample,features,data,target,feature_names,username):
    #正确分类标签
    right_labels=target
    K=int(K)
    #五个标签
    mix_label='undefined'
    explain_label='undefined'
    dimension_label='undefined'
    number_label='undefined'
    noise_label='undefined'

    print("sample_number is",sample)
    XY=visualization(data)
    print('Starting Cluster')
    if(Algorithm=='KMeans'):
        print('KMeans:')
        labels,score=kmeans(data,K,right_labels)
    elif(Algorithm=='KModes'):
        print('KModes:')
        labels,score = kmodes(data, K,right_labels)
    elif(Algorithm=='MeanShift'):
        print('MeanShift:')
        labels,score = meanshift(data, K,right_labels)
    elif(Algorithm=='Agglomerative'):
        print('Agglomerative:')
        labels,score = agglomerative(data, K,'ward',right_labels)
    elif(Algorithm=='Birch'):
        print('Birch')
        labels,score = birch(data,K,right_labels)
    elif(Algorithm=='DBSCAN'):
        print('DBSCAN:')
        labels,score=dbscan(data,right_labels)
    elif(Algorithm=='Spectral Clustering'):
        print('Spectral Clustering:')
        labels,score = spectralclustering(data, K,right_labels)
    insert(username, filename, Algorithm, K, mix_label, explain_label, dimension_label, number_label,noise_label,
                   sample, data, features, feature_names, target, labels, score, XY)
    return
#自动化算法
#KMeans,KModes,MeanShift,Agglomerative,Birch,DBSCAN,Spectral Clustering
def Auto_Cluster(K,mix_label,explain_label,filename,sample,features,data,target,feature_names,username):
    right_labels = target
    K = int(K)
    print("sample_number is", sample)
    XY = visualization(data)
    print('Starting Cluster')
    dimension=len(feature_names)-1
    number=sample

    # 数值型：0，混合型：1，可解释：1，不可解释：0
    # 五个标签
    # mix_label = 'undefined'
    # explain_label = 'undefined'
    dimension_label = 'undefined'
    number_label = 'undefined'
    noise_label = LOF(data)
    if(mix_label==1):
        labels,score=kmodes(data,K,right_labels)
        insert(username, filename, 'FreeSelect_KModes', K, mix_label, explain_label, dimension_label, number_label, noise_label,
               sample, data, features, feature_names, target, labels, score, XY)
    else:
        #
        if(dimension/number>1):
            dimension_label=True
            labels,score=spectralclustering(data, K,right_labels)
            insert(username, filename, 'FreeSelect_Spectral Clustering', K, mix_label, explain_label, dimension_label, number_label,
                   noise_label,
                   sample, data, features, feature_names, target, labels, score, XY)
        else:
            dimension_label=False
            #检测使用算法，将样本数的是否走向调换了
            if(number/dimension<1):
                number_label=True
                if(noise_label==True):
                    labels, score = spectralclustering(data, K, right_labels)
                    insert(username, filename, 'FreeSelect_Spectral Clustering', K, mix_label, explain_label, dimension_label, number_label,noise_label,
                           sample, data, features, feature_names, target, labels, score, XY)
                else:
                    if(explain_label==1):
                        #GMM
                        birch_labels,birch_score=birch(data,K,right_labels)
                        insert(username, filename, 'Birch', K, mix_label, explain_label, dimension_label,
                               number_label, noise_label,
                               sample, data, features, feature_names, target, birch_labels, birch_score, XY)
                    else:
                        kmeans_labels,kmeans_score=kmeans(data,K,right_labels)
                        insert(username, filename, 'KMeans', K, mix_label, explain_label, dimension_label,
                               number_label, noise_label,
                               sample, data, features, feature_names, target, kmeans_labels, kmeans_score, XY)
                        meanshift_labels,meanshift_score=meanshift(data,K,right_labels)
                        insert(username, filename, 'MeanShift', K, mix_label, explain_label, dimension_label,
                               number_label, noise_label,
                               sample, data, features, feature_names, target, meanshift_labels, meanshift_score, XY)
                        if(kmeans_score>=meanshift_score):
                            insert(username, filename, 'FreeSelect_KMeans', K, mix_label, explain_label, dimension_label,
                                   number_label, noise_label,
                                   sample, data, features, feature_names, target, kmeans_labels, kmeans_score, XY)
                        else:
                            insert(username, filename, 'FreeSelect_MeanShift', K, mix_label, explain_label, dimension_label,
                                   number_label, noise_label,
                                   sample, data, features, feature_names, target, meanshift_labels, meanshift_score, XY)
            else:
                number_label=False
                if(noise_label==True):
                    dbscan_labels, dbscan_score = dbscan(data,right_labels)
                    insert(username, filename, 'DBSCAN', K, mix_label, explain_label, dimension_label,
                           number_label, noise_label,
                           sample, data, features, feature_names, target, dbscan_labels, dbscan_score, XY)
                    spectral_labels, spectral_score = spectralclustering(data, K, right_labels)
                    insert(username, filename, 'Spectral Clustering', K, mix_label, explain_label, dimension_label,
                           number_label, noise_label,
                           sample, data, features, feature_names, target, spectral_labels, spectral_score, XY)
                    if (dbscan_score >= spectral_score):
                        insert(username, filename, 'FreeSelect_DBSCAN', K, mix_label, explain_label, dimension_label,
                               number_label, noise_label,
                               sample, data, features, feature_names, target, dbscan_labels, dbscan_score, XY)
                    else:
                        insert(username, filename, 'FreeSelect_Spectral Clustering', K, mix_label, explain_label, dimension_label,
                               number_label, noise_label,
                               sample, data, features, feature_names, target, spectral_labels, spectral_score, XY)
                else:
                    if(explain_label==1):
                        birch_labels, birch_score = birch(data, K, right_labels)
                        insert(username, filename, 'Birch', K, mix_label, explain_label, dimension_label,
                               number_label, noise_label,
                               sample, data, features, feature_names, target, birch_labels, birch_score, XY)
                        #GMM
                        agglomerative_labels,agglomerative_score=agglomerative(data, K,'ward',right_labels)
                        insert(username, filename, 'Agglomerative', K, mix_label, explain_label, dimension_label,
                               number_label, noise_label,
                               sample, data, features, feature_names, target, agglomerative_labels, agglomerative_score, XY)

                        max_score=max(birch_score,agglomerative_score)
                        if(max_score==birch_score):
                            insert(username, filename, 'FreeSelect_Birch', K, mix_label, explain_label, dimension_label,
                                   number_label, noise_label,
                                   sample, data, features, feature_names, target, birch_labels, birch_score, XY)
                        elif(max_score==agglomerative_score):
                            insert(username, filename, 'FreeSelect_Agglomerative', K, mix_label, explain_label, dimension_label,
                                   number_label, noise_label,
                                   sample, data, features, feature_names, target, agglomerative_labels,
                                   agglomerative_score, XY)
                    else:
                        kmeans_labels, kmeans_score = kmeans(data, K, right_labels)
                        insert(username, filename, 'KMeans', K, mix_label, explain_label, dimension_label,
                               number_label, noise_label,
                               sample, data, features, feature_names, target, kmeans_labels, kmeans_score, XY)
                        meanshift_labels, meanshift_score = meanshift(data, K, right_labels)
                        insert(username, filename, 'MeanShift', K, mix_label, explain_label, dimension_label,
                               number_label, noise_label,
                               sample, data, features, feature_names, target, meanshift_labels, meanshift_score, XY)
                        if (kmeans_score >= meanshift_score):
                            insert(username, filename, 'FreeSelect_KMeans', K, mix_label, explain_label,
                                   dimension_label,
                                   number_label, noise_label,
                                   sample, data, features, feature_names, target, kmeans_labels, kmeans_score, XY)
                        else:
                            insert(username, filename, 'FreeSelect_MeanShift', K, mix_label, explain_label,
                                   dimension_label,
                                   number_label, noise_label,
                                   sample, data, features, feature_names, target, meanshift_labels, meanshift_score, XY)

    return

#KMeans,KModes,MeanShift,Agglomerative,Birch,DBSCAN,Spectral Clustering
#插入MongoDB数据库
def insert(username,filename,Algorithm,K,mix_label,explain_label,dimension_label,number_label,noise_label,sample,data,features,feature_names,target,labels,score,XY):
    print('The score of ' + Algorithm + ' is:', score)
    print('Insert DataBase:')
    db = Connectdatabase()
    db.clusterModel.insert({
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "type": "Cluster",
        "username": username,
        "data_name": filename,
        "algorithm": Algorithm,
        "k_NUM": K,
        "mix_label": mix_label,
        "explain_label": explain_label,
        "dimension_label": dimension_label,
        "number_label": number_label,
        "noise_label": noise_label,
        "sample_count": sample,
        "data": data.astype(np.float64).tolist(),
        "features_count": features,
        "features_names": feature_names.astype('object').tolist(),
        "right_labels": target.astype('object').tolist(),
        "result_labels": labels.astype(np.float64).tolist(),
        "score": float(score),
        "xy": XY.astype(np.float64).tolist()
    })
    print('End Insert')
    # print('The result_labels of ' + Algorithm + ' is:')
    # print(labels)

# python /superalloy/python/cluster/Cluster.py KMeans 3 0 0 glass.csv /superalloy/data/piki/glass.csv piki
if __name__ == '__main__':

    #python 算法文件路径 算法名称0 K1 数值型混合型标记2 可解释标记3 数据文件名编码4 调用文件路径5 用户名6
    #算法路径
    # Algorithm_path='/编程文件/高温合金机器学习平台/聚类算法/Cluster.py'
    # #算法名称 0duedue
    # Algorithm='kmeans'
    # #K1
    # K=4
    # #数据地址 2
    # Data_path='/编程文件/高温合金机器学习平台/聚类算法/iris_data.csv'
    # #user 3
    # user_name='piki'
    # iris=load_data('iris_data.csv')
    # LOF(iris.data)
    length_argv=len(sys.argv)
    print(length_argv)

    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)

    #自动化
    if(length_argv==7):
        inputdata = load_data(parameterlist[4])
        print('inputdata is:')
        print(inputdata.data)
        Auto_Cluster(parameterlist[0],parameterlist[1],parameterlist[2],parameterlist[3],inputdata.sample,inputdata.features,inputdata.data,inputdata.target,inputdata.feature_names,parameterlist[5])
    elif(length_argv==6):
        inputdata = load_data(parameterlist[3])
        print('inputdata is:')
        print(inputdata.data)
        NonAuto_Cluster(parameterlist[0],parameterlist[1],parameterlist[2],inputdata.sample,inputdata.features,inputdata.data,inputdata.target,inputdata.feature_names,parameterlist[4])






# def Cluster(Algorithm,K,mix_label,explain_label,filename,sample,features,data,target,feature_names,username):
