import pandas as pd
import numpy as np
import math
import scipy.stats
import random

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
import pymongo
import time
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

#基于模型的推荐算法
#冷启动问题计算两个算法的性能，返回算法列表和性能列表
#输入 Bunch中的data属性，需要冷启动计算的算法名称列表
#输出 冷启动计算的算法名称的性能值
def Model_calculate_cluster(data,cluster_list):
    perform_list = []
    for alg_temp in cluster_list:
        best, pre_label, sil_score, model = select_algorithm(data, alg_temp)
        perform_list.append(sil_score)
    return perform_list

#输入 Bunch中的data属性，性能矩阵值，冷启动中算法名称列表，算法名称列表
#输出 推荐的两种算法
def Model_calculation(data, train,cluster_list,alg):
    #构建一个与模型性能矩阵相同的临时变量，用于在最后一行插入冷启动的的性能向量
    train_result_list_temp = train
    #得到冷启动的算法的性能
    perform_list = Model_calculate_cluster(data, cluster_list)
    #用模型性能矩阵中对应算法的平均值去填补性能向量
    perform_list_temp = np.zeros(len(alg))
    for i in range(len(alg)):
        for j in range(len(cluster_list)):
            if alg[i] == cluster_list[j]:
                perform_list_temp[i] = perform_list[j]
                break
            else:
                perform_list_temp[i] = np.mean(np.array(train)[:, i])
    perform_list_temp = perform_list_temp.tolist()
    #将新数据的性能向量填充到临时变量的最后一行
    train_result_list_temp.append(perform_list_temp)
    #对完全矩阵进行svd矩阵分解
    u, sigma, vt = np.linalg.svd(train_result_list_temp)
    sig3 = np.mat([[sigma[0], 0, 0], [0, sigma[1], 0], [0, 0, sigma[2]]])
    pre = u[:, :3] * sig3 * vt[:3, :]
    #得到矩阵分解再重组后的最后一行性能向量
    pre_list = pre[-1].tolist()[0]
    #对性能向量进行降序排序，取出前两个评价指标较好的算法作为推荐算法。
    score_dict = {}
    for i in range(len(pre_list)):
        score_dict[alg[i]] = pre_list[i]
    score_dict_sorted = sorted(score_dict.items(), key=lambda x: x[1],reverse=True)
    alg_list = []
    for i in range(2):
        alg_list.append(score_dict_sorted[i][0])
    return alg_list

#集成基于模型的推荐算法
#输入 性能矩阵（包括文件名），测试数据，冷启动算法列表，聚类算法列表
#输出 推荐的两种算法
def Model_Recommend(result_values, data, cluster_list, alg):
    train_list = [temp.tolist()[1:] for temp in result_values]
    Model_alg_list = Model_calculation(data, train_list, cluster_list, alg)
    return Model_alg_list

#基于物品的推荐算法
#欧式距离计算，欧式距离越小代表相似度越大
def Item_similarity_E(vec1, vec2):
    # print(vec1,vec2)
    np_vec1 = np.array(vec1)
    np_vec2 = np.array(vec2)
    result = np.sqrt(np.sum(np.square(np_vec1-np_vec2)))
    return result

#冷启动问题计算两个算法的性能，返回算法列表和性能列表
#输入 Bunch中的data属性，需要冷启动计算的算法名称列表
#输出 冷启动计算的算法名称的性能值
def Item_calculate_cluster(data,cluster_list):
    perform_list = []
    for alg_temp in cluster_list:
        best, pre_label, sil_score, model = select_algorithm(data, alg_temp)
        perform_list.append(sil_score)
    return perform_list

#计算性能矩阵的topk文件
#输入 冷启动的算法性能向量，冷启动算法名称列表，性能矩阵（包括文件名），聚类算法列表，取出topk文件的数目
#输出 topk的相似样本（文件名+欧式距离）
def Item_topk(perform_list,cluster_list, result_values, alg, k):
    #计算冷启动算法在聚类算法列表中的序号
    index_list = [alg.tolist().index(cluster_temp) for cluster_temp in cluster_list]
    #组合模型中的冷启动算法的性能矩阵
    cluster_perform_list = [[temp[i+1] for i in index_list] for temp in result_values]
    #性能矩阵中的文件名列表
    train_file_name_list = [temp[0] for temp in result_values]
    #字典，键是相似样本文件名，值是欧式距离
    similarity_dict = {}
    for i in range(len(cluster_perform_list)):
        temp = Item_similarity_E(cluster_perform_list[i], perform_list)
        similarity_dict[train_file_name_list[i]] = temp
    #将欧氏距离升序排序，值越小表示相似度越大
    similarity_dict_sorted = sorted(similarity_dict.items(),key=lambda x: x[1])
    #取出前k个相似样本
    topk_result = similarity_dict_sorted[:k]
    return topk_result

#计算推荐的算法集
#输入 topk的相似样本（文件名+欧式距离），归一化的性能矩阵，聚类算法列表
#输出 推荐的两种算法
def Item_calculation(topk_result, norm_values,alg):
    file_name = [temp[0] for temp in topk_result]
    #得到对应文件的得分
    score_list = []
    for name in file_name:
        for item in norm_values:
            if name == item[0]:
                score_list.append(item[1:])

    alg_list = []
    score_avg = {}
    for i in range(len(alg)):
        sum = 0
        for j in range(len(file_name)):
            temp = score_list[j][i]
            #其中轮廓系数为-1.0的值变为1.0进行计算，因为norm中是与max的差值，越小越好，所以这样改变
            if temp == -1.0:
                temp = 1.0
            sum += temp
        score_avg[alg[i]] = sum
    score_avg_sorted = sorted(score_avg.items(),key=lambda x: x[1])
    #top2个算法
    for i in range(2):
        alg_list.append(score_avg_sorted[i][0])
    return alg_list

#集成基于物品的推荐算法
#输入 归一化的性能矩阵，性能矩阵，测试数据，冷启动算法列表，聚类算法列表
def Item_Recommend(norm_values, result_values, data, cluster_list, alg):
    Item_perform_list = Item_calculate_cluster(data, cluster_list)
    Item_topk_result = Item_topk(Item_perform_list, cluster_list, result_values, alg, 5)
    Item_alg_list = Item_calculation(Item_topk_result, norm_values, alg)
    return Item_alg_list

#基于用户的推荐算法
#皮尔逊，不受平移影响，值越大表示相似度越高
def User_similarity_pearson(vec1, vec2):
    value = range(len(vec1))

    sum_vec1 = sum([vec1[i] for i in value])
    sum_vec2 = sum([vec2[i] for i in value])

    square_sum_vec1 = sum([pow(vec1[i], 2) for i in value])
    square_sum_vec2 = sum([pow(vec2[i], 2) for i in value])

    product = sum([vec1[i] * vec2[i] for i in value])

    numerator = product - (sum_vec1 * sum_vec2 / len(vec1))
    dominator = ((square_sum_vec1 - pow(sum_vec1, 2) / len(vec1)) * (
                square_sum_vec2 - pow(sum_vec2, 2) / len(vec2))) ** 0.5

    if dominator == 0:
        return 0
    result = numerator / (dominator * 1.0)

    return result

#计算元特征矩阵的topk文件
#输入 测试数据的元特征，元特征矩阵（包括文件名），取出topk文件的数目
#输出 topk的相似样本（文件名+皮尔逊值）
def User_topk(new_data, model_data, k):
    pearson_dict = {}
    for data in model_data:
        temp = User_similarity_pearson(data[1:], new_data[1:])
        pearson_dict[data[0]] = temp
    pearson_dict_sorted = sorted(pearson_dict.items(), key=lambda x: x[1], reverse = True)
    topk_result = pearson_dict_sorted[:k]
    return topk_result

#计算推荐的两种算法名称
#输入 topk的相似样本（文件名+皮尔逊值），归一化的性能矩阵
#输出 两种推荐算法
def User_calculation(topk, norm_values, alg):
    file_name = [temp[0] for temp in topk]

    #得到对应文件的得分
    score_list = []
    for name in file_name:
        for item in norm_values:
            if name == item[0]:
                score_list.append(item[1:])

    alg_list = []
    score_avg = {}
    for i in range(len(alg)):
        sum = 0
        for j in range(len(file_name)):
            temp = score_list[j][i]
            #其中轮廓系数为-1.0的值变为1.0进行计算，因为norm中是与max的差值，越小越好，所以这样改变
            if temp == -1.0:
                temp = 1.0
            sum += temp
        score_avg[alg[i]] = sum
    score_avg_sorted = sorted(score_avg.items(), key=lambda x: x[1])
    # top2个算法
    for i in range(2):
        alg_list.append(score_avg_sorted[i][0])

    return alg_list

#集成基于用户的推荐算法
#输入 元特征矩阵，性能矩阵，测试数据的元特征
#输出 两种推荐算法
def User_Recommend(mf_values, norm_values, data, alg):
    User_topk_list = User_topk(data, mf_values, 5)
    # print(User_topk_list)
    User_alg_list = User_calculation(User_topk_list, norm_values, alg)
    return User_alg_list

#链接MongoDB数据库
def Connectdatabase():
    conn=pymongo.MongoClient(host="localhost",port=27017)
    db=conn.MongoDB_Data
    return db
#User
def insert_User(file_name,file_path,user_name,alg,Train1Mf_values,Train1ResultNorm_values):
    db = Connectdatabase()
    new = db.MetaFeature.find_one({'file_name':file_name})
    new_data = [file_name] + new['meta_feature']

    # User_Based
    User_df = pd.DataFrame(index=[file_name], columns=alg)
    print(new_data[0])
    User_alg_list = User_Recommend(Train1Mf_values, Train1ResultNorm_values, new_data, alg)
    print(User_alg_list)
    for j in range(len(alg)):
        if alg[j] in User_alg_list:
            User_df.values[0][j] = 1
        else:
            User_df.values[0][j] = 0
    User_df.to_csv('D:\\superlloy\\automl\\cluster\\selection\\recommend_user\\' + file_name.rstrip(
            '.xls') + '_userbased.csv')

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    file_name = file_name
    user_name = user_name
    User_alg_list = User_alg_list
    type = 'User_Based'
    print('Insert DataBase:')
    db.RecommendResult.insert({
        "date": date,
        "file_name": file_name,
        "user_name": user_name,
        "alg_recommend": User_alg_list,
        "type": type
    })
    print('End Insert')
#Item
def insert_Item(file_name,file_path,user_name,alg,Train1ResultNorm_values,Train1Result_values,cluster_list):
    db = Connectdatabase()

    Item_df = pd.DataFrame(index=[file_name], columns=alg)
    global data_file
    data_file = load_data(file_path)
    Item_alg_list = Item_Recommend(Train1ResultNorm_values, Train1Result_values, data_file.data, cluster_list, alg)
    print(Item_alg_list)
    for j in range(len(alg)):
        if alg[j] in Item_alg_list:
            Item_df.values[0][j] = 1
        else:
            Item_df.values[0][j] = 0

    Item_df.to_csv('D:\\superlloy\\automl\\cluster\\selection\\recommend_item\\' + file_name.rstrip(
            '.xls') + '_itembased.csv')

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    file_name = file_name
    user_name = user_name
    Item_alg_list = Item_alg_list
    type = 'Item_Based'
    print('Insert DataBase:')
    db.RecommendResult.insert({
        "date": date,
        "file_name": file_name,
        "user_name": user_name,
        "alg_recommend": Item_alg_list,
        "type": type
    })
    print('End Insert')
# Model
def insert_Model(file_name, file_path, user_name, alg, Train1Result_values, cluster_list):
    db = Connectdatabase()

    Model_df = pd.DataFrame(index=[file_name], columns=alg)
    # global data_file
    data_file = load_data(file_path)
    Model_alg_list = Model_Recommend(Train1Result_values, data_file.data, cluster_list, alg)
    print(Model_alg_list)
    for j in range(len(alg)):
        if alg[j] in Model_alg_list:
            Model_df.values[0][j] = 1
        else:
            Model_df.values[0][j] = 0

    Model_df.to_csv(
        'D:\\superlloy\\automl\\cluster\\selection\\recommend_model\\' + file_name.rstrip(
            '.xls') + '_modelbased.csv')

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    file_name = file_name
    user_name = user_name
    Model_alg_list = Model_alg_list
    type = 'Model_Based'
    print('Insert DataBase:')
    db.RecommendResult.insert({
        "date": date,
        "file_name": file_name,
        "user_name": user_name,
        "alg_recommend": Model_alg_list,
        "type": type
    })
    print('End Insert')

if __name__ == '__main__':
    Train1Mf_path = 'D:\\superlloy\\automl\\cluster\\selection\\Train1\\Train1_MetaFeatures.csv'
    Train1Result_path = 'D:\\superlloy\\automl\\cluster\\selection\\Train1\\Train1_Result.csv'
    Train1ResultNorm_path = 'D:\\superlloy\\automl\\cluster\\selection\\Train1\\Train1_ResultNorm.csv'

    Train1Mf_df = pd.read_csv(Train1Mf_path)
    Train1Mf_values = Train1Mf_df.values
    Train1Result_df = pd.read_csv(Train1Result_path)
    Train1Result_values = Train1Result_df.values
    Train1ResultNorm_df = pd.read_csv(Train1ResultNorm_path)
    Train1ResultNorm_values = Train1ResultNorm_df.values
    alg = Train1Result_df.columns[1:]
    cluster_list = ['kmeans', 'meanshift']



    # file_name = 'auto-test.xlsx'
    # file_path = '/Users/buming/Documents/Super_Alloy/SuperAlloy_System/superalloy/data/piki/auto-test.xlsx'
    # username = 'piki'

    # insert_User(file_name, file_path, username, alg, Train1Mf_values, Train1ResultNorm_values)
    # insert_Item(file_name, file_path, username, alg, Train1ResultNorm_values, Train1Result_values, cluster_list)
    # insert_Model(file_name, file_path, username, alg, Train1Result_values, cluster_list)

    # python .py file_name0 file_path1  username2
    length_argv = len(sys.argv)
    print(length_argv)

    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)

    # data_file = load_data(parameterlist[1])
    # insert(parameterlist[0], parameterlist[1], parameterlist[2], parameterlist[3])

    insert_User(parameterlist[0], parameterlist[1], parameterlist[2], alg, Train1Mf_values, Train1ResultNorm_values)
    insert_Item(parameterlist[0], parameterlist[1], parameterlist[2], alg, Train1ResultNorm_values, Train1Result_values, cluster_list)
    insert_Model(parameterlist[0], parameterlist[1], parameterlist[2], alg, Train1Result_values, cluster_list)
