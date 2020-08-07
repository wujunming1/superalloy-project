import numpy as np
import pandas as pd
import matplotlib
import sys
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib
import sklearn.datasets as ds
from matplotlib import cm
from sklearn.cluster import DBSCAN
import pymongo
import time
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
from sklearn.cluster import estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn.mixture import GaussianMixture
from collections import Counter
from random import choice
from sklearn import metrics
from sklearn.decomposition import PCA
import multiprocessing as mp
import re
import os
from hyperopt import fmin,tpe,hp,partial
import csv
import os
import sys
import time
import pymongo
import pickle
from bson.binary import Binary
import json

from matplotlib import cm

from sklearn.metrics import silhouette_samples
from sklearn.metrics import adjusted_rand_score
# from sklearn.metrics import calinski_harabaz_score

from sklearn.neighbors import LocalOutlierFactor

'判断特征选择个数是否大于等于1'
def judge_feature(feature_population):
    for i in range(feature_population.shape[0]):
        num=0
        for j in range(feature_population.shape[1]):
            if feature_population.iat[i,j] == 1:
                num=num+1
        if num<=0:
            return False
            break
    if num>=1:
        return True


'判断算法选择是否为0'
def judge_algorithm(algorithm_population):
    for i in range(algorithm_population.shape[0]):
        num=0
        for j in range(algorithm_population.shape[1]):
            if algorithm_population.iat[i,j] == 1:
                num=num+1
        if num<=0:
            return False
            break
    if num>=1:
        return True


'初始化特征种群'
def feature_population(feature_num,individual_feature_num):
    matrix = [[0 for i in range(feature_num)] for i in range(individual_feature_num)]
    '算法策略'
    for i in range(individual_feature_num):
        for j in range(feature_num):
            matrix[i][j]=np.random.randint(0,2)
    column=[]
    for each_feature in range(feature_num):
        column.append(str(each_feature))
    feature_population=pd.DataFrame(columns=column,data=matrix)
    return feature_population

'初始化算法种群'
def algorithm_population(algorithm_num,individual_algorithm_num):
    matrix = [[0 for i in range(algorithm_num)] for i in range(individual_algorithm_num)]
    '算法策略'
    for i in range(individual_algorithm_num):
        for j in range(algorithm_num):
            matrix[i][j] = np.random.randint(0, 2)
    column = ['dbscan', 'kmeans', 'meanshift', 'agglomerative', 'gaussianmixture', 'birch', 'affinity']
    algorithm_population = pd.DataFrame(columns=column, data=matrix)
    return algorithm_population

#评估标准
def sil_score(data,labels):
    try:
        score = metrics.calinski_harabaz_score(data,labels)
    except Exception as err:
        # print(err)
        return 0
    else:
        return score

#DBSCAN
def hyper_dbscan(args):
    global basic_data
    global all_data
    db = DBSCAN(eps = args["eps"], min_samples = int(args["min_samples"]), metric = args["metric"])
    # db.fit(data_file.data)
    pred = db.fit_predict(basic_data)
    temp = sil_score(all_data,pred)
    # print(args)
    return -temp

def dbscan_best():
    metric_list = ['euclidean','manhattan','chebyshev']
    if basic_data.shape[0] < 30:
        space = {
            "eps": hp.uniform("eps", 0, 2),
            "min_samples": hp.choice("min_samples", range(2, basic_data.shape[0]-1)),
            "metric": hp.choice("metric", metric_list)
        }
    else:
        space = {
            "eps": hp.uniform("eps", 0, 2),
            "min_samples": hp.choice("min_samples", range(2, 30)),
            "metric": hp.choice("metric", metric_list)
        }


    algo = partial(tpe.suggest,n_startup_jobs = 10)
    best = fmin(hyper_dbscan, space, algo = algo, max_evals = 300)

    return best
#KMeans
#这里输出的best是与range的位置差
def hyper_kmeans(args):
    global basic_data
    global all_data
    km = KMeans(n_clusters = int(args["n_iter"]),init = 'k-means++',n_init = 10,max_iter = 300,random_state = 0)
    km.fit(basic_data)
    pred = km.predict(basic_data)
    temp = sil_score(all_data,pred)
    return -temp

def kmeans_best():
    # print(data.shape[0])
    if basic_data.shape[0] < 30:
        space = {
            "n_iter": hp.choice("n_iter", range(1, basic_data.shape[0]))
        }
    else:
        space = {
            "n_iter": hp.choice("n_iter", range(1, 30))
        }
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(hyper_kmeans, space, algo=algo, max_evals = 300)

    return best

#MeanShift
def hyper_meanshift(args):
    global basic_data
    global all_data
    ms = MeanShift(bandwidth = args['bandwidth'],min_bin_freq = int(args['min_bin_freq']))
    pred = ms.fit_predict(basic_data)
    temp = sil_score(all_data,pred)
    # print(args)
    return -temp

def meanshift_best():
    bandwidth = estimate_bandwidth(basic_data)
    # print(bandwidth)
    if(bandwidth - bandwidth/2) <0 and (bandwidth + bandwidth/2) >0:
        # print(1)
        space = {
            'bandwidth': hp.uniform('bandwidth', 0, bandwidth + bandwidth / 2),
            'min_bin_freq': hp.choice('min_bin_freq', range(1, 30))
        }
    elif (bandwidth + bandwidth/2) <=0 :
        # print(2)
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
    if basic_data.shape[0] <1000:
        best = fmin(hyper_meanshift, space, algo=algo, max_evals=300)
    else:
        best = fmin(hyper_meanshift, space, algo=algo, max_evals=300)

    return best

#Agglomerative 三种连接方式：min、max、avg
def hyper_agglomerative(args):
    global basic_data
    global all_data
    ag = AgglomerativeClustering(n_clusters = int(args['n_clusters']), affinity = args['affinity'], linkage = args['linkage'])
    pred = ag.fit_predict(basic_data)
    temp = sil_score(all_data,pred)
    # print(args)
    return -temp

def agglomerative_best():
    linkage_list = ['complete', 'average', 'single']
    affinity_list = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    if basic_data.shape[0] < 30:
        space = {
            'n_clusters': hp.choice('n_clusters', range(1, basic_data.shape[0])),
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
    best = fmin(hyper_agglomerative, space, algo = algo, max_evals =300)
    return best

#ward 使用Agglomerative中的ward方法
def hyper_gaussianmixture(args):
    global basic_data
    global all_data
    wd = GaussianMixture(n_components=int(args['n_components']))
    pred = wd.fit_predict(basic_data)
    temp = sil_score(all_data, pred)
    # print(args)
    return -temp


def gaussianmixture_best():
    if basic_data.shape[0] < 30:
        space = {
            'n_components': hp.choice('n_components', range(1,basic_data.shape[0])),
        }
    else:
        space = {
            'n_components': hp.choice('n_components', range(1, 30)),
        }
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(hyper_gaussianmixture, space, algo=algo, max_evals=300)

    return best

#Birch
def hyper_birch(args):
    global basic_data
    global all_data

    bir = Birch(threshold=args['threshold'], branching_factor=int(args['branching_factor']))
    pred = bir.fit_predict(basic_data)
    temp = sil_score(all_data, pred)
    # print(args)
    return -temp

def birch_best():
    space = {
        'threshold': hp.uniform('threshold', 0, 1),
        'branching_factor': hp.choice('branching_factor', range(25,75)),
        # 'n_clusters': hp.choice('n_clusters', range(1,20))
    }
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(hyper_birch, space, algo=algo, max_evals=300)

    return best

# Affinity propagation
def hyper_affinity(args):
    global basic_data
    global all_data
    ap = AffinityPropagation(damping = args['damping'])
    pred = ap.fit_predict(basic_data)
    temp = sil_score(all_data, pred)
    # print(args)
    return -temp

def affinity_best():
    space = {
        'damping': hp.uniform('damping',0.5, 0.99)
    }
    algo = partial(tpe.suggest, n_startup_jobs=10)
    best = fmin(hyper_affinity, space, algo=algo, max_evals=300)
    # print(best)
    # model = AffinityPropagation(damping = best['damping'])
    return best


'基本聚类算法的特征子集聚类结果'
def basic_feature_cluster(data,feature_selection):
    chooses=[]
    global basic_data
    global all_data
    all_data=data
    metric_list = ['euclidean', 'manhattan', 'chebyshev']
    linkage_list = ['complete', 'average', 'single']
    affinity_list = ['euclidean', 'l1', 'l2', 'manhattan', 'cosine']
    for i in range(len(feature_selection)):
        if feature_selection[i]==1:
            chooses.append(i)

    basic_data=data.ix[:, chooses]
    best_dbscan=dbscan_best()
    dbscan = DBSCAN(eps = best_dbscan["eps"], min_samples = int(best_dbscan["min_samples"]+2), metric = metric_list[best_dbscan["metric"]]).fit(basic_data).labels_
    best_kmeans=kmeans_best()
    kmeans = KMeans(n_clusters = int(best_kmeans["n_iter"]+1),init = 'k-means++',n_init = 10,max_iter = 300,random_state = 0).fit(basic_data).labels_
    best_meanshift=meanshift_best()
    meanshift = MeanShift(bandwidth = best_meanshift['bandwidth'], min_bin_freq = int(best_meanshift['min_bin_freq']+1)).fit(basic_data).labels_
    best_agglomerative=agglomerative_best()
    agglomerative = AgglomerativeClustering(n_clusters = int(best_agglomerative['n_clusters']+1), affinity = affinity_list[best_agglomerative['affinity']], linkage = linkage_list[best_agglomerative['linkage']]).fit(basic_data).labels_
    best_gaussianmixture=gaussianmixture_best()
    gaussianmixture = GaussianMixture(n_components=int(best_gaussianmixture['n_components']+1)).fit(basic_data).predict(basic_data)
    best_birch=birch_best()
    birch = Birch(threshold=best_birch['threshold'], branching_factor=int(best_birch['branching_factor'] + 25)).fit(basic_data).labels_
    best_affinity=affinity_best()
    affinity = AffinityPropagation(damping = best_affinity['damping']).fit(basic_data).labels_
    column = ['dbscan', 'kmeans', 'meanshift', 'agglomerative', 'gaussianmixture', 'birch', 'affinity']
    label=np.array([dbscan, kmeans, meanshift, agglomerative, gaussianmixture, birch,affinity])
    basic_labels=pd.DataFrame(columns=column,data=label.T)
    return basic_labels

'每个算法的各个评价指标'
def all_evaluate(data,consensus_basic_cluster):
    dbscan=evaluate(data,consensus_basic_cluster['dbscan'].values)
    kmeans = evaluate(data, consensus_basic_cluster['kmeans'].values)
    meanshift = evaluate(data, consensus_basic_cluster['meanshift'].values)
    agglomerative = evaluate(data, consensus_basic_cluster['agglomerative'].values)
    gaussianmixture = evaluate(data, consensus_basic_cluster['gaussianmixture'].values)
    birch = evaluate(data, consensus_basic_cluster['birch'].values)
    affinity = evaluate(data, consensus_basic_cluster['affinity'].values)
    score=[dbscan,kmeans,meanshift,agglomerative,gaussianmixture,birch,affinity]
    return score



'各个评价指标评分'
def evaluate(data,label):

    try:
        si = metrics.silhouette_score(data, label)
        ch = metrics.calinski_harabaz_score(data, label)
        dbi=metrics.davies_bouldin_score(data,label)
    except Exception as err:
            # print(err)
         score = [-1, 0,100]
         return score
    else:
        score = [si, ch,dbi]
        return score






'染色体所代表的算法'
def one_evaluate(all_score,chromosome):
    matrix=[]
    column=['si','ch','dbi']
    index=[]
    if chromosome['dbscan']==1:
        matrix.append(all_score[0])
        index.append(0)
    if chromosome['kmeans']==1:
        matrix.append(all_score[1])
        index.append(1)
    if chromosome['meanshift']==1:
        matrix.append(all_score[2])
        index.append(2)
    if chromosome['agglomerative']==1:
        matrix.append(all_score[3])
        index.append(3)
    if chromosome['gaussianmixture']==1:
        matrix.append(all_score[4])
        index.append(4)
    if chromosome['birch']==1:
        matrix.append(all_score[5])
        index.append(5)
    if chromosome['affinity']==1:
        matrix.append(all_score[6])
        index.append(6)
    score=pd.DataFrame(columns=column,data=matrix,index=index)
    return score



def find_max(data,j):
    max=data.iat[0,j]
    for i in range(data.shape[0]):
        if data.iat[i,j]>=max:
            max=data.iat[i,j]

    return max

def find_min(data,j):
    min=data.iat[0,j]
    for i in range(data.shape[0]):
        if data.iat[i,j]<=min:
            min=data.iat[i,j]

    return min


def rescore(score):
    new_score=score
    for j in range(new_score.shape[1]):
        max=find_max(new_score,j)
        min=find_min(new_score,j)
        field=max-min
        if field==0:
            for i in range(new_score.shape[0]):
                rescore_item=1
                new_score.iat[i, j] = rescore_item

        else:
            for i in range(new_score.shape[0]):
                rescore_item = (new_score.iat[i, j] - min) / field
                # new_score.iat[i, j] = rescore_item
                if j==2:
                    new_score.iat[i, j] = 1-rescore_item
                else:
                    new_score.iat[i, j] = rescore_item
    return new_score



def length(a):
    length = 0
    for i in range(len(a)):
        length = length + len(a[i])
    return length


def notfind(x,a):
    flag=0
    for i in a:
        for each in i:
            if each==x:
                flag=1

    if flag==0:
        return True
    else:
        return False




def find(x,a):
    for i in range(len(a)):
        if x in a[i]:
            return i
            break

'相似性矩阵共识函数'
def con_cluster(all_score,labels,chromosome):
    choose=[]

    one = rescore(one_evaluate(all_score,chromosome))
    SM=[[0 for i in range(labels.shape[0])] for i in range(labels.shape[0])]
    for i in range(len(chromosome)):
        if chromosome[i]==1:
            choose.append(i)
    m=0
    for i in choose:
        m=m+(one.at[i,'si']+one.at[i,'ch']+one.at[i,'dbi'])/3
    for i in range(labels.shape[0]):
        for j in range(labels.shape[0]):
            c=0
            if i==j:
                SM[i][j]=1
            else:

                for k in choose:
                    label=labels.ix[:,k].values
                    x=label[i]
                    y=label[j]
                    if ((x==y) and (x!=-1) and (y!=-1)):
                        c=c+(one.at[k,'si']+one.at[k,'ch']+one.at[k,'dbi'])/3
                SM[i][j] = float(c/m)
    column=[]
    index=[]
    for i in range(labels.shape[0]):
        column.append(str(i))
        index.append(str(i))

    sm = pd.DataFrame(columns=column, data=SM, index=index)
    a=[]
    n_cluster=0
    while(length(a)!=labels.shape[0]):
        for i in range(labels.shape[0]):
            if notfind(i,a):
                a.append([i])
                n_cluster=n_cluster+1
                for j in range(labels.shape[0]):
                    if SM[i][j]>=0.5and i!=j and notfind(j,a):
                        a[n_cluster-1].append(j)


    cluster=[]
    for i in range(labels.shape[0]):
        cluster.append(find(i,a))
    return cluster



'评估算法集成策略好坏'
def algorithm_score(all_algorithm_population,data,labels,algorithm_population):
    algorithm_population['score'] = None
    algorithm_population['label']=None
    algorithm_population['allscore'] = None
    algorithm_num=7
    flag=0
    print(algorithm_population)
    print(all_algorithm_population)
    all_score = all_evaluate(data, labels)
    for i in range(algorithm_population.shape[0]):
        if all_algorithm_population.shape[0]==0:
            print("记录为空")
            label = con_cluster(all_score, labels, algorithm_population.loc[i])
            print(label)
            # label_1 = list(set(label))
            score=evaluate(data,label)

            algorithm_population.at[i, 'allscore'] = score
            algorithm_population.at[i, 'label'] = label
            all_algorithm_population= all_algorithm_population.append(algorithm_population.loc[i, :], ignore_index=True)
            print(algorithm_population)
            print('第' + str(i + 1) + '条算法染色体')
            print('allscore', score)
        else:
            print('记录不为空')

            for index in range(all_algorithm_population.shape[0]):
                if judge_same(list(all_algorithm_population.iloc[index, :algorithm_num].values),list(algorithm_population.iloc[i, :algorithm_num].values)):
                    print('找到了')
                    algorithm_population.at[i, 'allscore'] = all_algorithm_population.at[index, 'allscore']
                    algorithm_population.at[i, 'label'] = all_algorithm_population.at[index, 'label']
                    flag = 1
                    print(algorithm_population)
                    print('第' + str(i + 1) + '条算法染色体')
                    print('allscore', all_algorithm_population.at[index, 'allscore'])
                    break
            if flag==0:
                print("没找到")
                label = con_cluster(all_score, labels, algorithm_population.loc[i])
                score = evaluate(data, label)

                algorithm_population.at[i, 'allscore'] = score
                algorithm_population.at[i, 'label'] = label
                all_algorithm_population = all_algorithm_population.append(algorithm_population.loc[i, :],
                                                                           ignore_index=True)
                print(algorithm_population)
                print('第' + str(i + 1) + '条算法染色体')
                print('allscore', score)
            flag=0

    return algorithm_population,all_algorithm_population


def select_algorithm_positive(algorithm_population):
    select=[]
    mid=0
    score=algorithm_population['score']
    sort_score=sorted(score)

    for index in range(algorithm_population.shape[0]):
        if algorithm_population['score'][index]>=sort_score[int(algorithm_population.shape[0]/2)]:
            select.append(index)
    return algorithm_population.loc[select]

#个体的适应度，分数越高，适应度越高
def algorithm_fitness(algorithm_population):

    score=algorithm_population['score']
    min=score.sort_values().values[0]
    max= score.sort_values().values[int(algorithm_population.shape[0])-1]
    field=max-min
    print(max)
    print(min)
    score=score.values
    if field==0:
        for i in range(algorithm_population.shape[0]):
            rescore_item=1
            score[i] = rescore_item

    else:
        for i in range(algorithm_population.shape[0]):
            rescore_item = float((score[i] - min) / field)
            score[i] = rescore_item


    sum=np.sum(score)
    algorithm_fitness=[]
    for each in score:
        algorithm_fitness.append(each/sum)
    return algorithm_fitness

#累计概率
def cumulative_probability(fitness):
    probability=[]
    cumulative=0
    for each in fitness:
        cumulative=cumulative+each
        probability.append(cumulative)
    return probability

#在配种池中选择两个个体
def choose_two(random,population):
    choose_two=[]
    for index in range(len(random)):
        choose_two.append(judge_bin(random[index],population['cp']))
    return choose_two



#判断随机数代表的个体
def judge_bin(random,cp):
    cp_index=[]
    for index in cp.index:
        cp_index.append(index)
    for index in range(len(cp_index)):
        if index==0:
            if 0<=random<=cp[cp_index[index]]:
                return cp_index[index]
        else:
            if cp[cp_index[index-1]]<=random<=cp[cp_index[index]]:
                return cp_index[index]
#交叉
def crossover(population,random):
    position=np.random.randint(1,population.shape[1])
    a=population.loc[random[0],:]
    b=population.loc[random[1],:]
    a_ncross=[]
    b_ncross=[]
    a_cross=[]
    b_cross=[]
    a_new=[]
    b_new=[]
    for index in range(a.size):
        if index>=position:
            a_cross.append(a.values[index])
        else:
            a_ncross.append(a.values[index])
    for index in range(b.size):
        if index>=position:
            b_cross.append(b.values[index])
        else:
            b_ncross.append(b.values[index])
    a_new=a_ncross+b_cross
    b_new=b_ncross+a_cross
    for index in range(a.size):
        a.values[index]=a_new[index]
    for index in range(b.size):
        b.values[index] = b_new[index]
    c=pd.DataFrame([a,b])
    return c


#复制
def reproduction(population,random):
    a = population.loc[random[0], :]
    b = population.loc[random[1], :]
    c = pd.DataFrame([a, b])
    return c


#变异
def mutation(population):
    random1=np.random.randint(0,population.shape[0])
    random2=np.random.randint(0,population.shape[1])
    if population.iloc[random1,random2]==1:
        population.iloc[random1, random2]=0
    else:
        population.iloc[random1, random2]=1
    return population

def population_rescore(algorithm_population):
    score=algorithm_population['allscore']
    score=pd.array(score)
    score=pd.DataFrame(data=score)
    score=rescore(score)
    for i in range(algorithm_population.shape[0]):
        n=float((score.ix[i,0]+score.ix[i,1]+score.ix[i,2])/3)
        algorithm_population.at[i,'score']=n

    print(score)
    print(algorithm_population)
    return algorithm_population
'下一代算法种群'
def next_algorithm_population( all_algorithm_population,data,individual_algorithm_num,cluster_basic_feature,algorithm_population,algorithm_cross_probability,algorithm_mutation_probability):
    '得出算法分数和综合标签'
    algorithm_population,all_algorithm_population=algorithm_score(all_algorithm_population,data,cluster_basic_feature,algorithm_population)
    print('dsad sa ',all_algorithm_population)
    print(algorithm_population)
    algorithm_population=population_rescore(algorithm_population)
    positive_algorithm_population=select_algorithm_positive(algorithm_population)
    positive_algorithm_population['fitness'] = algorithm_fitness(positive_algorithm_population)
    positive_algorithm_population['cp'] = cumulative_probability(positive_algorithm_population['fitness'])
    print(positive_algorithm_population)
    new_algorithm_population=pd.DataFrame()
    for i in range(int(individual_algorithm_num/2)):
        random = np.random.random(2)
        two_choice = choose_two(random, positive_algorithm_population)
        print(two_choice)
        random_cross=np.random.random()
        column = ['dbscan', 'kmeans', 'meanshift', 'agglomerative', 'gaussianmixture', 'birch', 'affinity']
        if random_cross<=algorithm_cross_probability:
            new_two_algorithm_population=crossover(positive_algorithm_population.loc[:, column], two_choice)
            algorithm_judge =judge_algorithm( new_two_algorithm_population)
            while (algorithm_judge == False):
                new_two_algorithm_population = crossover(positive_algorithm_population.loc[:, column], two_choice)
                algorithm_judge = judge_algorithm( new_two_algorithm_population)
        else:
            new_two_algorithm_population=reproduction(positive_algorithm_population.loc[:, column], two_choice)
        random_mutation = np.random.random()
        if random_mutation <= algorithm_mutation_probability:
            mutation(new_two_algorithm_population)
            algorithm_judge = judge_algorithm(new_two_algorithm_population)
            while (algorithm_judge == False):
                mutation(new_two_algorithm_population)
                algorithm_judge = judge_algorithm(new_two_algorithm_population)
        new_algorithm_population= new_algorithm_population.append(new_two_algorithm_population,ignore_index=True)
    print(new_algorithm_population)
    return new_algorithm_population,all_algorithm_population


def final_algorithm_population(all_algorithm_population,data, cluster_basic_feature,new_algorithm_population):
    '得出算法分数和综合标签'
    algorithm_population ,all_algorithm_population= algorithm_score(all_algorithm_population,data, cluster_basic_feature,new_algorithm_population)
    max_score=0
    max_allscore=[]
    algorithm_population = population_rescore(algorithm_population)
    for index in range(algorithm_population.shape[0]):
        if algorithm_population['score'][index]>=max_score:
            max_score=algorithm_population['score'][index]
            max_allscore=algorithm_population['allscore'][index]
            max=algorithm_population.loc[index,:]
    return max_allscore,max
def select_feature_positive(feature_population):
    select=[]
    mid=0
    score=feature_population['score']
    print(score)
    sort_score=sorted(score)
    print(sort_score)

    for index in range(feature_population.shape[0]):
        if feature_population['score'][index]>=sort_score[int(feature_population.shape[0]/2)]:
            select.append(index)

    return feature_population.iloc[select,:]



#个体的适应度，分数越高，适应度越高
def feature_fitness(feature_population):
    sum=np.sum(feature_population['score'])
    feature_fitness=[]
    for each in feature_population['score']:
        feature_fitness.append(each/sum)
    return feature_fitness

'选择最佳算法'
def best_algorithm(i,data,feature_population,individual_algorithm_num,algorithm_evolution,algorithm_cross_probability,algorithm_mutation_probability):
    '初始化算法种群'
    algorithm_num=7
    column = ['dbscan', 'kmeans', 'meanshift', 'agglomerative', 'gaussianmixture', 'birch', 'affinity']
    column.append('score')
    column.append('label')
    column.append('allscore')
    all_algorithm_population = pd.DataFrame(columns=column)
    new_algorithm_population = algorithm_population(algorithm_num, individual_algorithm_num)
    '判断是否有空的算法子集，若有，则重新生成'
    algorithm_judge = judge_algorithm(new_algorithm_population)
    while (algorithm_judge == False):
        new_algorithm_population = algorithm_population(algorithm_num, individual_algorithm_num)
        algorithm_judge = judge_algorithm(new_algorithm_population)
    print(new_algorithm_population)
    cluster_basic_feature = basic_feature_cluster(data, feature_population.iloc[i].values)
    for j in range(algorithm_evolution):
        print('第' + str(j+1) + '次算法进化')
        if j+1 <algorithm_evolution:
            new_algorithm_population,all_algorithm_population= next_algorithm_population(all_algorithm_population,data, individual_algorithm_num, cluster_basic_feature,new_algorithm_population, algorithm_cross_probability,algorithm_mutation_probability)
        else:
            max_allscore,best_algorithm=final_algorithm_population(all_algorithm_population,data, cluster_basic_feature,new_algorithm_population)

    return max_allscore,best_algorithm

'下一代特征选择'
def final_feature_population(data,all_feature_population,feature_population,individual_algorithm_num,algorithm_evolution,algorithm_cross_probability,algorithm_mutation_probability):
    new_feature_score = []
    algorithm_num=7
    new_feature_score = []
    max_score=0
    feature_population['score'] = None
    feature_population['algorithm'] = None
    feature_population['allscore'] = None
    print(feature_population)

    feature_num = int(data.shape[1])
    flag=0
    for i in range(feature_population.shape[0]):
        '算法选择，返回每个特征染色体的最佳算法分值'
        '算法选择，返回每个特征染色体的最佳算法分值'
        print('第' + str(i + 1) + '条特征染色体')
        if all_feature_population.shape[0] == 0:
            print('记录为空')

            score, algorithm = best_algorithm(i, data, feature_population, individual_algorithm_num,
                                              algorithm_evolution, algorithm_cross_probability,
                                              algorithm_mutation_probability)
            feature_population.at[i, 'allscore'] = score
            feature_population.at[i, 'algorithm'] = algorithm
            all_feature_population = all_feature_population.append(feature_population.loc[i, :], ignore_index=True)
            print(feature_population)
            print(feature_population.at[i, 'algorithm'])
            print(all_feature_population)
        else:
            print('记录不为空')

            for index in range(all_feature_population.shape[0]):

                if judge_same(list(all_feature_population.iloc[index, :feature_num].values),
                              list(feature_population.iloc[i, :feature_num].values)):
                    print('找到了')
                    feature_population.at[i, 'allscore'] = all_feature_population.at[index, 'allscore']
                    feature_population.at[i, 'algorithm'] = all_feature_population.at[index, 'algorithm']
                    flag = 1
                    break

            if flag == 0:

                score, algorithm = best_algorithm(i, data, feature_population, individual_algorithm_num,
                                                  algorithm_evolution, algorithm_cross_probability,
                                                  algorithm_mutation_probability)
                feature_population.at[i, 'allscore'] = score
                feature_population.at[i, 'algorithm'] = algorithm
                all_feature_population = all_feature_population.append(feature_population.loc[i, :], ignore_index=True)
            flag = 0
            print(feature_population)
            print(feature_population.at[i, 'algorithm'])
            print(all_feature_population)

    print(feature_population)
    feature_population = population_rescore(feature_population)
    for index in range(feature_population.shape[0]):
        if feature_population['score'][index]>=max_score:
            max_score=feature_population['score'][index]
            max_allscore=feature_population['allscore'][index]
            max=feature_population.loc[index,:]
    return max_allscore,max

def judge_same(array1,array2):
    if  array1==array2:
        return True
    else:
        return False

def best(i,all_feature_population,feature_num,data,feature_population,individual_algorithm_num,algorithm_evolution,algorithm_cross_probability,algorithm_mutation_probability):
    flag = 0
    if all_feature_population.shape[0] == 0:
        print('记录为空')

        score, algorithm = best_algorithm(i, data, feature_population, individual_algorithm_num, algorithm_evolution,
                                          algorithm_cross_probability, algorithm_mutation_probability)
        feature_population.at[i, 'allscore'] = score
        feature_population.at[i, 'algorithm'] = algorithm
        all_feature_population = all_feature_population.append(feature_population.loc[i, :], ignore_index=True)
        print(feature_population)
        print(feature_population.at[i, 'algorithm'])
        print(all_feature_population)
    else:
        print('记录不为空')

        for index in range(all_feature_population.shape[0]):

            if judge_same(list(all_feature_population.iloc[index, :feature_num].values),
                          list(feature_population.iloc[i, :feature_num].values)):
                print('找到了')
                feature_population.at[i, 'allscore'] = all_feature_population.at[index, 'allscore']
                feature_population.at[i, 'algorithm'] = all_feature_population.at[index, 'algorithm']
                flag = 1
                break

        if flag == 0:
            score, algorithm = best_algorithm(i, data, feature_population, individual_algorithm_num,
                                              algorithm_evolution, algorithm_cross_probability,
                                              algorithm_mutation_probability)
            feature_population.at[i, 'allscore'] = score
            feature_population.at[i, 'algorithm'] = algorithm
            # result = pool.apply_async(best_algorithm, (
            #     i, data, feature_population, individual_algorithm_num, algorithm_evolution,
            #     algorithm_cross_probability,
            #     algorithm_mutation_probability,))
            # feature_population.at[i, 'score'] = result.get()[0]
            # feature_population.at[i, 'algorithm'] = result.get()[1]
            all_feature_population = all_feature_population.append(feature_population.loc[i, :], ignore_index=True)
        flag = 0
        print(feature_population)
        print(feature_population.at[i, 'algorithm'])
        print(all_feature_population)

    return feature_population,all_feature_population
'下一代特征选择'
def next_feature_population(data,all_feature_population,feature_population,individual_algorithm_num,individual_feature_num,algorithm_evolution,algorithm_cross_probability,feature_cross_probability,algorithm_mutation_probability,feature_mutation_probability):
    new_feature_score = []
    algorithm_num=7
    flag=0
    feature_population['score']=None
    feature_population['algorithm'] = None
    feature_population['allscore'] = None
    pool = mp.Pool(processes=1)
    feature_num = int(data.shape[1])
    for i in range(feature_population.shape[0]):
        '算法选择，返回每个特征染色体的最佳算法分值'
        print('第' + str(i+1) + '条特征染色体')
        result = pool.apply_async(best, (i,all_feature_population,feature_num, data, feature_population, individual_algorithm_num, algorithm_evolution, algorithm_cross_probability,algorithm_mutation_probability,))
        feature_population=result.get()[0]
        all_feature_population=result.get()[1]
    pool.close()
    pool.join()
    print('jaja',feature_population)
    feature_population=population_rescore(feature_population)
    positive_feature_population = select_feature_positive(feature_population)
    positive_feature_population['fitness'] = feature_fitness(positive_feature_population)
    positive_feature_population['cp'] = cumulative_probability(positive_feature_population['fitness'])
    print(positive_feature_population)
    new_feature_population = pd.DataFrame()
    for i in range(int(individual_feature_num / 2)):
        random = np.random.random(2)
        two_choice = choose_two(random, positive_feature_population)
        print(two_choice)
        random_cross = np.random.random()
        column = []
        for each_feature in range(data.shape[1]):
            column.append(str(each_feature))
        if random_cross <= feature_cross_probability:
            new_two_feature_population = crossover(positive_feature_population.loc[:, column], two_choice)
            feature_judge = judge_feature(new_two_feature_population)
            while (feature_judge == False):
                new_two_feature_population = crossover(positive_feature_population.loc[:, column], two_choice)
                feature_judge = judge_feature(new_two_feature_population)
        else:
            new_two_feature_population = reproduction(positive_feature_population.loc[:, column], two_choice)
        random_mutation = np.random.random()
        if random_mutation <= feature_mutation_probability:
            mutation(new_two_feature_population)
            feature_judge = judge_feature(new_two_feature_population)
            while (feature_judge == False):
                mutation(new_two_feature_population)
                feature_judge = judge_feature(new_two_feature_population)
        new_feature_population = new_feature_population.append(new_two_feature_population, ignore_index=True)
    print(new_feature_population)
    return new_feature_population,all_feature_population


def Genetic_Algorithm(data,individual_algorithm_num,individual_feature_num,algorithm_evolution,feature_evolution,algorithm_cross_probability,feature_cross_probability,algorithm_mutation_probability,feature_mutation_probability):
    '特征个数'
    feature_num = data.shape[1]

    column = []
    for each_feature in range(feature_num):
        column.append(str(each_feature))
    column.append('score')
    column.append('algorithm')
    all_feature_population = pd.DataFrame(columns=column)
    '初始化特征种群'
    new_feature_population = feature_population(feature_num, individual_feature_num)
    '判断是否会有空的数据子集，若有，重新生成'
    feature_judge = judge_feature(new_feature_population)
    while (feature_judge == False):
        new_feature_population = feature_population(feature_num, individual_feature_num)
        feature_judge = judge_feature(new_feature_population)
    print(new_feature_population)
    for i in range(feature_evolution):
        print('第' + str(i+1) + '次特征进化')
        if i+1<feature_evolution:
            new_feature_population,all_feature_population=next_feature_population(data,all_feature_population,new_feature_population,individual_algorithm_num,individual_feature_num,algorithm_evolution,algorithm_cross_probability,feature_cross_probability,algorithm_mutation_probability,feature_mutation_probability)
        else:
            max_score,max=final_feature_population(data,all_feature_population,new_feature_population,individual_algorithm_num,algorithm_evolution,algorithm_cross_probability,algorithm_mutation_probability)
    return max_score,max

#链接MongoDB数据库
def Connectdatabase():
    conn=pymongo.MongoClient(host="localhost",port=27017)
    db=conn.MongoDB_Data
    return db

def visualization(data):
    print('Starting PCA')
    pca=PCA(n_components=2)
    # pca.fit(data)
    X=pca.fit_transform(data)
    print(X)
    print('End PCA')
    return X

def insert(username,filename,data,score,algorithm,feature_select,predict_label,real_label):

    features=data.shape[1]
    sample=data.shape[0]
    feature_names=data.columns.values
    data = np.array(data)
    xy = visualization(data)
    db = Connectdatabase()
    db.GA_cluster.insert({
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "type": "Cluster",
        "username": username,
        "data_name": filename,
        "features_count": features,
        "features_names": feature_names.tolist(),
        "data": data.tolist(),
        "sample_count": sample,
        "score": score,
        "algorithm": algorithm.astype(np.int).tolist(),
        "feature": feature_select.astype(np.int).tolist(),
        "predict_label": predict_label.astype(np.int).tolist(),
        "real_label": real_label.astype(np.int).tolist(),
        "xy": xy.tolist()


    })
def start_insert(username,filename,status):


    db = Connectdatabase()
    db.cluster_status.insert({
        "date": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "type": "Cluster",
        "username": username,
        "data_name": filename,
        "status":status

    })
def status_update(username,filename,status):
    db = Connectdatabase()
    db.cluster_status.update({
        "type": "Cluster",
        "username": username,
        "data_name": filename,
       },{"$set":{"status":status}})
def status_delete(username,filename):
    db = Connectdatabase()
    db.cluster_status.delete_many({
        "type": "Cluster",
        "username": username,
        "data_name": filename})
def cluster_delete(username,filename):
    db = Connectdatabase()
    db.GA_cluster.delete_many({
        "type": "Cluster",
        "username": username,
        "data_name": filename,
        })
if __name__=='__main__':
    '获取数据集'
    length_argv = len(sys.argv)
    print(length_argv)
    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)
    fname = os.path.basename(parameterlist[1])
    '新上传'
    if(parameterlist[2]=="1"):
        print("上传")
        start_insert(parameterlist[0], fname, 1)
    '删除新上传'
    if (parameterlist[2] == "4"):
        status_delete(parameterlist[0], fname)
    '删除已完成'
    if (parameterlist[2] == "3"):
        cluster_delete(parameterlist[0], fname)
    '点击运行'
    if(parameterlist[2]=="2"):
        print("运行")
        status_update(parameterlist[0], fname, 2)
        old_data = pd.read_excel(parameterlist[1])
        data = old_data.drop(old_data.columns[len(old_data.columns) - 1], axis=1, inplace=False)
        real_label = old_data[old_data.columns[-1]]
        '设置输入参数'
        '算法种群中染色体条数'
        individual_algorithm_num = 2
        '特征种群中染色体条数'
        individual_feature_num = 2
        '进化代数'
        algorithm_evolution = 1
        feature_evolution = 1
        '交叉率'
        algorithm_cross_probability = 0.1
        feature_cross_probability = 0.1
        '变异率'
        algorithm_mutation_probability = 0.1
        feature_mutation_probability = 0.1
        '算法种类'
        algorithm_num = 7
        max_score, max = Genetic_Algorithm(data, individual_algorithm_num, individual_feature_num, algorithm_evolution,
                                           feature_evolution, algorithm_cross_probability, feature_cross_probability,
                                           algorithm_mutation_probability, feature_mutation_probability)
        print(max['algorithm'])
        label = max['algorithm']['label']
        algorithm = max['algorithm'].drop(["score", "label"]).values
        max = max.drop(["algorithm", "score"]).values
        label = np.array(label)
        status_update(parameterlist[0], fname, 3)
        insert(parameterlist[0], fname, data, max_score, algorithm, max, label, real_label)

