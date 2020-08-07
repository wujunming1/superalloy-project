# coding: utf-8

# In[18]:

# import os
import time
import csv
import sys
# import pymongo
import numpy as np
import pandas as pd

from minepy import MINE
from sklearn import preprocessing
from sklearn.linear_model import LassoCV, Lasso, RidgeCV
from sklearn.svm import SVR
import warnings
warnings.filterwarnings("ignore")
from sklearn.cross_validation import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn import neighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
import json
import pymongo

# In[35]:
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


# In[182]:
def Connectdatabase():
    conn = pymongo.MongoClient(host="localhost", port=27017)
    db = conn.MongoDB_Data
    return db


def load_data(filename):
    with open(filename) as f:
        data_file = csv.reader(f)
        data = []
        target = []
        temp = next(data_file)
        feature_names = np.array(temp)
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
        target = target.reshape(-1, 1)
        n_samples = data.shape[0]
        n_features = data.shape[1]
    return Bunch(sample=n_samples, features=n_features, data=data, target=target,
                 feature_names=feature_names)
# In[183]:

def ImportData(fileName):
    sampleData = load_data(fileName)
    return sampleData


# In[36]:

def Normalize(sampleData):
    minMaxScaler = preprocessing.MinMaxScaler()
    X = minMaxScaler.fit_transform(sampleData.data)
    Y = minMaxScaler.fit_transform(sampleData.target)
    return Bunch(X=X, Y=Y)


# In[37]:

def ValueCounts(inputX, threshold, sample, name):
    counts = []
    arrayX = np.array(inputX)
    for i in xrange(len(inputX[0])):
        counts.append(pd.value_counts(arrayX[:, i]).values[0] * 100.0 / sample)
    indices = np.argsort(counts)[::-1]
    filtered = filter(lambda x: counts[x] > threshold, indices)
    indices = filter(lambda x: counts[x] <= threshold, indices)
    counts = np.array(counts)
    print "稀疏系数："
    print "过滤特征：", dict(zip(name[filtered], counts[filtered]))
    print "余下特征：", dict(zip(name[indices], counts[indices]))
    return Bunch(indices=indices, filtered=filtered, counts=counts)


# In[38]:

def SparsityEvaluate(inputX, threshold):
    vt = VarianceThreshold()
    vt.fit_transform(inputX)
    importance = vt.variances_
    indices = np.argsort(importance)[::-1]
    filtered = filter(lambda x: importance[x] <= threshold, indices)
    indices = filter(lambda x: importance[x] > threshold, indices)
    importance = np.array(importance)
    return Bunch(indices=indices, filtered=filtered), importance, filtered, indices


# In[39]:

def SvrPrediction(trainX, trainY, testX):
    rbfSVR = GridSearchCV(SVR(kernel='rbf'), cv=5,
                          param_grid={"C": np.logspace(-3, 3, 7),
                                      "gamma": np.logspace(-2, 2, 5)})
    rbfSVR.fit(trainX, trainY)
    predictY = rbfSVR.predict(testX)
    return predictY


# In[40]:

def modelEvaluation(testY, predictY):
    rmse = np.sqrt(mean_squared_error(testY, predictY))
    mae = mean_absolute_error(testY, predictY)
    r2 = r2_score(testY, predictY)
    return Bunch(rmse=rmse, mae=mae, r2=r2)

# In[41]:
def LayerFourSelection(fourInputData,resultIndex,username):
    pcaRmse = resultIndex.rmse
    optResultIndex = resultIndex
    nFeatures = len(fourInputData.dataX[0])
    n_components = [] * nFeatures
    new_X = []
    if nFeatures > 1:
        for i in range(1, nFeatures):
            n_components.append(i)
        Cs = np.logspace(-4, 4, 3)
        Cr = np.logspace(-3, 3, 7)
        g = np.logspace(-2, 2, 5)
        trainX, trainY = fourInputData.dataX[fourInputData.trainIndex], fourInputData.dataY[fourInputData.trainIndex]
        testX, testY = fourInputData.dataX[fourInputData.testIndex], fourInputData.dataY[fourInputData.testIndex]
        svr = SVR(kernel='rbf')
        pca = decomposition.PCA()
        pipe = Pipeline(steps=[('pca', pca), ('svr', svr)])
        estimator = GridSearchCV(pipe,
                                 dict(pca__n_components=n_components,
                                      svr__C=Cs, svr__gamma=g))
        estimator.fit(trainX, trainY)
        pcaPredictY = estimator.predict(testX)
        pcaResultIndex = modelEvaluation(testY, pcaPredictY)
        if pcaResultIndex.rmse <= pcaRmse:
            db=Connectdatabase()
            print("sssssddddd")
            pcaRmse = pcaResultIndex.rmse
            optResultIndex = pcaResultIndex
            pca_fit = decomposition.PCA(n_components=estimator.best_params_['pca__n_components'])
            new_X = pca_fit.fit_transform(fourInputData.dataX)
            print pca_fit.get_params()
            print pca_fit.explained_variance_ratio_
            print pca_fit.explained_variance_#主成分分析的方差
            print(pca_fit.get_covariance())
            n_components = pca_fit.get_params()["n_components"]
            explained_variance = pca_fit.explained_variance_
            explained_variance_ratio = pca_fit.explained_variance_ratio_
            '''
            n_components:主成分的个数
            explained_variance：主成分的方差
            explained_variance_ratio：主成分的方差贡献度

            '''
            db.outputparameter.insert({"type":"fourlayeroutput",
                                       "username":username,
                                       "n_components": n_components,
                                       "explained_variance": list(explained_variance),
                                       "explained_variance_ratio": list(explained_variance_ratio)})

        else:
            db = Connectdatabase()
            pca_fit = decomposition.PCA(n_components=2)
            new_X = pca_fit.fit_transform(fourInputData.dataX)
            explained_variance = pca_fit.explained_variance_
            explained_variance_ratio = pca_fit.explained_variance_ratio_
            db.outputparameter.insert({"type":"fourlayeroutput",
                                       "username":username,
                                       "n_components": 2,
                                       "explained_variance": list(explained_variance),
                                       "explained_variance_ratio": list(explained_variance_ratio)})

    else:
        print "cant pca2 ...."
        new_X = fourInputData.dataX
    return new_X
# In[42]:

def feature_selection(n_samples, n_features, data, target, feature_names, username, expert_remain3):
    db = Connectdatabase()
    #     remain=map(int,remain.split(","))
    #     first_remain=remain
    if len(expert_remain3)==0 or expert_remain3==None or expert_remain3=="null":
        expert_remain3=[]
    else:
        expert_remain3=map(int,expert_remain3.split(","))
    print "origin feature num", n_features
    no_target = False
    if target == []:
        no_target = True
    inputData = Bunch(sample=n_samples, features=n_features, data=data, target=target, feature_names=feature_names)
    normalizeData = Normalize(inputData)
    oneInputDatadict = db.outputparameter.find()
    for i in oneInputDatadict:
        if "result3" in i.keys():
            result=i["result3"]
            # print(result)
    kf = KFold(inputData.sample, n_folds=3)
    foldNum = 1
    totalRmse = 0
    for train_index, test_index in kf:
        indices=list(result)+list(expert_remain3)
        print(indices)
        kRmse = 0
        orgTrainX, orgTestX = normalizeData.X[train_index], normalizeData.X[test_index]
        orgTrainY, orgTestY = normalizeData.Y[train_index], normalizeData.Y[test_index]
        orgPredictY = SvrPrediction(orgTrainX, orgTrainY, orgTestX)
        orgResultIndex = modelEvaluation(orgTestY, orgPredictY)
        kRmse = orgResultIndex.rmse
        oneInputData = Bunch(dataX=normalizeData.X[:,indices], dataY=normalizeData.Y, trainIndex=train_index,
                             testIndex=test_index, sample=inputData.sample)
        if no_target == False:
            oneOutputData = LayerFourSelection(oneInputData,orgResultIndex,username)
            print oneOutputData
            #             db.outputparameter.insert({"type":"onelayeroutput","username":username,"result1":list(onereainindices),"first_remain": list(first_remain),
            #                                      "show1":show,"allshow1":allshow,"Variance": list(oneOutputData.coefficient1[allshow]),
            #                                        "Corcoefficient": list(oneOutputData.coefficient2[allshow])})
            return
# In[43]:

if __name__ == '__main__':
    # inputData = ImportData('glass.csv')
    # feature_selection(inputData.sample, inputData.features, inputData.data, inputData.target, inputData.feature_names,
    #                   "wujunming")
    parameterlist=[]
    for i in range(1, len(sys.argv)):
        para=sys.argv[i]
        parameterlist.append(para)
    print parameterlist
    inputData = ImportData(parameterlist[0])
    print parameterlist[1]
    feature_selection(inputData.sample, inputData.features, inputData.data, inputData.target, inputData.feature_names,
                      parameterlist[1],parameterlist[2])


# In[ ]:



