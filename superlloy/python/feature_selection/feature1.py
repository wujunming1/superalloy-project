
# coding: utf-8

# In[34]:
import os
import time
import csv
import sys
import pymongo
import numpy as np
import pandas as pd
from minepy import MINE
from sklearn import preprocessing
from sklearn.linear_model import LassoCV,Lasso,RidgeCV
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
# In[35]:
class Bunch(dict):
    def __init__(self,**kwargs):
        dict.__init__(self,kwargs)
    def __setattr__(self,key,value):
        self[key]=value
    def __getattr__(self,key):
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
def load_data1(filename):
    file_list=filename.strip().split(".")
    if file_list[-1]=="xlsx":
        data_frame=pd.read_excel(filename)
    elif file_list[-1]=="csv":
        data_frame=pd.read_csv(filename)
    else:
        print("this is vasp/clf file")
    dataset=data_frame.as_matrix()
    # data_set=[]
    # for data in dataset:
    #     data=map(float,data)
    #     data_set.append(data)
    # print(data_set)
    data_set=np.asarray(dataset,dtype=np.float)
    n_samples=dataset.shape[0]
    print n_samples
    n_features=dataset.shape[1]-1
    data=np.asarray(dataset[:,:-1],dtype=np.float)
    target=np.asarray(dataset[:,-1],dtype=np.float)
    feature_names=[column for column in data_frame]
    feature_names=np.array(feature_names)
    return Bunch(sample=n_samples, features=n_features, data=data, target=target,
                 feature_names=feature_names)
def load_data(filename):
    with open(filename) as f:
        data_file = csv.reader(f)
        data = []
        target = []
        temp = next(data_file)
        feature_names = np.array(temp)
        for i,d in enumerate(data_file):
            temp=[]
            for j in d:
                if j=='':
                    j=0
                temp.append(j)
            data.append(np.asarray(temp[:-1],dtype = np.float))
            target.append(np.asarray(d[-1],dtype = np.float))
        data = np.array(data)
        print(data)
        target = np.array(target)
        target = target.reshape(-1,1)
        n_samples = data.shape[0]
        print(n_samples)
        n_features = data.shape[1]
        print(n_features)
    return Bunch(sample = n_samples,features = n_features,data = data,target = target,
feature_names = feature_names)
# In[183]:
def ImportData(fileName):
    sampleData = load_data(fileName)
    return sampleData
# In[36]:
def Normalize(sampleData):
    minMaxScaler = preprocessing.MinMaxScaler()
    X = minMaxScaler.fit_transform(sampleData.data)
    Y = minMaxScaler.fit_transform(sampleData.target)
    return Bunch(X = X, Y = Y)
# In[37]:
def ValueCounts(inputX, threshold, sample, name):
    counts=[]
    arrayX=np.array(inputX)
    for i in xrange(len(inputX[0])):
        counts.append(pd.value_counts(arrayX[:,i]).values[0]*100.0/sample)
    indices = np.argsort(counts)[::-1]#返回counts中数组值降序排列的索引
    filtered = filter(lambda x: counts[x] > threshold, indices)
    indices = filter(lambda x: counts[x] <= threshold, indices)
    counts = np.array(counts)
    print "xi shu xi shu:"
    print "guo lv te zheng:",dict(zip(name[filtered],counts[filtered]))
    print "remain feature:",dict(zip(name[indices],counts[indices]))
    return Bunch(indices = indices, filtered = filtered, counts = counts)
# In[38]:
def SparsityEvaluate(inputX, threshold):
    # print threshold
    vt = VarianceThreshold()
    vt.fit_transform(inputX)
    importance = vt.variances_
    indices = np.argsort(importance)[::-1]
    filtered = filter(lambda x: importance[x] <= threshold, indices)
    indices = filter(lambda x: importance[x] > threshold, indices)
    importance = np.array(importance)
    return Bunch(indices = indices, filtered = filtered), importance, filtered, indices

# In[39]:

def SvrPrediction(trainX, trainY, testX):
    rbfSVR = GridSearchCV(SVR(kernel='rbf'), cv=5,
               param_grid={"C": np.logspace(-3, 3, 7),
               "gamma": np.logspace(-2, 2, 5)})
    rbfSVR.fit(trainX, trainY)
    predictY = rbfSVR.predict(testX)
    return predictY
# In[40]:
def modelEvaluation(testY,predictY):
    rmse=np.sqrt(mean_squared_error(testY,predictY))
    mae=mean_absolute_error(testY,predictY)
    r2 = r2_score(testY, predictY)
    return Bunch(rmse = rmse, mae = mae, r2 = r2)
# In[41]:
def LayerOneSelection(oneInputData,resultIndex,name):
    v_threshold=95
    threshold=0.01
    running=True
    seRmse=resultIndex.rmse
    optResultIndex=resultIndex
    outputX=oneInputData.dataX
    trainX, trainY = oneInputData.dataX[oneInputData.trainIndex], oneInputData.dataY[oneInputData.trainIndex]
    testX, testY = oneInputData.dataX[oneInputData.testIndex], oneInputData.dataY[oneInputData.testIndex]
    indices_all = [i for i in range(0,len(oneInputData.dataX[0]))]
    sIndex = ValueCounts(oneInputData.dataX, v_threshold, oneInputData.sample, name)
    v_indices = np.array(sIndex.indices)
    indices = v_indices
    outputX = oneInputData.dataX[:, v_indices]
    print len(indices)
    print name[indices]
    seIndex1, importance_all, filtered1, ind1=SparsityEvaluate(oneInputData.dataX,threshold)
    while running:
        seIndex, importance, filtered, ind = SparsityEvaluate(oneInputData.dataX[:, v_indices], threshold)
        seTrainX = trainX[:, v_indices[seIndex.indices]]
        seTestX = testX[:, v_indices[seIndex.indices]]
        seTrainY = trainY
        seTestY = testY
        dataLen = len(seIndex.indices)
        if dataLen > 0:
            sePredictY = SvrPrediction(seTrainX, seTrainY, seTestX)
            seResultIndex = modelEvaluation(seTestY,sePredictY)
            if seResultIndex.rmse <= seRmse:
                seRmse = seResultIndex.rmse
                optResultIndex = seResultIndex
                threshold = threshold + 0.01
                outputX = oneInputData.dataX[:, v_indices[seIndex.indices]]
                indices = v_indices[seIndex.indices]
            else:
                running = False
                # print threshold
        else:
            running = False
    criticalthreshold=threshold-0.01
    print criticalthreshold
    print "fang cha:"
    filter_indices = [i for i in indices_all if i not in indices]
    # print "特征:",dict(zip(name[v_indices],importance[v_indices]))
    return Bunch(oneDataX = outputX,
                 oneDataY = oneInputData.dataY,
                 trainIndex = oneInputData.trainIndex,
                 testIndex = oneInputData.testIndex,
                 resultIndex = optResultIndex,
                 indices = indices,
                 filter_indices=filter_indices,
                 coefficient1 = sIndex.counts,
                 coefficient2 = importance_all,i_threshold=v_threshold,
                 invar_threshold=threshold,
                 fivar_threshold=criticalthreshold)
# In[42]:
def feature_selection(n_samples, n_features, data, target, feature_names,username):
    # in_json = json.loads(json,encoding='utf-8')
    # filepath = in_json["filepath"]
    # remain = in_json["remain"]
    # username = in_json["username"]
    db = Connectdatabase()
    print "origin feature num",n_features
    no_target = False
    if target == []:
        no_target = True
    inputData = Bunch(sample = n_samples, features = n_features, data = data, target = target, feature_names = feature_names)
    print inputData
    normalizeData = Normalize(inputData)
    kf = KFold(inputData.sample, n_folds = 3)
    foldNum = 1
    totalRmse = 0
    for train_index, test_index in kf:
        print train_index, test_index
        kRmse = 0
        orgTrainX, orgTestX = normalizeData.X[train_index], normalizeData.X[test_index]
        orgTrainY, orgTestY = normalizeData.Y[train_index], normalizeData.Y[test_index]
        orgPredictY = SvrPrediction(orgTrainX, orgTrainY,orgTestX)
        orgResultIndex = modelEvaluation(orgTestY, orgPredictY)
        kRmse = orgResultIndex.rmse
        oneInputData = Bunch(dataX = normalizeData.X, dataY = normalizeData.Y, trainIndex = train_index, 
                             testIndex = test_index, sample = inputData.sample)
        if no_target == False:
            oneOutputData = LayerOneSelection(oneInputData, orgResultIndex, inputData.feature_names)
            print("The first retained features are:")
            # print(oneOutputData.indices)
            onereainindices = oneOutputData.indices
            onereainindices.tolist()#将numpy数组转成list
            print onereainindices
            filter_indices = oneOutputData.filter_indices
            # show = [i for i in first_remain if i not in oneOutputData.indices]
            # allshow = [i for i in [i for i in range(0,len(oneInputData.dataX[0]))] if i not in list(onereainindices)]
            # print show
            # show = map(int, show)
            # allshow = map(int, allshow)
            '''
            向outputparameter集合插入文档
            result1:算法第一层筛选后留下的特征序号
            filter_indices:第一层特征筛删除的特征序号
            sparse_coef:输出每个特征的稀疏系数
            variance:输出每个特征的方差值
            s_threshold:稀疏系数的阈值
            invar_threshold:设置的初始方差阈值
            fivar_threshold:模型验证的最终阈值
            在平台上用户可以选择是否删除filter_indices所指代的特征
            '''
            db.outputparameter.insert({"type": "onelayeroutput",
                                       "username": username,
                                       "result1": onereainindices.astype(np.int32).tolist(),
                                       "filter_indices":list(filter_indices),
                                       "sparse_coef":list(oneOutputData.coefficient1),
                                       "variance": list(oneOutputData.coefficient2),
                                       "s_threshold":oneOutputData.i_threshold,
                                       "invar_threshold":oneOutputData.invar_threshold,
                                       "fivar_threshold":oneOutputData.fivar_threshold}
                                      )
            print  oneOutputData.coefficient2
            print(len(oneOutputData.indices))
            print (inputData.feature_names[oneOutputData.indices])
            print
            return
# In[43]:
if __name__ == '__main__':
    # inputData = ImportData('glass.csv')
    # feature_selection(inputData.sample, inputData.features, inputData.data, inputData.target,
    #                   inputData.feature_names,"wujunming","0,1,2")
    parameterlist=[]
    for i in range(1, len(sys.argv)):
        para=sys.argv[i]
        parameterlist.append(para)
    print parameterlist
        # in_json = json.loads(json,encoding='utf-8')
        # filepath = in_json["filepath"]
    inputData = ImportData(parameterlist[0])
    print parameterlist[1]
    feature_selection(inputData.sample, inputData.features, inputData.data, inputData.target, inputData.feature_names,parameterlist[1])

