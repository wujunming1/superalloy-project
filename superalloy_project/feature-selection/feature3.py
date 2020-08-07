
# coding: utf-8

# In[1]:

import os
import time
import csv
import numpy as np
import pandas as pd
import pymongo
import sys
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
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn import decomposition
from sklearn import neighbors
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge


# In[2]:

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
        target = np.array(target)
        target = target.reshape(-1, 1)
        n_samples = data.shape[0]
        n_features = data.shape[1]
    return Bunch(sample = n_samples,features = n_features,data = data,target = target,
feature_names = feature_names)
# In[183]:

def ImportData(fileName):
    sampleData = load_data(fileName)
    return sampleData


# In[3]:

def Normalize(sampleData):
    minMaxScaler = preprocessing.MinMaxScaler()
    X = minMaxScaler.fit_transform(sampleData.data)
    Y = minMaxScaler.fit_transform(sampleData.target)
    return Bunch(X = X, Y = Y)


# In[4]:
from sklearn import tree
import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
def RandomForest(dataX, dataY,feature_names):#randomforest reduce dimension
    rf = RandomForestRegressor(n_estimators=6)
    rf.fit(dataX, dataY)
    Estimators=rf.estimators_
    # numTrees=len(Estimators)
    # print numTrees
    # for num in range(0,numTrees):
    #     dot_data = tree.export_graphviz(Estimators[num], out_file=None,filled=True, rounded=True,
    #                                     feature_names=feature_names,
    #                                     special_characters=True)
    #     graph = pydotplus.graph_from_dot_data(dot_data)
    #     graph.write_png("iris" + str(num) + ".png")
    importance = rf.feature_importances_
    indices = np.argsort(importance)
    return importance,indices
def LassoReduceDim(dataX,dataY,indices):#lasso回归做降维
    lassocv=LassoCV()
    lassocv.fit(dataX,dataY)
    importance=lassocv.coef_
    mark=importance!=0
    impindices=np.arange(len(importance))
    index=[inx for inx in impindices if importance[inx]!=0]
    # lassoindices=[]
    # for i in index:
    #     lassoindices.append(indices[i])
    # # new_dataX=dataX[:,mark]
    return index
def SvrPrediction(trainX, trainY, testX):
    rbfSVR = GridSearchCV(SVR(kernel='rbf'), cv=5,
               param_grid={"C": np.logspace(-3, 3, 7),
               "gamma": np.logspace(-2, 2, 5)})
    rbfSVR.fit(trainX, trainY)
    predictY = rbfSVR.predict(testX)
    return predictY
def modelEvaluation(testY,predictY):
    rmse=np.sqrt(mean_squared_error(testY,predictY))
    mae=mean_absolute_error(testY,predictY)
    r2 = r2_score(testY, predictY)
    return Bunch(rmse = rmse, mae = mae, r2 = r2)


# In[5]:

def LayerThreeSelection(threeInputData, resultIndex, indices,feature_names):
    ###m>5000或者n>40 select randomforest;otherwise choose Lasso;###
    reRmse = resultIndex.rmse
    optResultIndex = resultIndex
    dataX = threeInputData.dataX[:,indices]
    m,n=dataX.shape
    dataY = threeInputData.dataY
    outputX = threeInputData.dataX[:,indices]
    outputX=np.array(outputX)
    m,n=outputX.shape
    trainX, trainY = threeInputData.dataX[threeInputData.trainIndex], threeInputData.dataY[threeInputData.trainIndex]
    testX, testY = threeInputData.dataX[threeInputData.testIndex], threeInputData.dataY[threeInputData.testIndex]
    if m<5000 or n<40:
        importance,coorIndex=RandomForest(dataX, dataY,feature_names)
        optIndex = coorIndex
        for i in range(1, len(coorIndex)):
            coorIn = [] * (len(coorIndex) - i)
            for j in range(i, len(coorIndex)):
                coorIn.append(coorIndex[j])
            rePredictY = SvrPrediction(trainX[:, coorIn], trainY, testX[:, coorIn])
            reResultIndex = modelEvaluation(testY, rePredictY)
            if reResultIndex.rmse <= reRmse:
                reRmse = reResultIndex.rmse
                optIndex = coorIn
                optResultIndex = reResultIndex
            else:
                break
    else:
        optIndex=LassoReduceDim(dataX,dataY,indices)
    outputX = threeInputData.dataX[:,optIndex]
    return Bunch(threeDataX = outputX, threeDataY = dataY, trainIndex = threeInputData.trainIndex, 
    testIndex = threeInputData.testIndex, resultIndex = optResultIndex,
                 optIndex = optIndex,importance=importance)

# In[6]:
def feature_selection(n_samples, n_features, data, target, feature_names,username,expert_remain2):
    print(type(expert_remain2))
    print(expert_remain2)
    if len(expert_remain2)==0 or expert_remain2==None or expert_remain2=="null":
        print "hello"
        expert_remain2=[]
    else:
        expert_remain2=map(int,expert_remain2.split(","))

    print "原始特征数：",n_features
    no_target = False
    if target == []:
        no_target = True
    inputData = Bunch(sample = n_samples, features = n_features, data = data, target = target, feature_names = feature_names)
    normalizeData = Normalize(inputData)

    kf = KFold(inputData.sample, n_folds = 3)
    foldNum = 1
    totalRmse = 0
    result=[]
    first_remain=[]
    db=Connectdatabase()
    oneInputDatadict=db.outputparameter.find()
    for i in oneInputDatadict:
        if "result2" in i.keys():
            result=i["result2"]
    for train_index, test_index in kf:
        # indices =[3, 0, 6, 23, 15, 25, 19, 14, 1, 26, 20, 5, 12, 11, 8, 16, 21, 10, 13, 2, 9, 17, 24, 22, 18, 7]
        indices =list(result)+list(expert_remain2)
        print indices
        kRmse = 0
        orgTrainX, orgTestX = normalizeData.X[train_index], normalizeData.X[test_index]
        orgTrainY, orgTestY = normalizeData.Y[train_index], normalizeData.Y[test_index]
        orgPredictY = SvrPrediction(orgTrainX[:,indices], orgTrainY,orgTestX[:,indices])
        orgResultIndex = modelEvaluation(orgTestY, orgPredictY)
        kRmse = orgResultIndex.rmse
        oneInputData = Bunch(dataX = normalizeData.X, dataY = normalizeData.Y, trainIndex = train_index, 
                             testIndex = test_index, sample = inputData.sample)
        if no_target == False:
            threeOutputData = LayerThreeSelection(oneInputData, orgResultIndex, indices
                                                  ,inputData.feature_names[indices])
            importance=threeOutputData.importance
            threeRetainIndex = [] * len(threeOutputData.optIndex)
            for i in range(0,len(threeOutputData.optIndex)):
                threeRetainIndex.append(indices[threeOutputData.optIndex[i]])
            # threeremain
            print("The three retain features are:")
            print len(threeRetainIndex)
            print(threeRetainIndex)
            filter_indices= [i for i in indices if i not in threeRetainIndex]
            '''
            usrname:用户名
            result3:第三层特征筛留下的特征
            filter_indices:第三层特征筛删除的特征
            importance:第三层输入的所有特征的重要度(平均不纯度)
            '''
            db.outputparameter.insert({"type":"threelayeroutput",
                                       "username":username,
                                       "importance":list(importance),
                                       "result3":threeRetainIndex,
                                       "filter_indices":filter_indices
                                       })
            # db.oneoutputdata.insert({"show3":threeRetainIndex})
            print inputData.feature_names[threeRetainIndex]
            print
#         out_json = []
#         out_json.append(inputData.feature_names[threeRetainIndex])
#         out_json.append(inputData.feature_names[show])
#         python_to_json = json.dumps(out_json,ensure_ascii=False)
        return
# In[7]:
if __name__ == '__main__':
    # inputData = ImportData('glass.csv')
    # feature_selection(inputData.sample, inputData.features,
    #                   inputData.data, inputData.target, inputData.feature_names,"wujunming")
    '''
    :parameterlist:接收来自平台传来的参数列表
    
    '''
    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print parameterlist
    # in_json = json.loads(json,encoding='utf-8')
    # filepath = in_json["filepath"]

    inputData = ImportData(parameterlist[0])
    print parameterlist[1]
    feature_selection(inputData.sample, inputData.features,
                      inputData.data, inputData.target,
                      inputData.feature_names
                      ,parameterlist[1],parameterlist[2])

