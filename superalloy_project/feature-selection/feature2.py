
# coding: utf-8

# In[31]:

import os
import time
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')
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
import pandas as pd
import openpyxl
import xlrd
import sys
import json
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
# In[32]:
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
    '''
    连接数据库
    :return:
    '''
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
        target = target.reshape(-1,1)
        n_samples = data.shape[0]
        n_features = data.shape[1]
    return Bunch(sample = n_samples,features = n_features,data = data,target = target,
feature_names = feature_names)
# In[183]:

def ImportData(fileName):
    sampleData = load_data(fileName)
    return sampleData
# In[33]:
def Normalize(sampleData):
    minMaxScaler = preprocessing.MinMaxScaler()
    X = minMaxScaler.fit_transform(sampleData.data)
    Y = minMaxScaler.fit_transform(sampleData.target)
    return Bunch(X = X, Y = Y)

def excel_to_db(filename):
    '''
    excel表中的数据写入到mongodb数据库中
    :param filename:
    :return:
    '''
    db=Connectdatabase()
    account = db.weibo
    data = xlrd.open_workbook(filename)
    table = data.sheets()[0]
    # 读取excel第一行数据作为存入mongodb的字段名
    rowstag = table.row_values(0)
    nrows = table.nrows
    # ncols=table.ncols
    # print rows
    returnData = {}
    for i in range(1, nrows):
        # 将字段名和excel数据存储为字典形式，并转换为json格式
        returnData[i] = json.dumps(dict(zip(rowstag, table.row_values(i))))
        # 通过编解码还原数据
        returnData[i] = json.loads(returnData[i])
        # print returnData[i]
        account.insert(returnData[i])
# In[34]:

def MicEvaluate(dataX,dataY,name):
    '''
    计算每一个条件属性与决策属性之间的最大信息系数
    :param dataX:条件属性X
    :param dataY:决策属性Y
    :param name:
    :return:
    '''
    dataY = dataY.reshape(1,-1)[0]
    nFeatures = len(dataX[0])
    coorArray = [] * nFeatures
    mine = MINE(alpha=0.6, c=15)
    for i in range(0, nFeatures):
        l = [x[i] for x in dataX]
        mine.compute_score(l, dataY)
        temp = mine.mic()
        coorArray.append(abs(temp))
    coorIndex = np.argsort(coorArray)
    cooroptIndex = coorIndex
    coorArray = np.array(coorArray)
    print ("MIC:")
    print "feature:",dict(zip(name[coorIndex],coorArray[coorIndex]))
    return cooroptIndex, coorArray
# In[35]:
def CorrelationEvaluate(dataX,dataY,name):
    '''
    计算每一个条件属性与决策属性之间的皮尔逊相关系数
    :param dataX:
    :param dataY:
    :param name:
    :return:
    '''
    dataY = dataY.reshape(1, -1)[0]
    nFeatures = len(dataX[0])
    coorArray = [] * nFeatures
    for i in range(0, nFeatures):
        l = [x[i] for x in dataX]
        coor = pearsonr(l, dataY)
        coorArray.append(abs(coor[0]))
    coorIndex = np.argsort(coorArray)
    cooroptIndex=coorIndex
    coorArray = np.array(coorArray)
    print "pearson:"
    print "feature:",dict(zip(name[coorIndex],coorArray[coorIndex]))
    return cooroptIndex, coorArray
def Condfea_corcoef(inputData):
    '''
    @author:wujunming
    计算条件属性之间的相关系数
    :param inputData:
    :return:
    '''
    feature_colums = inputData.feature_names
    sample = inputData.sample
    sample_index = np.arange(sample)
    df = pd.DataFrame(inputData.data, index=sample_index, columns=feature_colums[:-1])
    c = df.corr()
    isExists=os.path.exists("E:\\superalloy\\cor_coeff")
    if not isExists:
        os.makedirs("E:\\superalloy\\cor_coeff")
        print("目录创建成功")
    else:
        print("目录已存在")
    c.to_excel("E:\\superalloy\\cor_coeff\\cor_coeff.xlsx")

# In[36]:

def SvrPrediction(trainX, trainY, testX):
    rbfSVR = GridSearchCV(SVR(kernel='rbf'), cv=5,
               param_grid={"C": np.logspace(-3, 3, 7),
               "gamma": np.logspace(-2, 2, 5)})
    rbfSVR.fit(trainX, trainY)
    predictY = rbfSVR.predict(testX)
    return predictY
# In[37]:
def modelEvaluation(testY,predictY):
    rmse=np.sqrt(mean_squared_error(testY,predictY))
    mae=mean_absolute_error(testY,predictY)
    r2 = r2_score(testY, predictY)
    return Bunch(rmse = rmse, mae = mae, r2 = r2)
# In[38]:

def LayerTwoSelection(twoInputData, resultIndex, sample, indices, name):
    ceRmse = resultIndex.rmse
    optResultIndex = resultIndex
    dataX = twoInputData.dataX[:,indices]
    dataY = twoInputData.dataY
    trainX, trainY = twoInputData.dataX[twoInputData.trainIndex], twoInputData.dataY[twoInputData.trainIndex]
    testX, testY = twoInputData.dataX[twoInputData.testIndex], twoInputData.dataY[twoInputData.testIndex]
    '''
    样本数大于100时，选择最大信息相关系数
    否则，选择皮尔逊相关系数
    '''
    if (sample>100):
        coorIndex, coorArray = MicEvaluate(dataX, dataY, name)
        optIndex = coorIndex
        coefficient = coorArray
    else:
        coorIndex, coorArray = CorrelationEvaluate(dataX, dataY, name)
        optIndex = coorIndex
        coefficient = coorArray
    incor_threshold=coorArray[coorIndex[0]]#初始的相关系数阈值
    for i in range(1, len(coorIndex)):
        coorIn = [] * (len(coorIndex) - i)
        for j in range(i, len(coorIndex)):
            coorIn.append(coorIndex[j])
        cePredictY = SvrPrediction(trainX[:, coorIn], trainY, testX[:, coorIn])
        ceResultIndex = modelEvaluation(testY, cePredictY)
        if ceResultIndex.rmse <= ceRmse:
            ceRmse = ceResultIndex.rmse
            optIndex = coorIn
            optResultIndex = ceResultIndex
        else:
            print coorArray[j]
            break
    ficor_threshold=coorArray[j]
    outputX = dataX[:, optIndex]
    '''
    coefficient:每个特征的相关系数
    i_threshold:初始的相关系数阈值
    f_threshold:最终的相关系数阈值
    '''
    return Bunch(twoDataX = outputX, twoDataY = dataY, trainIndex = twoInputData.trainIndex, 
                 testIndex = twoInputData.testIndex, resultIndex = optResultIndex, optIndex = optIndex,
                 coefficient = coefficient,i_threshold=incor_threshold,f_threshold=ficor_threshold)
# In[39]:

# def draw_heatmap(data,xlabels,ylabels):
#     cmap = cm.get_cmap('rainbow', 1000)
#     figure=plt.figure(facecolor='w')
#     ax=figure.add_subplot(1,1,1,position=[0.1,0.15,0.8,0.8])
#     ax.set_yticks(range(len(ylabels)))
#     ax.set_yticklabels(ylabels)
#     ax.set_xticks(range(len(xlabels)))
#     ax.set_xticklabels(xlabels)
#     vmax=data[0][0]
#     vmin=data[0][0]
#     for i in data:
#         for j in i:
#             if j>vmax:
#                 vmax=j
#             if j<vmin:
#                 vmin=j
#     map=ax.imshow(data,interpolation='nearest',cmap=cmap,aspect='auto',vmin=vmin,vmax=vmax)
#     cb=plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
#     plt.show()
#     # plt.savefig("1.png")
def feature_selection(n_samples, n_features, data,target, feature_names,username,expert_remain1):
    '''
    :param n_samples: 样本数
    :param n_features: 属性个数
    :param data: 输入X
    :param target: 输出Y
    :param feature_names: 特征名
    :param username: 使用该算法的用户名
    :param expert_remain1: 专家在第一层删除的特征中选择要保留的属性
    :return:
    '''
    # if(len(expert_remain1)==0 or expert_remain1==None):
    #     expert_remain1=None
    # else:
    #     expert_remain1=map(int,expert_remain1.split(","))
    # print expert_remain1
    if len(expert_remain1)==0 or expert_remain1==None or expert_remain1=="null":
        expert_remain1=[]
    else:
        expert_remain1=map(int,expert_remain1.split(","))
    print "---------------"
    print "origin feature num:",n_features
    no_target = False
    if target == []:
        no_target = True
    inputData = Bunch(sample = n_samples, features = n_features, data = data, target = target, feature_names = feature_names)
    # feature_colums = inputData.feature_names
    # sample = inputData.sample
    # sample_index = np.arange(sample)
    # df = pd.DataFrame(inputData.data, index=sample_index, columns=feature_colums[:-1])
    # c = df.corr()
    # c.to_excel("1.xlsx")
    # e=c.as_matrix(columns=None)
    # # df=pd.read_excel("1.xlsx")
    # xlabels = feature_colums
    # ylabels = feature_colums
    # draw_heatmap(e, xlabels, ylabels)
    normalizeData = Normalize(inputData)
    kf = KFold(inputData.sample, n_folds = 3)
    foldNum = 1
    totalRmse = 0
    db=Connectdatabase()
    result=[]
    first_remain=[]
    oneInputDatadict=db.outputparameter.find()
    for i in oneInputDatadict:
        if "result1" in i.keys():
            result=i["result1"]
            print i["result1"]
    print oneInputDatadict
    for train_index, test_index in kf:
        # indices = [3,4,5,24,8,15,12,20,26,1,2,6,7,11,0,14,16,17,18,19,22,25,10,9,21,23,13]
        print("input feature to second :")
        indices=list(result)+list(expert_remain1)
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
            twoOutputData = LayerTwoSelection(oneInputData, orgResultIndex, inputData.sample, indices, inputData.feature_names[indices])
            corcoeff=twoOutputData.coefficient #第二层输入的所有条件属性与决策属性之间的相关系数
            i_threshold=twoOutputData.i_threshold
            f_threshold=twoOutputData.f_threshold
            twoRetainIndex = [] * len(twoOutputData.optIndex)
            for i in range(0, len(twoOutputData.optIndex)):
                twoRetainIndex.append(indices[twoOutputData.optIndex[i]])
            print("The second retained features are:")
            print len(twoRetainIndex)
            print twoRetainIndex
            filter_indices=[i for i in indices if i not in twoRetainIndex]
            '''
            corcoefficent:第二层输入的所有特征的相关系数
            incor_threshold:初始的相关系数阈值
            ficor_threshold:最终的相关系数阈值
            result2:第二层特征筛留下的特征
            filter_indices:第二层特征筛删除的特征
            '''
            db.outputparameter.insert({"type":"twolayeroutput",
                                       "corcoefficent":list(corcoeff),
                                       "incor_threshold":i_threshold,
                                       "ficor_threshold":f_threshold,
                                       "username":username,
                                       "result2":list(twoRetainIndex),
                                       "filter_indices":list(filter_indices)
                                       })
            print inputData.feature_names[twoRetainIndex]
            print twoOutputData.coefficient
            return
# In[40]:
if __name__ == '__main__':
    # inputData = ImportData('glass.csv')
    
    # '''
    #
    # '''
    # feature_selection(inputData.sample, inputData.features, inputData.data, inputData.target, inputData.feature_names,"wujunming","2")
    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print parameterlist
    inputData = ImportData(parameterlist[0])
    print type(inputData)
    Condfea_corcoef(inputData)
    print parameterlist[1]
    feature_selection(inputData.sample, inputData.features, inputData.data, inputData.target, inputData.feature_names,parameterlist[1],parameterlist[2])

