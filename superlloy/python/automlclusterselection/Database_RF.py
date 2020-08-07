import pandas as pd
import numpy as np
import math
import scipy.stats
import random
from sklearn.ensemble import RandomForestRegressor
import pymongo
import time
import sys

#将三种推荐算法结合
def merge(Item,User,Model):
    input = []
    for i in range(len(Item)):
        temp = np.hstack((Item[i][1:], User[i][1:], Model[i][1:]))
        input.append(temp.tolist())
    input = np.array(input)
    return input

#将三种推荐算法结合
def merge_test(Item,User,Model):
    input = Item[1:] + User[1:] + Model[1:]
    # input.append(Item[1:])
    # input.append(User[1:])
    # input.append(Model[1:])
    # input = np.array(input)
    return input

#返回随机森林模型
def get_rf(input,output):
    clf = RandomForestRegressor(max_depth=2, random_state=0, n_jobs=2, n_estimators=100)
    clf.fit(input,output)
    return clf

def recommend(model_list, data):
    pre = {}
    for i in range(len(alg)):
        temp = model_list[i].predict(data)
        pre[alg[i]] = temp.tolist()[0]
    pre_sorted = sorted(pre.items(), key = lambda x : x[1], reverse = True)
    print(pre_sorted)
    alg_list = []
    for i in range(2):
        alg_list.append(pre_sorted[i][0])
    return alg_list

#链接MongoDB数据库
def Connectdatabase():
    conn=pymongo.MongoClient(host="localhost",port=27017)
    db=conn.MongoDB_Data
    return db

def get_recommend(type,alg,file_name):
    db = Connectdatabase()
    data = db.RecommendResult.find_one({'type':type})
    recommend = data['alg_recommend']
    # print(recommend)
    temp = [file_name]
    for j in range(len(alg)):
        if alg[j] in recommend:
            temp.append(1)
        else:
            temp.append(0)
    print(temp)
    return temp

def insert(file_name,file_path,user_name,model_list,alg):
    db = Connectdatabase()
    TestItem_values = get_recommend('Item_Based', alg, file_name)
    TestUser_values = get_recommend('User_Based', alg, file_name)
    TestModel_values = get_recommend('Model_Based', alg, file_name)
    print(TestItem_values)
    Test_input = merge_test(TestItem_values, TestUser_values, TestModel_values)
    print(Test_input)
    df_Test = pd.DataFrame(index=[file_name], columns=alg)
    data = np.array(Test_input).reshape(1, -1)
    temp_recommend = recommend(model_list, data)
    print(temp_recommend)
    for j in range(len(alg)):
        if alg[j] in temp_recommend:
            df_Test.values[0][j] = 1
        else:
            df_Test.values[0][j] = 0
    df_Test.to_csv(
        'D:\\superlloy\\automl\\cluster\\selection\\recommend_rf\\' + file_name.rstrip(
            '.xls') + '_rf.csv')

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    file_name = file_name
    user_name = user_name
    temp_recommend = temp_recommend
    type = 'RF_Based'
    print('Insert DataBase:')
    db.RecommendResult.insert({
        "date": date,
        "file_name": file_name,
        "user_name": user_name,
        "alg_recommend": temp_recommend,
        "type": type
    })
    print('End Insert')


if __name__ == '__main__':
    Train2Result_path = 'D:\\superlloy\\automl\\cluster\\selection\\Train2\\Train2_Result.csv'
    Train2Result_df = pd.read_csv(Train2Result_path)
    Train2Result_values = Train2Result_df.values

    Train2ItemPath = 'D:\\superlloy\\automl\\cluster\\selection\\Train2\\Train2-ItemBased.csv'
    Train2UserPath = 'D:\\superlloy\\automl\\cluster\\selection\\Train2\\Train2-UserBased.csv'
    Train2ModelPath = 'D:\\superlloy\\automl\\cluster\\selection\\Train2\\Train2-ModelBased.csv'
    Train2Item_df = pd.read_csv(Train2ItemPath)
    Train2Item_values = Train2Item_df.values
    Train2User_df = pd.read_csv(Train2UserPath)
    Train2User_values = Train2User_df.values
    Train2Model_df = pd.read_csv(Train2ModelPath)
    Train2Model_values = Train2Model_df.values
    # 三种推荐算法矩阵
    Train2_input = merge(Train2Item_values, Train2User_values, Train2Model_values)
    # 性能矩阵
    Train2_output = np.array([temp[1:].tolist() for temp in Train2Result_values])

    alg = Train2Result_df.columns[1:]
    cluster_list = ['kmeans', 'meanshift']

    model_list = []
    for i in range(len(alg)):
        model_list.append(get_rf(Train2_input, Train2_output[:, i]))

    # python .py file_name0 file_path1  username2
    length_argv = len(sys.argv)
    print(length_argv)

    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)
    # file_name = 'auto-test.xlsx'
    # file_path = '/Users/buming/Documents/Super_Alloy/SuperAlloy_System/superalloy/data/piki/auto-test.xlsx'
    # username = 'piki'
    insert(parameterlist[0], parameterlist[1], parameterlist[2], model_list,alg)

# TestItemPath = '/Users/buming/Documents/Super_Alloy/SuperAlloy_System/superalloy/automl/cluster/selection/recommend_item/'+ file_name.rstrip(
    #         '.xls') + '_itembased.csv'
    # TestUserPath = '/Users/buming/Documents/Super_Alloy/SuperAlloy_System/superalloy/automl/cluster/selection/recommend_user/'+ file_name.rstrip(
    #         '.xls') + '_userbased.csv'
    # TestModelPath = '/Users/buming/Documents/Super_Alloy/SuperAlloy_System/superalloy/automl/cluster/selection/recommend_model/'+ file_name.rstrip(
    #         '.xls') + '_modelbased.csv'
    # TestItem_df = pd.read_csv(TestItemPath)
    # TestItem_values = TestItem_df.values
    # print(type(TestItem_values[0]))
    # TestUser_df = pd.read_csv(TestUserPath)
    # TestUser_values = TestUser_df.values
    # TestModel_df = pd.read_csv(TestModelPath)
    # TestModel_values = TestModel_df.values
    # # 三种推荐算法矩阵
    # Test_input = merge(TestItem_values, TestUser_values, TestModel_values)
    # Test_FileList = [temp[0] for temp in TestModel_values]
    # # print(Test_input,Test_FileList)
    #
    # df_Test = pd.DataFrame(index = Test_FileList, columns = alg)
    # for i in range(len(Test_FileList)):
    #     data = np.array(Test_input[i]).reshape(1,-1)
    #     temp_recommend = recommend(model_list, data)
    #     print(temp_recommend)
    #     for j in range(len(alg)):
    #         if alg[j] in temp_recommend:
    #             df_Test.values[i][j] = 1
    #         else:
    #             df_Test.values[i][j] = 0
    # df_Test.to_csv('/Users/buming/Documents/Super_Alloy/DataSet/result1/Test-RF.csv')





