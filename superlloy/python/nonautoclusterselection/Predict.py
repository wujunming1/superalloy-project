import sys
import time
import extension
import errors
import pandas as pd
import numpy as np
from sklearn.externals import joblib
import pymongo

def run_model(data,model):
    """
    导入模型，进行预测
    :param data:要进行预测的测试数据集，dataframe数据
    :param model:要导入的模型文件名
    :return:预测完成的dataframe数据
    """
    model = joblib.load('D:/superlloy/cluster/cluster_model/'+model)
    predict_result= model.predict(data)
    return predict_result

def read_data(path):
    """
    读取路径下的文件数据
    :param path: 文件路径
    :return: DataFrame类型的数据
    """
    try:
        file_extension = extension.file_extension(path)
        if file_extension == "csv":
            data_frame = pd.read_csv(path)
        elif file_extension == "xls" or file_extension == "xlsx":
            data_frame = pd.read_excel(path)
        elif file_extension == "txt":
            csv = np.loadtxt(path)
            data_frame = pd.DataFrame(csv)
        else:
            raise errors.DataTypeError
        return data_frame
    except Exception as e:
        print(e)

#链接MongoDB数据库
def Connectdatabase():
    conn=pymongo.MongoClient(host="localhost",port=27017)
    db=conn.MongoDB_Data
    return db

def insert(file_name,file_path,alg,model,username):
    """
        插入预测之后的数据到mongodb
        :param file_name: 文件名
        :param file_path: 文件的绝对路径
        :param alg: 此处固定为Predict
        :param model: 需要载入的.m模型文件
        :param username: 进行预测的用户名
        :return: 无
    """
    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    user_name = username
    file_name = file_name
    algorithm_name = alg
    data = read_data(file_path)
    feature_names = data.columns.values
    feature_count = data.shape[1]

    predict_result = run_model(data, model)

    print('Insert DataBase:')
    db = Connectdatabase()
    db.NonAutoClusterModel.insert({
        "date": date,
        "user_name": user_name,
        "file_name": file_name,
        "algorithm_name": algorithm_name,
        "data": data.astype(np.float64).values.tolist(),
        "result_label":  predict_result.astype(np.float64).tolist(),
        "feature_names": feature_names.astype('object').tolist(),
        "feature_count": feature_count,
    })
    print('End Insert')

if __name__ == "__main__":
	# ['iris.xlsx', 'D:\\superlloy\\data\\piki\\iris.xlsx', 'Predict', 'iris_svm_model.m','piki']
    length_argv = len(sys.argv)
    print(length_argv)
    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)

    insert(parameterlist[0],parameterlist[1],parameterlist[2],parameterlist[3],parameterlist[4])