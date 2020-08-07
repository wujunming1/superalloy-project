#coding=utf8

import time
import numpy as np
import pandas as pd
from hyperopt import hp, Trials, tpe, fmin, STATUS_OK
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from hyperopt import fmin,tpe,hp
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

#读取文件操作
def load_data(filename):
    print(filename)
    data_file = pd.read_excel(filename)

    data = data_file.values
    n_samples = data.shape[0]
    feature_names = data_file.columns.values[:-1]
    n_features = feature_names.shape[0] - 1
    X_train = data[:, :-1]
    y_train = data[:, -1]

    return Bunch(sample=n_samples, features=n_features, data=data,
                 feature_names=feature_names, X_train=X_train, y_train=y_train)


def hyperopt_train_test(params):
    algorithm = params['type']
    del params['type']
    # print(params)
    if algorithm == 0:
        clf = LogisticRegression(**params)
    elif algorithm == 1:
        clf = SVC(**params)
    elif algorithm == 2:
        clf = LinearSVC(**params)
    elif algorithm == 3:
        clf = SGDClassifier(**params)
    elif algorithm == 4:
        clf = KNeighborsClassifier(**params)
    elif algorithm == 5:
        clf = GaussianNB(**params)
    elif algorithm == 6:
        clf = MultinomialNB(**params)
    elif algorithm == 7:
        clf = BernoulliNB(**params)
    elif algorithm == 8:
        clf = DecisionTreeClassifier(**params)
    elif algorithm == 9:
        clf = MLPClassifier(**params)
    else:
        return 0
    time_start = time.clock()
    print(time_start)
    score = cross_val_score(clf, train_data, y.astype('int').astype("int"), cv=10).mean()
    time_end = time.clock()
    print(time_end)
    train_time = (time_end - time_start)/10
    performance = score/(train_time ** (1/40))
    return score, train_time, performance

def f(params):
    global best_performance, best_score, best_time, best_params
    score, train_time, performance = hyperopt_train_test(params)
    if train_time > 1:
        if performance > best_performance:
            print('new best: ', performance, 'time: ',train_time, 'score: ', score,  'using ', params)
            best_performance = performance
            best_time = train_time
            best_score = score
            best_params = params
    else:
        if score > best_score:
            print('score: ', score, 'using ', params)
            best_performance = score
            best_time = train_time
            best_score = score
            best_params = params
    return {'loss': performance, 'time': train_time, 'score': score, 'status': STATUS_OK}

#选择算法操作，返回最好的模型参数，预测标签，得分情况，最好的模型（用于存储）
def select_algorithm(X_train, y_train, algorithm):
    if X_train.shape[1] > 10:
        max_features = 10
    else:
        max_features = X_train.shape[1]
    trials = Trials()
    space_logistic = {'C': hp.uniform('C', 0, 10),
                      'solver': hp.choice('solver', ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']), 'type': 0}
    space_svc = {'C': hp.uniform('C', 0, 10.0),
                 'kernel': hp.choice('kernel', ['linear', 'rbf', 'poly', 'sigmoid']),
                 'gamma': hp.uniform('gamma', 0, 20.0), 'type': 1}
    space_linearsvc = {'C': hp.uniform('C_l', 0, 10), 'type': 2}
    space_sgd = {'loss': hp.choice('loss',
                                   ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_loss',
                                    'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']),
                 'alpha': hp.uniform('alpha', 1e-7, 1e-3),
                 'l1_ratio': hp.uniform('l1_ratio', 0, 1),
                 'penalty': hp.choice('penalty', ['l2', 'l1', 'elasticnet']),
                 'learning_rate': hp.choice('learning_rate', ['constant', 'optimal', 'invscaling']),
                 'eta0': hp.uniform('eta0', 0, 5.0),
                 'power_t': hp.uniform('power_t', 0.0, 1.0),
                 'warm_start': hp.choice('warm_start', ['True', 'False']),
                 'tol': hp.uniform('tol', 1e-5, 1e-1), 'type': 3}
    space_knnc = {'n_neighbors': hp.choice('n_neighbors', range(1, 50)),
                  'algorithm': hp.choice('algorithm', ['ball_tree', 'kd_tree', 'brute']),
                  'leaf_size': hp.uniform('leaf_size', 20, 40), 'type' : 4}
    space_gaussiannb = {'var_smoothing': hp.uniform('var_smoothing', 1e-10, 1e-8), 'type': 5}
    space_multinomialnb = {'alpha': hp.uniform('alpha', 0.0, 2.0), 'type': 6}
    space_bernoullinb = {'alpha': hp.uniform('alpha', 0.0, 2.0), 'type': 7}
    space_dtc = {'max_depth': hp.choice('max_depth', range(1, 20)),
                 'splitter': hp.choice('splitter', ['best', 'random']),
                 'max_features': hp.choice('max_features', range(1, max_features)),
                 'criterion': hp.choice('criterion', ["gini", "entropy"]), 'type': 8}
    space_mlp = {'activation': hp.choice('activation', ['identity', 'logistic', 'tanh', 'relu']),
                 'solver': hp.choice('solver', ['lbfgs', 'sgd', 'adam']),
                 'alpha': hp.uniform('alpha', 1e-6, 1e-2),
                 'learning_rate': hp.choice('learning_rate', ['constant', 'invscaling', 'adaptive']),
                 'learning_rate_init': hp.uniform('learning_rate_init', 1e-5, 0.1),
                 'power_t': hp.uniform('power_t', 0.0, 1.0),
                 'momentum': hp.uniform('momentum', 0.0, 1.0),
                 'validation_fraction': hp.uniform('validation_fraction', 0.1, 0.3),
                 'beta_1': hp.uniform('beta_1', 0.0, 1.0),
                 'beta_2': hp.uniform('beta_2', 0.0, 1.0),
                 'epsilon': hp.uniform('epsilon', 1e-9, 1e-7), 'type': 9}
    try:
        if algorithm == 'svc':
            fmin(f, space_svc, algo=tpe.suggest, max_evals=100, trials=trials)
            clf = SVC(**best_params).fit(X_train, y_train)
            model = clf
            label = clf.predict(X_train)
            score = clf.score(X_train, y_train)
        elif algorithm == 'mlp':
            fmin(f, space_mlp, algo=tpe.suggest, max_evals=100, trials=trials)
            clf = MLPClassifier(**best_params).fit(X_train, y_train)
            model = clf
            label = clf.predict(X_train)
            score = clf.score(X_train, y_train)
        elif algorithm == 'sgd':
            fmin(f, space_sgd, algo=tpe.suggest, max_evals=100, trials=trials)
            clf = SGDClassifier(**best_params).fit(X_train, y_train)
            model = clf
            label = clf.predict(X_train)
            score = clf.score(X_train, y_train)
        elif algorithm == 'logi':
            fmin(f, space_logistic, algo=tpe.suggest, max_evals=100, trials=trials)
            clf = LogisticRegression(**best_params).fit(X_train, y_train)
            model = clf
            label = clf.predict(X_train)
            score = clf.score(X_train, y_train)
        elif algorithm == 'linearsvc':
            fmin(f, space_linearsvc, algo=tpe.suggest, max_evals=100, trials=trials)
            clf = LinearSVC(**best_params).fit(X_train, y_train)
            model = clf
            label = clf.predict(X_train)
            score = clf.score(X_train, y_train)
        elif algorithm == 'gnb':
            fmin(f, space_gaussiannb, algo=tpe.suggest, max_evals=100, trials=trials)
            clf = GaussianNB(**best_params).fit(X_train, y_train)
            model = clf
            label = clf.predict(X_train)
            score = clf.score(X_train, y_train)
        elif algorithm == 'mnb':
            fmin(f, space_multinomialnb, algo=tpe.suggest, max_evals=100, trials=trials)
            clf = MultinomialNB(**best_params).fit(X_train, y_train)
            model = clf
            label = clf.predict(X_train)
            score = clf.score(X_train, y_train)
        elif algorithm == 'bnb':
            fmin(f, space_bernoullinb, algo=tpe.suggest, max_evals=100, trials=trials)
            clf = BernoulliNB(**best_params).fit(X_train, y_train)
            model = clf
            label = clf.predict(X_train)
            score = clf.score(X_train, y_train)
        elif algorithm == 'dt':
            fmin(f, space_dtc, algo=tpe.suggest, max_evals=100, trials=trials)
            clf = DecisionTreeClassifier(**best_params).fit(X_train, y_train)
            model = clf
            label = clf.predict(X_train)
            score = clf.score(X_train, y_train)
        elif algorithm == 'knn':
            fmin(f, space_knnc, algo=tpe.suggest, max_evals=100, trials=trials)
            clf = KNeighborsClassifier(**best_params).fit(X_train, y_train)
            model = clf
            label = clf.predict(X_train)
            score = clf.score(X_train, y_train)

    except Exception as err:
        print(err)
        return err, err, err
    else:
        return model, label, score

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
    # algorithm_list = ['dbscan', 'kmeans', 'meanshift', 'agglomerative', 'ward', 'birch', 'affinity']

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    user_name = username
    file_name = file_name
    algorithm_name = alg
    data_file = load_data(file_path)
    X_train = data_file.X_train
    y_train = data_file.y_train
    global train_data, y
    global best_performance, best_score, best_time, best_params
    best_score = 0
    best_performance = 0
    best_time = 0
    best_params = None
    train_data = X_train
    y = y_train
    feature_names = data_file.feature_names
    feature_count = data_file.features
    xy = visualization(X_train)
    model, label, score = select_algorithm(X_train, y_train, alg)

    try:
        joblib.dump(model, 'D:\\superlloy\\automl\\classifier\\selection\\classifier_model\\' +
                    file_name.split('.')[0] + '_' + alg + '_model.m')
    except Exception as err:
        print(err)

    # print(date,user_name,file_name,algorithm_name,data,result_label,score,feature_names,feature_count,parameter_list,xy)

    print('Insert DataBase:')
    db = Connectdatabase()
    db.ClassifierModel.insert({
        "date": date,
        "user_name": user_name,
        "file_name": file_name,
        "algorithm_name": algorithm_name,
        "data": X_train.astype(np.float64).tolist(),
        "result_label": y_train.astype(np.float64).tolist(),
        "score": score,
        "feature_names": feature_names.astype('object').tolist(),
        "feature_count": feature_count,
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

    # insert('iris.xlsx', r'D:\gwhj\superlloy\data\piki\iris.xlsx', 'sgd', 'piki')
