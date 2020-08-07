import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error,r2_score
import math


def gaussian_model(paras):

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    alpha = float(paras[0])
    optimizer = int(paras[1])
    # parameter = "C(1, (1e-4, 10000)) * RationalQuadratic(alpha=0.01, length_scale_bounds=(1e-5, 20))"
    parameter = ' gaussian_model '
    # kernel_5 = C(1, (1e-4, 1)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.01, 2000))
    kernel = C(1, (0.1, 10)) * RationalQuadratic(alpha=0.01, length_scale_bounds=(0.1, 1500))
    model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=optimizer)
    return model, parameter


def svr_model(paras):
    #SVR model
    kernel = paras[0]
    C = float(paras[1])
    from sklearn.svm import SVR
    parameter = ' svr_model '
    model = SVR(kernel=kernel, C=C, gamma='auto')
    return model, parameter


def random_forest_model(paras):
    #random forest model

    from sklearn.ensemble import RandomForestRegressor
    parameter = ' RandomForest_model '
    estimators = int(paras[0])
    depth = int(paras[1])
    # model = RandomForestRegressor(n_estimators=15, max_depth=4, criterion='mae', bootstrap=True)
    model = RandomForestRegressor(n_estimators=estimators, max_depth=depth, criterion='mae', bootstrap=True)
    return model, parameter


def pre_progressing(original_data):
    #Normalization X: Max-Min normalization Y :log scale
    data = original_data[:, :-1]
    target = original_data[:, -1]
    normalize_data = MinMaxScaler().fit_transform(data)
    normalize_target = np.log(target)
    return normalize_data, normalize_target


def lasso_regression(paras):
    from sklearn.linear_model import Ridge, Lasso
    parameter = 'Lasso_model'
    alpha = float(paras[0])
    model = Lasso(alpha=alpha) #alpha = 0.1,,0.5,1.0
    return model, parameter


def ridge_model(paras):
    from sklearn.linear_model import Ridge
    alpha = float(paras[0])
    parameter = ' Elastic_model '
    model = Ridge(alpha=alpha)#0.5
    return model, parameter


def fitness_fuction(y_pred, y_true):
    return 1-abs(np.sum((y_pred-y_true)/y_true)/len(y_pred))



def mape_function(y_pred, y_true):
    return abs(np.sum((y_pred - y_true) / y_true) / len(y_pred))


def get_feature_importance():
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    df = pd.read_excel("data_files/多尺度样本_revision.xlsx")
    feature_name = [column for column in df][1:]
    print(feature_name)
    sample_data = df.values[:, 1:]  # 多尺度数据集
    # sample_data = df.values[:, 1:23]
    print(sample_data)
    original_dataX = sample_data[:, :-1]
    train_dataX, train_dataY = pre_progressing(sample_data)
    k = 10 # or k=10
    kf = KFold(n_splits=k)
    rf = RandomForestRegressor()
    rf.fit(train_dataX, train_dataY)
    feature_impor_all_data = rf.feature_importances_
    f_importances = np.zeros(27,)
    for train_index, test_index in kf.split(train_dataX):
        trainX, trainY = train_dataX[train_index], train_dataY[train_index]
        testX, testY = train_dataX[test_index], train_dataY[test_index]
        rf.fit(trainX, trainY)
        f_importances+=rf.feature_importances_
    x_train, x_test, y_train, y_test = train_test_split( train_dataX, train_dataY, test_size=0.01, random_state=0)
    rf = RandomForestRegressor()
    gbr = GradientBoostingRegressor()
    rf.fit(x_train, y_train)

    print(rf.feature_importances_)
    importances = rf.feature_importances_
    average_importance = f_importances/k
    indices = np.argsort(average_importance)[::-1]
    features = []
    feature_importances = []
    for f in range(train_dataX.shape[1]):
        features.append(feature_name[indices[f]])
        feature_importances.append(average_importance[indices[f]])
        print("%2d) %-*s %f" % (f + 1, 30, feature_name[indices[f]],average_importance[indices[f]]))
    # model1.fit(train_dataX, train_dataY)
    f_im = np.column_stack((features, feature_importances))
    df = pd.DataFrame(f_im, columns=["特征名", "重要度"])
    df.to_excel("feature_importance.xlsx")


def run_validation_microdescriptor():
    df = pd.read_excel("data_files/多尺度样本_revision.xlsx")
    feature_name = [column for column in df][1:]
    print(feature_name)
    sample_data = df.values[:, 1:] #多尺度数据集
    # sample_data = df.values[:, 1:23]
    print(sample_data)
    original_dataX = sample_data[:, :-1]
    train_dataX, train_dataY = pre_progressing(sample_data)
    train_dataX = train_dataX[:, 0:23]
    k = 10 # or k=10
    kf = KFold(n_splits=k)
    model, parameter = lasso_regression() #svr_model()
    sum_rmse = 0
    sum_mape = 0
    y_pred = []
    rmse_all = []
    mape_all = []
    for train_index, test_index in kf.split(train_dataX):
        trainX, trainY = train_dataX[train_index], train_dataY[train_index]
        testX, testY = train_dataX[test_index], train_dataY[test_index]
        model.fit(trainX, trainY)
        predictedY = model.predict(testX)
        model_i_mape = mape_function(predictedY, testY)
        sum_mape += model_i_mape
        y_pred.extend(predictedY)
        model_i_rmse = mean_squared_error(testY, model.predict(testX))
        sum_rmse += model_i_rmse
        rmse_all.append(model_i_rmse)
        # fitness_all.append(fitness_fuction(predictedY, testY))
        mape_all.append(model_i_mape)
    print("预测精度mape为", sum_mape / k, "标准差为", np.std(mape_all))
    print("预测精度rmse为", sum_rmse / k, "标准差为", np.std(rmse_all))
    true_pred = np.column_stack((train_dataY, y_pred))
    df1 = pd.DataFrame(true_pred, columns=["真实值", "预测值"])
    # df1.to_excel("micro_descriptors.xlsx")
    df1.to_excel("micro_descriptors.xlsx")


def DCSAtrue_pred(filePath, username):
    df = pd.read_excel(filePath + "cluster_TruePred_" + username + ".xlsx")
    sample_data = df.values
    print(sample_data)
    # print(sample_data)
    rmse = np.sqrt(mean_squared_error(sample_data[:, -2], sample_data[:,-1]))
    mape = mape_function(sample_data[:, -2], sample_data[:, -1])
    r2 = r2_score(sample_data[:, -2], sample_data[:, -1])
    df_predict = pd.DataFrame(np.array([rmse, mape, r2]), columns=["RMSE", "MAPE", "R2"])
    df_predict.to_excel(filePath + "DCSA_PredictedResults_" + username + ".xlsx", index=False)
    print(np.sqrt(mean_squared_error(sample_data[:, -2], sample_data[:,-1])))
    print(mape_function(sample_data[:, -2], sample_data[:, -1]))
    print(r2_score(sample_data[:, -2], sample_data[:, -1]))


def run_comparitive_experiments():
    #与DCSA相比较的6种模型SVR、GPR、RF、MLR、Lasso和Ridge
    df = pd.read_excel("data_files/多尺度样本_revision.xlsx")
    feature_name = [column for column in df][1:]
    print(feature_name)
    sample_data = df.values[:, 1:]
    print(sample_data)
    original_dataX = sample_data[:, :-1]
    train_dataX, train_dataY = pre_progressing(sample_data)
    k = 10  # or k=10
    kf = KFold(n_splits=k)
    model1, parameter1 = svr_model()
    model2, parameter2 = gaussian_model()
    model3, parameter3 = random_forest_model()
    # model4, parameter4 = linear_model()
    model4, parameter4 = lasso_regression()
    model5, parameter5 = ridge_model()
    # model7, parameter7 = network_model()
    predicted_models = [model1, model2, model3, model4, model5]
    model_name = ["SVR", "GPR", "RF", "Lasso", "Ridge"]

    model_rmse = []
    model_rmse_dev = []
    model_mape = []
    model_mape_dev = []
    model_r2 = []
    model_r2_dev = []
    model_index = 0
    for model_i in predicted_models:
        sum_rmse = 0
        sum_mape = 0
        sum_r2 = 0
        y_pred = []
        mape_all = []
        rmse_all = []
        r2_all = []
        for train_index, test_index in kf.split(train_dataX):
            trainX, trainY = train_dataX[train_index], train_dataY[train_index]
            testX, testY = train_dataX[test_index], train_dataY[test_index]
            model = model_i
            model.fit(trainX, trainY)
            # print(testY, model.predict(testX))
            predictedY = model.predict(testX)
            # predict_Y.append(model.predict(testX))
            model_i_mape = mape_function(predictedY, testY)
            sum_mape += model_i_mape
            model_i_r2 = r2_score(predictedY, testY)
            r2_all.append(model_i_r2)
            sum_r2 += model_i_r2
            y_pred.extend(predictedY)
            model_i_rmse = mean_squared_error(testY, model.predict(testX))
            rmse_all.append(model_i_rmse)
            # fitness_all.append(fitness_fuction(predictedY, testY))
            mape_all.append(model_i_mape)
            # cross_score +=   r2_score(testY, predictedY)
            sum_rmse += model_i_rmse
        rmse_dev = np.std(rmse_all)
        mape_dev = np.std(mape_all)
        r2_dev = np.std(r2_all)
        model_mape_dev.append(mape_dev)
        model_rmse_dev.append(rmse_dev)
        model_r2_dev.append(r2_dev)
        print(y_pred)
        print("预测精度mape为", sum_mape/k)
        print("预测精度rmse为", sum_rmse/k)
        model_mape.append(sum_mape/k)
        model_rmse.append(sum_rmse/k)
        model_r2.append(sum_r2/k)
        true_pred = np.column_stack((train_dataY, y_pred))
        predict_data = np.column_stack((original_dataX, true_pred))
        predict_data1 = np.column_stack((predict_data, np.exp(train_dataY)))
        predict_data2 = np.column_stack((predict_data1, np.exp(y_pred)))
        df1 = pd.DataFrame(predict_data2, columns=feature_name[:-1] + ["真实值", "预测值",
                                                                       "还原后的真实值", "还原后的预测值"])
        df1.to_excel("Comparative_results/" + model_name[model_index] + "_results.xlsx")
        model_index = model_index + 1
    model_results = np.column_stack((model_name, model_rmse))
    model_re1 = np.column_stack((model_results, model_rmse_dev))
    model_re2 = np.column_stack((model_re1, model_mape))
    model_re3 = np.column_stack((model_re2, model_mape_dev))
    model_re4 = np.column_stack((model_re3, model_r2))
    model_re5 = np.column_stack((model_re4, model_r2_dev))
    df2 = pd.DataFrame(model_re5, columns=["预测模型","rmse", "rmse_dev", "mape",
                                           "mape_dev", "r2", "r2_dev"])
    df2.to_excel("Comparative_results/Comparative_models_results.xlsx")


def cluster_visual(filePath, username):
    #画聚类散点图
    df = pd.read_excel(filePath+"cluster_split_data_"+username+".xlsx")
    data_array = df.values
    train_data, label = data_array[:, :-1], data_array[:, -1]
    cluster_number = list(set(label))
    return len(cluster_number)


def runDCSA(svr_paras, gpr_paras, rf_paras, lasso_paras, ridge_paras, filename, filePath, username):
    clusterNum = cluster_visual(filePath, username)
    fitness_cluster_model = []
    all_clusters_predicted = []
    df_1 = pd.read_excel(filePath + "clusterResult_" + username + ".xlsx",
                       sheetname="cluster_0")
    feature_name = [column for column in df_1][1:-1]
    print(feature_name)
    header = feature_name + ["true_value", "predicted_value"]
    #读第一个簇的数据
    df = pd.read_excel(filePath + "clusterResult_" + username + ".xlsx",
                       sheetname="cluster_0")
    feature_name = [column for column in df][1:]
    print(feature_name)
    cluster_data = df.values[:, 1:]  # 去除第一列
    print(cluster_data)
    original_dataX = cluster_data[:, :-1]
    print(original_dataX.shape)
    fileHostName = filename.split(".")[1]
    print(11, fileHostName)
    if (fileHostName == "csv"):
        df_original = pd.read_csv(filename)
    else:
        df_original = pd.read_excel(filename)
    original_data = df_original.values[:, :-1]
    scaler = MinMaxScaler()
    scaler.fit(original_data)  # 用全部266条数据的最大与最小值对每个簇上的特征进行最大最小归一化
    train_dataX = scaler.transform(original_dataX)
    print("归一化后的值", train_dataX)
    train_dataY = np.log(cluster_data[:, -1])
    k = 5  # or k=10
    kf = KFold(n_splits=k)
    model1, parameter1 = svr_model(svr_paras)
    model2, parameter2 = gaussian_model(gpr_paras)
    model3, parameter3 = random_forest_model(rf_paras)
    model4, parameter4 = lasso_regression(lasso_paras)
    model5, parameter5 = ridge_model(ridge_paras)
    predicted_models = [model1, model2, model3, model4, model5]
    model_name = ["SVR", "GPR", "RF", "Lasso", "Ridge"]
    model_fitness_1 = []
    true_pred_all = []
    for model_i in predicted_models:
        y_pred = []
        sum_fitness = 0
        sum_mape = 0
        fitness_all = []
        sum_fitness_rms = 0
        for train_index, test_index in kf.split(train_dataX):
            trainX, trainY = train_dataX[train_index], train_dataY[train_index]
            testX, testY = train_dataX[test_index], train_dataY[test_index]
            model = model_i
            model.fit(trainX, trainY)
            # print(testY, model.predict(testX))
            predictedY = model.predict(testX)
            # predict_Y.append(model.predict(testX))
            sum_fitness += fitness_fuction(predictedY, testY)
            sum_fitness_rms += fitness_fuction(predictedY, testY) ** 2
            model_i_mape = mape_function(predictedY, testY)
            sum_mape += model_i_mape
            y_pred.extend(predictedY)
            model_i_rmse = mean_squared_error(testY, model.predict(testX))
            fitness_all.append(fitness_fuction(predictedY, testY))
        print(y_pred)
        model_fitness_1.append(sum_fitness / k)
        true_pred = np.column_stack((train_dataY, y_pred))
        true_pred_all.append(true_pred)
    max_fitness_indice = model_fitness_1.index(max(model_fitness_1))
    cluster_row_fitness1 = ["cluster1"] + model_fitness_1+[model_name[max_fitness_indice]]

    true_pred_max = true_pred_all[max_fitness_indice]
    df1 = pd.DataFrame(np.column_stack((original_dataX, true_pred_max)), columns=header)
    fitness_cluster_model.append(cluster_row_fitness1)
    for index in range(1, clusterNum):

        df = pd.read_excel(filePath+"clusterResult_"+username+".xlsx",
                           sheetname="cluster_" + str(index))
        feature_name = [column for column in df][1:]
        print(feature_name)
        cluster_data = df.values[:, 1:]  # 去除第一列
        print(cluster_data)
        original_dataX = cluster_data[:, :-1]
        print(original_dataX.shape)
        m, n = original_dataX.shape
        fileHostName = filename.split(".")[1]
        print(11, fileHostName)
        # df_original = []
        if (fileHostName == "csv"):
            df_original = pd.read_csv(filename)
        else:
            df_original = pd.read_excel(filename)
        original_data = df_original.values[:, :-1]
        scaler = MinMaxScaler()
        scaler.fit(original_data)#用全部266条数据的最大与最小值对每个簇上的特征进行最大最小归一化
        train_dataX =scaler.transform(original_dataX)
        print("归一化后的值", train_dataX)
        #这里注意一下，不同数据集归一化方式不一样，这里为了能够演示平台，当输入数据为蠕变性能
        # （判断条件为特征数n等于27）数据时，决策属性取log来进行归一化；其他视情况而定，例如锂电池
        # 离子电导率数据，决策属性不进行归一化；将来这里需要看情况进行修改。
        if(n==27):
            train_dataY = np.log(cluster_data[:, -1])#
        else:
            train_dataY = cluster_data[:, -1]
        k = 5 # or k=10
        kf = KFold(n_splits=k)
        # model.fit(train_dataX, train_dataY)
        # print("ssddd", train_dataY, model.predict(train_dataX))
        model1, parameter1 = svr_model(svr_paras)
        model2, parameter2 = gaussian_model(gpr_paras)
        model3, parameter3 = random_forest_model(rf_paras)
        model4, parameter4 = lasso_regression(lasso_paras)
        model5, parameter5 = ridge_model(ridge_paras)
        predicted_models = [model1, model2, model3, model4, model5]
        model_fitness = []
        model_index = 0
        model_rmse = []
        model_rmse_dev = []  # std of rmse for each model
        model_fitness_dev = []  # std of fitness for each model
        model_mape = []
        model_mape_dev = []
        model_fitness_rms = []
        true_pred_all = []
        for model_i in predicted_models:
            predict_Y = []
            sum_rmse = 0
            # cross_score = 0
            y_pred = []
            sum_fitness = 0
            sum_mape = 0
            rmse_all = []
            fitness_all = []
            mape_all = []
            sum_fitness_rms = 0
            for train_index, test_index in kf.split(train_dataX):
                trainX, trainY = train_dataX[train_index], train_dataY[train_index]
                testX, testY = train_dataX[test_index], train_dataY[test_index]
                model = model_i
                model.fit(trainX, trainY)
                # print(testY, model.predict(testX))
                predictedY = model.predict(testX)
                # predict_Y.append(model.predict(testX))
                sum_fitness += fitness_fuction(predictedY, testY)
                sum_fitness_rms += fitness_fuction(predictedY, testY) ** 2
                model_i_mape = mape_function(predictedY, testY)
                sum_mape += model_i_mape
                y_pred.extend(predictedY)
                model_i_rmse = mean_squared_error(testY, model.predict(testX))
                rmse_all.append(model_i_rmse)
                fitness_all.append(fitness_fuction(predictedY, testY))
                mape_all.append(model_i_mape)
                # cross_score +=   r2_score(testY, predictedY)
                sum_rmse += model_i_rmse
            print(y_pred)
            # y_pred = np.array(y_pred)
            # train_dataY = np.array(train_dataY)
            print("预测精度rmse为", sum_rmse / k)
            print("预测精度fitness为", sum_fitness / k)
            print("预测精度mape为", sum_mape / k)
            fitness_rms = np.sqrt(sum_fitness_rms / k)
            print("预测精度fitness均方根", fitness_rms)
            model_fitness_rms.append(fitness_rms)
            model_fitness.append(sum_fitness / k)
            model_rmse.append(sum_rmse / k)
            model_mape.append(sum_mape / k)
            rmse_dev = np.std(rmse_all)
            fitness_dev = np.std(fitness_all)
            mape_dev = np.std(mape_all)
            model_mape_dev.append(mape_dev)
            model_rmse_dev.append(rmse_dev)
            model_fitness_dev.append(fitness_dev)
            true_pred = np.column_stack((train_dataY, y_pred))
            true_pred_all.append(true_pred)
            predict_data = np.column_stack((original_dataX, true_pred))
            predict_data1 = np.column_stack((predict_data, np.exp(train_dataY)))
            predict_data2 = np.column_stack((predict_data1, np.exp(y_pred)))
            # df1 = pd.DataFrame(predict_data2, columns=feature_name[:-1] + ["真实值", "预测值",
            #                                                                "还原后的真实值", "还原后的预测值"])
            # df1.to_excel("predicted_results/cluster_" + str(index + 1) + model_name[model_index] + "_results.xlsx")
            # model_index = model_index + 1
        max_fitness_indice = model_fitness.index(max(model_fitness))
        cluster_row_fitness = ["cluster" + str(index + 1)] + model_fitness+[model_name[max_fitness_indice]]
        true_pred_max = true_pred_all[max_fitness_indice]
        cluster_predict_data = np.column_stack((original_dataX, true_pred_max))
        df2_cluster_pred = pd.DataFrame(cluster_predict_data, columns=header)
        df1 = pd.concat([df1, df2_cluster_pred], axis=0)
        fitness_cluster_model.append(cluster_row_fitness)
    df_fitness = pd.DataFrame(fitness_cluster_model, columns=["model", "SVR", "GPR", "RF", "Lasso", "Ridge","SelectedModel"])
    df_fitness.to_excel(filePath + "cluster_Fitness_" + username + ".xlsx", index=False)
    df1.to_excel(filePath + "cluster_TruePred_" + username + ".xlsx", index=False)
    DCSAtrue_pred(filePath, username)


if __name__ == "__main__":
    # 平台传过来的参数：svr_paras, gpr_paras,
           #rf_paras, lasso_paras, ridge_paras
    # 文件名；生成文件绝对路径，用户名
    print('welcome to cluster world')
    parameterlist = []
    svr_paras = [sys.argv[1], sys.argv[2]]
    gpr_paras = [sys.argv[3], sys.argv[4]]
    rf_paras = [sys.argv[5], sys.argv[6]]
    lasso_paras = [sys.argv[7]]
    ridge_paras = [sys.argv[8]]
    for i in range(9, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)
    runDCSA(svr_paras, gpr_paras, rf_paras, lasso_paras, ridge_paras,
            parameterlist[0], parameterlist[1], parameterlist[2])
    # runDCSA()
    # run_comparitive_experiments()
    # true_pred()
    # run_validation_microdescriptor()
    # get_feature_importance()
