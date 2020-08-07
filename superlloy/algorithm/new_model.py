import numpy as np
import pandas as pd
import sys, os


def load_data(input_file):
    # source_data = pd.read_excel(io=input_file, sheetname='cluster_1')
    source_data = pd.read_excel(io=input_file)
    np_data = source_data.as_matrix()
    # print(source_data.columns)
    return source_data, np_data


def data_process(array_data):
    from sklearn import preprocessing
    for i in range(20):
        array_data[:, i+1] = preprocessing.minmax_scale(array_data[:, i+1])
    array_data[:, 21] = array_data[:, 21]/100
    array_data[:, 22] = array_data[:, 22] / 100
    array_data[:, 23] = np.log(array_data[:, 23])
    # print(array_data.shape)


def test_write_excel(test):
    r1 = pd.DataFrame(test)
    r = pd.concat([r1])
    r.to_excel('test.xls')


def data_split(array_data, random_seed):
    from sklearn.model_selection import train_test_split
    train_x, test_x, train_y, test_y = train_test_split(array_data[:, :23], array_data[:, 23],
                                                        test_size=0.1, random_state=random_seed)
    return train_x, test_x, train_y, test_y


def gaussian_model():
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RationalQuadratic, WhiteKernel, RBF
    from sklearn.gaussian_process.kernels import ConstantKernel as C
    # parameter = "C(1, (1e-4, 10000)) * RationalQuadratic(alpha=0.01, length_scale_bounds=(1e-5, 20))"
    parameter = ' gaussian_model'
    # kernel_5 = C(1, (1e-4, 1)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.01, 2000))
    kernel = C(1, (0.01, 10)) * RationalQuadratic(alpha=0.1, length_scale_bounds=(0.1, 2000))
    model = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=10)
    return model, parameter


def linear_model():
    from sklearn.linear_model import ridge_regression
    parameter = ' Ridge_model'
    model = ridge_regression()
    return model, parameter


def svr_model():
    from sklearn.svm import SVR
    parameter = ' svr_model'
    model = SVR(kernel='rbf', C=10, gamma='auto')
    return model, parameter


def forest_model():
    from sklearn.ensemble import RandomForestRegressor
    parameter = ' RandomForestRegressor_model'
    model = RandomForestRegressor(n_estimators=15, max_depth=4, criterion='mae', bootstrap=False)
    return model, parameter


def train_model(train_x, test_x, train_y, test_y, select_model):
    if select_model == 0:
        model, parameter = forest_model()
    elif select_model ==1:
        model, parameter = gaussian_model()
    else:
        model, parameter = svr_model()
    model.fit(train_x, train_y)         
    predict_y = model.predict(test_x)   

    true_err = predict_y - test_y
    absolute_err = abs(true_err)
    print(parameter, 'test_y is : ', test_y, '\npredict_y is : ', predict_y, '\nerr_percent is: ',
          sum(absolute_err/test_y)/len(test_y))
    return predict_y, true_err, parameter


def plot_out(predict_y, test_y, title, save_path):
    import matplotlib.pyplot as plt
    # print(predict_y.shape)
    min_y = min(predict_y)
    max_y = max(predict_y)
    min_x = min(test_y)
    max_x = max(test_y)
    plt.ylim(ymax=max(max_x, max_y), ymin=min(min_x, min_y))
    plt.xlim(xmax=max(max_x, max_y), xmin=min(min_x, min_y))
    min_value = min(min_x, min_y)
    max_value = max(max_x, max_y)
    plt.scatter(test_y, predict_y, linewidths=True, vmax=1)
    plt.plot([min_value, max_value], [min_value, max_value], "--")
    plt.xlim(3, 9)
    plt.ylim(3, 9)
    plt.plot([3, 9], [3, 9], "--")
    plt.xlabel('real_life')
    plt.ylabel('predict_lift')
    plt.title(title)
    plt.savefig(os.path.join(save_path, 'new_model.jpg'))
    #plt.show()


# def write_excel(source_data, array_data, predict_matrix, parameter, category):
#     source_life = array_data[:, 23]
#     columns_1 = pd.DataFrame(array_data)
#     columns_2 = pd.DataFrame(predict_matrix)
#     average_life, average_err, average_err_percent = calculate_indicator(array_data[:, 23], predict_matrix)
#     columns_3 = pd.DataFrame(array_data[:, 23])
#     columns_4 = pd.DataFrame(average_life)
#     columns_5 = pd.DataFrame(average_err)
#     columns_6 = pd.DataFrame(average_err_percent)
#     tags = []
#     for i in range(len(predict_matrix[0, :])):
#         tags.append('times_' + str(i))
#     columns_all = pd.concat([columns_1, columns_2, columns_3, columns_4, columns_5, columns_6], axis=1)
#     columns_all.columns = list(source_data.columns) + list(tags) + ['source_life', 'average_life',
#                                                                     'average_err', 'average_err_percent']
#     print('the model err_percent is :' + str(sum(average_err_percent)/len(average_err_percent)))
#     columns_all.to_excel('C:\\Users\\15769\\Desktop\\result\\' + category + parameter + 'test.xls')
#     plot_out(average_life, source_life, parameter + 'all samples err_percent is: ' +
#              str(sum(average_err_percent) / len(average_err_percent)))


def write_plot(source_data, test_x, test_y, predict_y, path1):
    r1 = pd.DataFrame(test_x)
    r0 = pd.DataFrame(test_y)
    r2 = pd.DataFrame(predict_y)
    r3 = pd.DataFrame(predict_y - test_y)
    r4 = pd.DataFrame(abs(predict_y - test_y)/test_y)
    r = pd.concat([r1, r2, r3, r4], axis=1)
    r.columns = list(source_data.columns[1:24]) + ['predict_y', 'err', 'err_percent']
    r.to_excel(os.path.join(path1, 'test.xls'))
    plot_out(test_y, predict_y, 'title', path1)


# def calculate_indicator(source_life, predict_matrix):
#     row_num = (len(predict_matrix[:, 0]))
#     average_life = np.zeros((row_num, 1))      
#     average_err = np.zeros((row_num, 1))
#     average_err_percent = np.zeros((row_num, 1))
#
#     for i in range(row_num):
#         average_life[i] = sum([x for x in predict_matrix[i, :] if x > 0]) / (predict_matrix[i, :] > 0).sum()
#         average_err[i] = abs(source_life[i]) - abs(average_life[i])
#         average_err_percent[i] = abs(average_err[i])/abs(source_life[i])
#         # print(i)
#     # print(average_life.shape, average_err.shape, average_err_percent.shape)
#     print('avg_err is :' + str(np.average(average_err_percent)))
#     return average_life, average_err, average_err_percent
#
#
# def run():
#     source_data, array_data = load_data("C:\\Users\\15769\\Desktop\\result\\cluster_condition_3.xls")
#     data_process(array_data)

#     for select_model in range(3):
#         times_num = 99
#         predict_matrix = np.zeros((len(array_data[:, 0]), times_num))
#         for times in range(times_num):
#             print('\n\nthis is the round: ' + str(times))
#             train_x, test_x, train_y, test_y = data_split(array_data, times+100)
#             testX = test_x
#             testY = test_y
#             # testX = train_x
#             # testY = train_y
#             predict_y, one_err, parameter = train_model(train_x, testX, train_y, testY, select_model)
#             for k in range(len(testY)):
#                 location = testX[k, 0]
#                 # print(location, k, predict_y[k], predict_matrix.shape, times)
#                 predict_matrix[int(location-1), times] = predict_y[k]
#         write_excel(source_data, array_data, predict_matrix, parameter, 'cluster_3')
#

def run_one(file,path):
    source_data, array_data = load_data(file)
    data_process(array_data)


    train_x, test_x, train_y, test_y = data_split(array_data, 100)
    testX = test_x
    testY = test_y
    predict_y, one_err, parameter = train_model(train_x, testX, train_y, testY, 10)
    # write_excel(source_data, array_data, predict_y, parameter, 'cluster_3')
    write_plot(source_data, test_x, test_y, predict_y, path)


if __name__ == "__main__":
    path = 'E:\\project\\algorithm\\model'
    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)
    run_one(parameterlist[0],parameterlist[1])  #parameterlist[0]
