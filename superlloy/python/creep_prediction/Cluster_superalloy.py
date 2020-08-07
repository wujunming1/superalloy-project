import numpy as np
import pandas as pd
import sys
import xlrd
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def load_data(input_file):
    print(111111, input_file)
    fileHostName = input_file.split(".")[1]
    print(11, fileHostName)
    source_data = []
    if(fileHostName=="csv"):
        source_data = pd.read_csv(input_file)
    else:
        source_data = pd.read_excel(input_file)
    feature_name = [column for column in source_data][:-1]
    np_data = source_data.as_matrix()
    # print(source_data.columns)
    return source_data, np_data, feature_name


def data_process(np_data):
    from sklearn import preprocessing
    m, n = np_data.shape
    array_data = np.zeros(np_data.shape)
    for i in range(n-1):
        # array_data[:, i+1] = preprocessing.minmax_scale(np_data[:, i+1])
        array_data[:, i] = preprocessing.minmax_scale(np_data[:, i])
    # array_data[:, 23] = np.log(np_data[:, 23])
    # print(array_data.shape)
    return array_data


def cluster_split(array_data):
    print(array_data[1, :-1])
    return array_data[:, :-1]


def train_cluster(train_data, array_data, source_data, cluster_number, filePath, username, feature_name):
    from sklearn.cluster import KMeans, DBSCAN
    # model = DBSCAN()
    cluster_number = int(cluster_number)
    model = KMeans(n_clusters=cluster_number)
    model.fit(train_data)
    label = np.zeros((len(model.labels_), 1), dtype=int)
    for i in range(len(model.labels_)):
        label[i, 0] = int(model.labels_[i])
    # print(train_data, model.cluster_centers_)
    # r = pd.concat([source_data, pd.Series(model.labels_, index=source_data.index)], axis=1)

    # print(labels)
    combine = np.concatenate((array_data, label), axis=1)
    writer = pd.ExcelWriter(filePath+"clusterResult_"+username+".xlsx")
    cluster_data = pd.concat([pd.DataFrame(array_data[:, :-1]), pd.DataFrame(str(label) for label in model.labels_)], axis=1)
    # r0.columns = ['temp', 'stress','stacking','DL','G','L','Ni3Al', 'label']
    cluster_data.columns = feature_name + ['cluster_label']
    cluster_data.to_excel(filePath+"cluster_split_data_"+username+".xlsx", index=False)
    print(filePath+"cluster_split_data_"+username+".xlsx")
    print("我成功运行到这里了！")
    for i in range(len(np.unique(model.labels_))):
        cluster_subset = combine[combine[:, -1] == i][:, :-1]
        # print(np.arange(0, len(cluster_subset[:, 0])+1, 1).T)
        r0 = pd.DataFrame(np.arange(0, int(len(cluster_subset[:, 0])), 1).T)
        r1 = pd.DataFrame(cluster_subset)
        r = pd.concat([r0, r1], axis=1)
        r.columns  = ["Alloy"] + list(source_data.columns)
        r.to_excel(writer, sheet_name='cluster_'+str(i), index=False)
    # plot_cluster(train_data, model.labels_)
    writer.save()


def cluster_sample_number(clusterNum, filePath, username):
    #统计每个簇的样本数
    cluster_number = []
    clusters = []
    clusterNum = int(clusterNum)
    for index in range(0, clusterNum):
        clusters.append("cluster"+str(index+1))
        df = pd.read_excel(filePath+"clusterResult_"+username+".xlsx",
                           sheetname="cluster_" + str(index))
        cluster_data = df.values
        cluster_number.append(str(len(cluster_data)))
    print(cluster_number)
    cluster_sample_number = np.column_stack((clusters, cluster_number))
    df = pd.DataFrame(cluster_sample_number, columns=["clusters", "number"])
    df.to_excel(filePath+"cluster_sample_"+username+".xlsx", index=False)


def cluster_visual(filePath, username):
    #画聚类散点图
    df = pd.read_excel(filePath+"cluster_split_data_"+username+".xlsx")
    data_array = df.values
    train_data, label = data_array[:, :-1], data_array[:, -1]
    train_data = MinMaxScaler().fit_transform(train_data)
    pca = PCA(n_components=2)
    trans_data = pca.fit_transform(train_data)
    print(trans_data)
    print(pca.explained_variance_ratio_)
    new_data = np.column_stack((trans_data, label))
    print("data of transformation is", new_data)
    df = pd.DataFrame(new_data, columns=["PC1", "PC2", "label"])
    df.to_excel(filePath+"clusterVisualation_"+username+".xlsx", index=False)


def cluster_analysis(clusterNum, filePath, username):
    clusterNum = int(clusterNum)
    #统计每个簇的样本数
    df1 = pd.read_excel(filePath + "clusterResult_" + username + ".xlsx",
                        sheetname="cluster_0")['Re']
    df1 = pd.DataFrame(df1.values, columns=["cluster1"])
    for index in range(1, clusterNum):
        df = pd.read_excel(filePath + "clusterResult_" + username + ".xlsx",
                           sheetname="cluster_" + str(index))
        cluster_ReIndex = pd.DataFrame(df['Re'].values, columns=["cluster" + str(index + 1)])
        df1 = pd.concat([df1, cluster_ReIndex], axis=1)
    df1.to_excel(filePath + "cluster_Re_" + username + ".xlsx", index=False)
    df2 = pd.read_excel(filePath+"clusterResult_"+username+".xlsx",
                        sheetname="cluster_0")['T']
    df2 = pd.DataFrame(df2.values, columns=["cluster1"])
    for index in range(1, clusterNum):
        # clusters.append("cluster"+str(index+1))
        df = pd.read_excel(filePath+"clusterResult_"+username+".xlsx",
                           sheetname="cluster_" + str(index))
        cluster_TempIndex = pd.DataFrame(df['T'].values, columns=["cluster" + str(index + 1)])
        df2 = pd.concat([df2, cluster_TempIndex], axis=1)
    df2.to_excel(filePath+"cluster_T_"+username+".xlsx", index=False)
    df3 = pd.read_excel(filePath + "clusterResult_" + username + ".xlsx",
                        sheetname="cluster_0")['S']
    df3 = pd.DataFrame(df3.values, columns=["cluster1"])
    for index in range(1, clusterNum):
        # clusters.append("cluster"+str(index+1))
        df = pd.read_excel(filePath + "clusterResult_" + username + ".xlsx",
                           sheetname="cluster_" + str(index))
        cluster_TempIndex = pd.DataFrame(df['S'].values, columns=["cluster" + str(index + 1)])
        df3 = pd.concat([df3, cluster_TempIndex], axis=1)
    df3.to_excel(filePath + "cluster_S_" + username + ".xlsx", index=False)


def plot_cluster(data_zs, r):
    from sklearn.manifold import TSNE

    tsne = TSNE()
    tsne.fit_transform(data_zs)  # 进行数据降维,降成两维
    # a=tsne.fit_transform(data_zs) #a是一个array,a相当于下面的tsne_embedding_
    tsne = pd.DataFrame(tsne.embedding_)  # 转换数据格式

    import matplotlib.pyplot as plt

    d = tsne[r == 0]
    plt.plot(d[0], d[1], 'k.')

    d = tsne[r == 1]
    plt.plot(d[0], d[1], 'r.')

    d = tsne[r == 2]
    plt.plot(d[0], d[1], 'y.')
    d = tsne[r == 3]
    plt.plot(d[0], d[1], 'g.')
    d = tsne[r == 4]
    plt.plot(d[0], d[1], 'c.')

    d = tsne[r == 5]
    plt.plot(d[0], d[1], 'm.')
    d = tsne[r == 6]
    plt.plot(d[0], d[1], 'b.')
    d = tsne[r == 7]
    plt.plot(d[0], d[1], '#EE82EE',marker='.',linestyle='dotted')

    plt.show()


def run_cluster( cluster_number,filename,filePath, username):
    print("run_cluster")
    resource_data, np_data, feature_name = load_data(filename)
    array_data = data_process(np_data)
    train_data = cluster_split(array_data)
    # print(array_data, np_data)
    train_cluster(train_data, np_data, resource_data, cluster_number, filePath, username, feature_name)


if __name__ == "__main__":
    #平台传过来的参数：聚类个数；文件名；生成文件绝对路径，用户名
    print('welcome to cluster world')
    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)
    run_cluster(parameterlist[0], parameterlist[1], parameterlist[2], parameterlist[3])
    cluster_sample_number(parameterlist[0], parameterlist[2], parameterlist[3])
    cluster_visual(parameterlist[2], parameterlist[3])
    #cluster_analysis(parameterlist[0], parameterlist[2], parameterlist[3])

