import pandas as pd
import xlrd
import openpyxl
import sys
import numpy as np
def load_data(filename):
    file_list = filename.strip().split(".")
    data_frame=None
    if file_list[-1] == "xlsx" or file_list[-1] == "xls":
            data_frame = pd.read_excel(filename)
    elif file_list[-1] == "csv":
            data_frame = pd.read_csv(filename)
    else:
        print("this is vasp/clf file")
    dataset = data_frame.as_matrix()
    print(dataset)
    return data_frame
if __name__=="__main__":
    try:
        #print("hello world!")
        parameterlist=[];
        for i in range(1, len(sys.argv)):
            para=sys.argv[i]
            parameterlist.append(para)
        print(parameterlist)
        df=load_data(parameterlist[0])
        #获取excel表的列字段
        colums_names=[colum for colum in df]
        #print(colums_names)
        n_rows,n_cols=df.as_matrix().shape
        #print("样本行数为%d,列数为%d"%(n_rows,n_cols))
        geo_average=[]
        for i in range(len(colums_names)):
            col_i=df.iloc[:,i]
            # print(col_i)
            product=1.0
            for j in col_i:
                product=product*j
            geo_average.append(pow(product,1/n_rows))
            # print("每一列的几何平均数为：",pow(product,1/n_rows))
        #print("每一列的几何平均数为",dict(zip(colums_names,geo_average)))
        #返回每一列的偏度
        skew_col=df.skew(axis=0)
        #print("实际每一列的偏度",skew_col)
        #返回每一列值的平均数
        col_aver=df.mean(axis=0)
        #print("每一列的平均值",col_aver)
        col_aver=np.asarray(col_aver,dtype=np.float)
        #print(col_aver)
        aver_value=np.tile(col_aver,[n_rows,1])
        dataset=np.asarray(df,dtype=np.float)
        cons_k=n_rows/((n_rows-1)*(n_rows-2))
        #print("常数k",cons_k)
        #print((dataset-aver_value).shape)
        std=np.asarray(df.std(axis=0),dtype=np.float)
        std=np.tile(std,(n_rows,1))
        # colum_skew=np.sum(pow((dataset-aver_value)/std,3),axis=0)*cons_k
        colum_skew=np.sum(((dataset-aver_value)/std)**3,axis=0)*cons_k
        #print("计算的偏度为",colum_skew)
        for colum in df:
            c_colcount=df[colum].value_counts()
        #     print("每一列中各个值出现的次数统计",c_colcount)
        # print("每一列的众数")
        #对每一列数据进行统计，包括计数，均值，std，各个分位数等。
        description=df.describe()
        #print("数据信息：",description)
        #print(type(description))
        des=pd.DataFrame(description)
        range=des.loc["max"]-des.loc["min"]
        #print("每一列的极差",range);
        #print("每一列的标准差", des.loc["std"]);
        #print("每一列的四分位数Q1",des.loc["25%"])
        #print("每一列的四分位数Q2", des.loc["50%"])
        #print("每一列的四分位数Q3", des.loc["75%"])
        distance_4=des.loc["75%"]-des.loc["25%"]
        #print("四分位间距",distance_4)
        upper_limit=des.loc["75%"]+1.5*distance_4
        #print("每一列的上限",upper_limit)
        lower_limit=des.loc["25%"]-1.5*distance_4
        index_names=["skew","range","geo_average"]
        # loc可以对没有的index进行赋值，而 iloc 则不允许，iloc只能对已经存在的位置进行操作。
        des.loc["skew"]=skew_col
        des.loc["range"]=range
        des.loc["geo_average"]=geo_average
        des.loc["distance_4"]=distance_4
        des.loc["upper_limit"]=upper_limit
        des.loc["lower_limit"]=lower_limit
        print(des)
        #将数据汇总和统计结果写入到excel表中
        # des.append(skew_col,ignore_index=True)
        # print(des)
        des.to_excel("D:\\superlloy\\dataQuality\\"+parameterlist[1]+".xlsx")
    except Exception as e:
        print(e)


