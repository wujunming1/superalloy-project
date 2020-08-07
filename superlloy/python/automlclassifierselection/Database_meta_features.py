# !/usr/bin/env python
# -*- coding:utf-8 -*-
# !/usr/bin/env python
# -*- coding:utf-8 -*-
import collections
import os
from collections import defaultdict
import scipy
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from numpy.compat import basestring
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import LocalOutlierFactor

import pymongo
import time
import sys
# from extraction import generate_excel
#


def traverse(f):
    global file
    fs = os.listdir(f)
    for f1 in fs:
        tmp_path = os.path.join(f, f1)
        if not os.path.isdir(tmp_path):
            file.append(tmp_path)
        else:
            traverse(tmp_path)

#categorical为1是非数值型类别属性
def transform(filename):
    categorical = []
    df = pd.read_excel(filename)
    df.dropna(inplace=True)
    # print(filename)
    # df = df.replace('?', np.nan)  # 去空缺值，只处理？ ? unkonwn的空缺值
    # df = df.replace('？', np.nan)
    # df = df.replace('unknown', np.nan)
    data = df.values
    data_transformed = df.values
    # 类别属性数值化
    lb = LabelEncoder()
    t = data_transformed.shape[1]
    for col in range(t):
        str_flag = 0
        for row in range(data_transformed.shape[0]):
            if isinstance(data_transformed[row][col], basestring):
                str_flag = 1
                break
        if str_flag == 1:
            data_transformed[:, col] = lb.fit_transform(data_transformed[:, col].astype(str))
            categorical.append(1)
        else:
            categorical.append(0)
    return data, data_transformed, categorical


def prob(data):  # 求各个类别的概率
    tar = list(data)
    Set = set(tar)
    class_probability = {}
    for item in Set:
        class_probability.update({item: tar.count(item) / data.shape[0]})
    return class_probability


def H(class_probability):  # 信息熵函数
    class_entropy = 0.0
    for prob in class_probability.values():
        class_entropy -= prob * np.log(prob)
    return class_entropy


def normH(data):
    n = data.shape[0]
    res = []
    for i in range(data.shape[1]):
        res.append(H(prob(data[:, i])) / np.log(n))
    return np.array(res)


def Hxy(C, X):  # 条件熵函数
    res = 0.0
    c_prob = prob(C)
    x_prob = prob(X)
    for i in c_prob.keys():
        for j in x_prob.keys():
            p_ij = X[(X == j) & (C == i)].shape[0] / X[X == j].shape[0]
            if p_ij == 0:
                res += 0
            else:
                res += x_prob[j] * p_ij * np.log(p_ij)
    return -res


def MI(data, target):  # 互信息函数
    H_c = H(prob(target))
    res = []
    for i in range(data.shape[1]):
        H_cx = Hxy(target, data[:, i])
        MI_cx = H_c - H_cx
        res.append(MI_cx)
    return np.array(res)


def MINMAX(data, target, i):
    ma = []
    for j in prob(target).keys():
        ma.append(np.max(data[:, i][target == j]))
    return np.min(ma)


def MAXMIN(data, target, i):
    mi = []
    for j in prob(target).keys():
        mi.append(np.min(data[:, i][target == j]))
    return np.max(mi)


def MINMIN(data, target, i):
    mi = []
    for j in prob(target).keys():
        mi.append(np.min(data[:, i][target == j]))
    return np.min(mi)


def MAXMAX(data, target, i):
    ma = []
    for j in prob(target).keys():
        ma.append(np.max(data[:, i][target == j]))
    return np.max(ma)


def F2(data, target):
    res = 1.0
    for i in range(data.shape[1]):
        temp = (MINMAX(data, target, i) - MAXMIN(data, target, i)) / (MAXMAX(data, target, i) - MINMIN(data, target, i))
        res *= temp
    return res


################################################################################
### Simple features



class Metafeatures:
    def __init__(self):
        self.metafeatures = {}

    def NumberOfInstances(self, X, y, categorical):
        return float(X.shape[0])

    def LogNumberOfInstances(self, X, y, categorical):
        return np.log(self.metafeatures["NumberOfInstances"])

    def NumberOfClasses(self, X, y, categorical):
        """
            Calculate the number of classes.

            Calls np.unique on the targets. If the dataset is a multilabel dataset,
            does this for each label seperately and returns the mean.
            """
        if len(y.shape) == 2:
            return np.mean([len(np.unique(y[:, i])) for i in range(y.shape[1])])
        else:
            return float(len(np.unique(y)))

    def NumberOfFeatures(self, X, y, categorical):
        return float(X.shape[1])

    def LogNumberOfFeatures(self, X, y, categorical):
        return np.log(self.metafeatures["NumberOfFeatures"])

    def NumberOfOutliers(self, X, y, categorical):
        clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
        clf.fit(X)
        X_scores = clf.negative_outlier_factor_
        X_scores_mean = X_scores.mean()
        return X[X_scores < X_scores_mean].shape[0]

    def PercentageOfInstancesWithOutliers(self, X, y, categorical):
        return float(self.metafeatures["NumberOfOutliers"] / self.metafeatures["NumberOfInstances"])

    # def MissingValues(self, X, y, categorical):
    #     missing = ~np.isfinite(X)
    #     return missing
    #
    # def NumberOfInstancesWithMissingValues(self, X, y, categorical):
    #     missing = self.metafeatures["MissingValues"]
    #     num_missing = missing.sum(axis=1)
    #     return float(np.sum([1 if num > 0 else 0 for num in num_missing]))
    #
    # def PercentageOfInstancesWithMissingValues(self, X, y, categorical):
    #     return float(self.metafeatures["NumberOfInstancesWithMissingValues"]
    #                  / float(self.metafeatures["NumberOfInstances"]))
    #
    # def NumberOfFeaturesWithMissingValues(self, X, y, categorical):
    #     missing = self.metafeatures["MissingValues"]
    #     num_missing = missing.sum(axis=0)
    #     return float(np.sum([1 if num > 0 else 0 for num in num_missing]))
    #
    # def PercentageOfFeaturesWithMissingValues(self, X, y, categorical):
    #     return float(self.metafeatures["NumberOfFeaturesWithMissingValues"] \
    #                  / float(self.metafeatures["NumberOfFeatures"]))
    #
    # def NumberOfMissingValues(self, X, y, categorical):
    #     return float(self.metafeatures["MissingValues"].sum())
    #
    # def PercentageOfMissingValues(self, X, y, categorical):
    #     return float(self.metafeatures["NumberOfMissingValues"]) / \
    #            float(X.shape[0] * X.shape[1])

    def NumberOfNumericFeatures(self, X, y, categorical):
        return len(categorical) - np.sum(categorical)

    def NumberOfCategoricalFeatures(self, X, y, categorical):
        return np.sum(categorical)

    def RatioNumericalToNominal(self, X, y, categorical):
        num_categorical = float(self.metafeatures["NumberOfCategoricalFeatures"])
        num_numerical = float(self.metafeatures["NumberOfNumericFeatures"])
        if num_categorical == 0.0:
            return 0.
        return num_numerical / num_categorical

    def RatioNominalToNumerical(self, X, y, categorical):
        num_categorical = float(self.metafeatures["NumberOfCategoricalFeatures"])
        num_numerical = float(self.metafeatures["NumberOfNumericFeatures"])
        if num_numerical == 0.0:
            return 0.
        else:
            return num_categorical / num_numerical

    # Number of attributes divided by number of samples
    def DatasetRatio(self, X, y, categorical):
        return float(self.metafeatures["NumberOfFeatures"]) / \
               float(self.metafeatures["NumberOfInstances"])

    def LogDatasetRatio(self, X, y, categorical):
        return np.log(self.metafeatures["DatasetRatio"])

    def InverseDatasetRatio(self, X, y, categorical):
        return float(self.metafeatures["NumberOfInstances"]) / \
               float(self.metafeatures["NumberOfFeatures"])

    def LogInverseDatasetRatio(self, X, y, categorical):
        return np.log(self.metafeatures["InverseDatasetRatio"])

    def ClassOccurences(self, X, y, categorical):
        if len(y.shape) == 2:
            occurences = []
            for i in range(y.shape[1]):
                occurences.append(self.ClassOccurences(X, y[:, i], categorical))
            return occurences
        else:
            occurence_dict = defaultdict(float)
            for value in y:
                occurence_dict[value] += 1
            return occurence_dict

    def ClassProbabilityMin(self, X, y, categorical):
        occurences = self.metafeatures["ClassOccurences"]

        min_value = np.iinfo(np.int64).max
        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                for num_occurences in occurences[i].values():
                    if num_occurences < min_value:
                        min_value = num_occurences
        else:
            for num_occurences in occurences.values():
                if num_occurences < min_value:
                    min_value = num_occurences
            return float(min_value) / float(y.shape[0])

    # aka default accuracy
    def ClassProbabilityMax(self, X, y, categorical):
        occurences = self.metafeatures["ClassOccurences"]
        max_value = -1

        if len(y.shape) == 2:
            for i in range(y.shape[1]):
                for num_occurences in occurences[i].values():
                    if num_occurences > max_value:
                        max_value = num_occurences
        else:
            for num_occurences in occurences.values():
                if num_occurences > max_value:
                    max_value = num_occurences
            return float(max_value) / float(y.shape[0])


    def ClassProbabilityMean(self, X, y, categorical):
        occurence_dict = self.metafeatures["ClassOccurences"]

        if len(y.shape) == 2:
            occurences = []
            for i in range(y.shape[1]):
                occurences.extend(
                    [occurrence for occurrence in occurence_dict[
                        i].values()])
            occurences = np.array(occurences)
        else:
            occurences = np.array([occurrence for occurrence in occurence_dict.values()],
                                  dtype=np.float64)
        return (occurences / y.shape[0]).mean()


    def ClassProbabilitySTD(self, X, y, categorical):
        occurence_dict = self.metafeatures["ClassOccurences"]

        if len(y.shape) == 2:
            stds = []
            for i in range(y.shape[1]):
                std = np.array(
                    [occurrence for occurrence in occurence_dict[
                        i].values()],
                    dtype=np.float64)
                std = (std / y.shape[0]).std()
                stds.append(std)
            return np.mean(stds)
        else:
            occurences = np.array([occurrence for occurrence in occurence_dict.values()],
                                  dtype=np.float64)
            return (occurences / y.shape[0]).std()


    def NumSymbols(self, X, y, categorical):
        symbols_per_column = []
        for i, column in enumerate(X.T):
            if categorical[i]:
                unique_values = np.unique(column)
                num_unique = np.sum(np.isfinite(unique_values))
                symbols_per_column.append(num_unique)
        return symbols_per_column


    def Kurtosisses(self, X, y, categorical):
        kurts = []
        for i in range(X.shape[1]):
            if not categorical[i]:
                kurts.append(scipy.stats.kurtosis(X[:, i]))
        return kurts


    def Skewness(self, X, y, categorical):
        skews = []
        for i in range(X.shape[1]):
            if not categorical[i]:
                skews.append(scipy.stats.skew(X[:, i]))
        return skews


    def PCA(self, X, y, categorical):
        import sklearn.decomposition
        pca = sklearn.decomposition.PCA(copy=True)
        rs = np.random.RandomState(42)
        indices = np.arange(X.shape[0])
        for i in range(10):
            try:
                rs.shuffle(indices)
                pca.fit(X[indices])
                return pca
            except scipy.linalg.LinAlgError as e:
                pass
        return None


    def SymbolsMin(self, X, y, categorical):
        minimum = None
        for unique in self.metafeatures["NumSymbols"]:
            if unique > 0 and (minimum is None or unique < minimum):
                minimum = unique
        return minimum if minimum is not None else 0


    def SymbolsMax(self, X, y, categorical):
        values = self.metafeatures["NumSymbols"]
        if len(values) == 0:
            return 0
        return max(max(values), 0)


    def SymbolsMean(self, X, y, categorical):
        # TODO: categorical attributes without a symbol don't count towards this
        # measure
        values = [val for val in self.metafeatures["NumSymbols"] if val > 0]
        mean = np.nanmean(values)
        return mean if np.isfinite(mean) else 0


    def SymbolsSTD(self, X, y, categorical):
        values = [val for val in self.metafeatures["NumSymbols"] if val > 0]
        std = np.nanstd(values)
        return std if np.isfinite(std) else 0


    def SymbolsSum(self, X, y, categorical):
        sum = np.nansum(self.metafeatures["NumSymbols"])
        return sum if np.isfinite(sum) else 0


    def KurtosisMin(self, X, y, categorical):
        kurts = self.metafeatures["Kurtosisses"]
        minimum = np.nanmin(kurts) if len(kurts) > 0 else 0
        return minimum if np.isfinite(minimum) else 0


    def KurtosisMax(self, X, y, categorical):
        kurts = self.metafeatures["Kurtosisses"]
        maximum = np.nanmax(kurts) if len(kurts) > 0 else 0
        return maximum if np.isfinite(maximum) else 0


    def KurtosisMean(self, X, y, categorical):
        kurts = self.metafeatures["Kurtosisses"]
        mean = np.nanmean(kurts) if len(kurts) > 0 else 0
        return mean if np.isfinite(mean) else 0


    def KurtosisSTD(self, X, y, categorical):
        kurts = self.metafeatures["Kurtosisses"]
        std = np.nanstd(kurts) if len(kurts) > 0 else 0
        return std if np.isfinite(std) else 0


    def SkewnessMin(self, X, y, categorical):
        skews = self.metafeatures["Skewness"]
        minimum = np.nanmin(skews) if len(skews) > 0 else 0
        return minimum if np.isfinite(minimum) else 0


    def SkewnessMax(self, X, y, categorical):
        skews = self.metafeatures["Skewness"]
        maximum = np.nanmax(skews) if len(skews) > 0 else 0
        return maximum if np.isfinite(maximum) else 0


    def SkewnessMean(self, X, y, categorical):
        skews = self.metafeatures["Skewness"]
        mean = np.nanmean(skews) if len(skews) > 0 else 0
        return mean if np.isfinite(mean) else 0


    def SkewnessSTD(self, X, y, categorical):
        skews = self.metafeatures["Skewness"]
        std = np.nanstd(skews) if len(skews) > 0 else 0
        return std if np.isfinite(std) else 0


    def ClassProbability(self, X, y, categorical):
        ClassProbability = self.metafeatures["ClassOccurences"]

        for key_occurences in ClassProbability.keys():
            ClassProbability[key_occurences] /= float(y.shape[0])
        # res = list(ClassProbability.values())
        return ClassProbability


    def OverlapVolumn(self, X, y, categorical):
        def MINMAX( data, target, i):
            ma = []
            for j in self.ClassProbability(data, target, i).keys():
                ma.append(np.max(data[:, i][target == j]))
            return np.min(ma)

        def MAXMIN( data, target, i):
            mi = []
            for j in self.ClassProbability(data, target, i).keys():
                mi.append(np.min(data[:, i][target == j]))
            return np.max(mi)

        def MINMIN( data, target, i):
            mi = []
            for j in self.ClassProbability(data, target, i).keys():
                mi.append(np.min(data[:, i][target == j]))
            return np.min(mi)

        def MAXMAX( data, target, i):
            ma = []
            for j in self.ClassProbability(data, target, i).keys():
                ma.append(np.max(data[:, i][target == j]))
            return np.max(ma)
        overlap_volumn = 1.0
        for i in range(X.shape[1]):
            temp = (MINMAX(X, y, i) - MAXMIN(X, y, i)) / (MAXMAX(X, y, i) - MINMIN(X, y, i))
            overlap_volumn *= temp
        return overlap_volumn


    def ClassEntropy(self, X, y, categorical):
        labels = 1 if len(y.shape) == 1 else y.shape[1]
        if labels == 1:
            y = y.reshape((-1, 1))

        entropies = []
        for i in range(labels):
            occurence_dict = collections.defaultdict(float)
            for value in y[:, i]:
                occurence_dict[value] += 1
            entropies.append(scipy.stats.entropy([occurence_dict[key] for key in
                                                  occurence_dict], base=2))
        return np.mean(entropies)

    def NormEntropy(self, X, y, categorical):
        n = X.shape[0]
        res = []
        for i in range(X.shape[1]):
            res.append(self.ClassEntropy(list(self.ClassProbability(X[:, i], y, categorical).values()) / np.log(n), y, categorical))
        return np.array(res)

    def ConditionalEntropy(self, X, y, categorical):  # 条件熵函数
        res = 0.0
        c_prob = self.ClassProbability(X, y, categorical)
        x_prob = self.ClassProbability(y, X, categorical)
        for i in c_prob.keys():
            for j in x_prob.keys():
                p_ij = X[(X == j) & (y == i)].shape[0] / X[X == j].shape[0]
                if p_ij == 0:
                    res += 0
                else:
                    res += x_prob[j] * p_ij * np.log(p_ij)
        return -res

    def MutualInformation(self, X, y, categorical):  # 互信息函数
        H_c = self.ClassEntropy(list(self.ClassProbability(X, y, categorical).values()), y, categorical)
        res = []
        for i in range(X.shape[1]):
            H_cx = self.ConditionalEntropy(y, X[:, i], categorical)
            MI_cx = H_c - H_cx
            res.append(MI_cx)
        return np.array(res)

    def NoiseSignalRatio(self, X, y, categorical):
        NormEntropy = self.NormEntropy(X, y, categorical)
        MutualInform = self.MutualInformation(X, y, categorical)
        noise_signal_ratio = (np.mean(NormEntropy * np.log(X.shape[0])) - np.mean(MutualInform)) \
                             / np.mean(MutualInform)
        return noise_signal_ratio

    def LandmarkLDA(self, X, y, categorical):
        import sklearn.discriminant_analysis
        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        try:
            for train, test in kf.split(X, y):
                lda = sklearn.discriminant_analysis.LinearDiscriminantAnalysis()

                if len(y.shape) == 1 or y.shape[1] == 1:
                    lda.fit(X[train], y[train])
                else:
                    lda = OneVsRestClassifier(lda)
                    lda.fit(X[train], y[train])

                predictions = lda.predict(X[test])
                accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
            return accuracy / 10
        except scipy.linalg.LinAlgError as e:
            print("LDA failed: %s Returned 0 instead!" % e)
            return np.NaN
        except ValueError as e:
            print("LDA failed: %s Returned 0 instead!" % e)
            return np.NaN

    def LandmarkNaiveBayes(self, X, y, categorical):
        import sklearn.naive_bayes

        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        for train, test in kf.split(X, y):
            nb = sklearn.naive_bayes.GaussianNB()

            if len(y.shape) == 1 or y.shape[1] == 1:
                nb.fit(X[train], y[train])
            else:
                nb = OneVsRestClassifier(nb)
                nb.fit(X[train], y[train])

            predictions = nb.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    def LandmarkDecisionTree(self, X, y, categorical):
        import sklearn.tree

        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            tree = sklearn.tree.DecisionTreeClassifier(random_state=random_state)

            if len(y.shape) == 1 or y.shape[1] == 1:
                tree.fit(X[train], y[train])
            else:
                tree = OneVsRestClassifier(tree)
                tree.fit(X[train], y[train])

            predictions = tree.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    def LandmarkDecisionNodeLearner(self, X, y, categorical):
        import sklearn.tree

        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy", max_depth=1, random_state=random_state,
                min_samples_split=2, min_samples_leaf=1, max_features=None)
            if len(y.shape) == 1 or y.shape[1] == 1:
                node.fit(X[train], y[train])
            else:
                node = OneVsRestClassifier(node)
                node.fit(X[train], y[train])
            predictions = node.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    def LandmarkRandomNodeLearner(self, X, y, categorical):
        import sklearn.tree

        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)
        accuracy = 0.

        for train, test in kf.split(X, y):
            random_state = sklearn.utils.check_random_state(42)
            node = sklearn.tree.DecisionTreeClassifier(
                criterion="entropy", max_depth=1, random_state=random_state,
                min_samples_split=2, min_samples_leaf=1, max_features=1)
            node.fit(X[train], y[train])
            predictions = node.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    def Landmark1NN(self, X, y, categorical):
        import sklearn.neighbors

        if len(y.shape) == 1 or y.shape[1] == 1:
            kf = sklearn.model_selection.StratifiedKFold(n_splits=10)
        else:
            kf = sklearn.model_selection.KFold(n_splits=10)

        accuracy = 0.
        for train, test in kf.split(X, y):
            kNN = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
            if len(y.shape) == 1 or y.shape[1] == 1:
                kNN.fit(X[train], y[train])
            else:
                kNN = OneVsRestClassifier(kNN)
                kNN.fit(X[train], y[train])
            predictions = kNN.predict(X[test])
            accuracy += sklearn.metrics.accuracy_score(predictions, y[test])
        return accuracy / 10

    def PCAFractionOfComponentsFor95PercentVariance(self, X, y, categorical):
        pca_ = self.metafeatures["PCA"]
        if pca_ is None:
            return np.NaN
        sum_ = 0.
        idx = 0
        while sum_ < 0.95 and idx < len(pca_.explained_variance_ratio_):
            sum_ += pca_.explained_variance_ratio_[idx]
            idx += 1
        return float(idx) / float(X.shape[1])

    def PCAKurtosisFirstPC(self, X, y, categorical):
        pca_ = self.metafeatures["PCA"]
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components

        kurtosis = scipy.stats.kurtosis(transformed)
        return kurtosis[0]

    def PCASkewnessFirstPC(self, X, y, categorical):
        pca_ = self.metafeatures["PCA"]
        if pca_ is None:
            return np.NaN
        components = pca_.components_
        pca_.components_ = components[:1]
        transformed = pca_.transform(X)
        pca_.components_ = components

        skewness = scipy.stats.skew(transformed)
        return skewness[0]


    def fit(self, filename):
        data, data_transformed, categorical = transform(filename)
        lb = LabelEncoder()
        t = data.shape[1]
        for col in range(t):
            str_flag = 0
            for row in range(data.shape[0]):
                if isinstance(data[row][col], basestring):
                    str_flag = 1
                    break
            if str_flag == 1:
                data[:, col] = lb.fit_transform(data[:, col].astype(str))
        X = data[:, :-1].astype('float64')
        y = data[:, -1].astype('int8')
        X_transformed = data_transformed[:, :-1].astype('float32')
        y_transformed = data_transformed[:, -1].astype('int8')
        self.metafeatures["NumberOfInstances"] = self.NumberOfInstances(X, y, categorical)
        self.metafeatures["LogNumberOfInstances"] = self.LogNumberOfInstances(X, y, categorical)
        self.metafeatures["NumberOfClasses"] = self.NumberOfClasses(X, y, categorical)
        self.metafeatures["NumberOfFeatures"] = self.NumberOfFeatures(X, y, categorical)
        self.metafeatures["LogNumberOfFeatures"] = self.LogNumberOfFeatures(X, y, categorical)
        self.metafeatures["NumberOfOutliers"] = self.NumberOfOutliers(X_transformed, y_transformed, categorical)
        self.metafeatures["PercentageOfInstancesWithOutliers"] = self.PercentageOfInstancesWithOutliers(X, y, categorical)
        self.metafeatures["NumberOfNumericFeatures"] = self.NumberOfNumericFeatures(X, y, categorical)
        self.metafeatures["NumberOfCategoricalFeatures"] = self.NumberOfCategoricalFeatures(X, y, categorical)
        self.metafeatures["RatioNumericalToNominal"] = self.RatioNumericalToNominal(X, y, categorical)
        self.metafeatures["RatioNominalToNumerical"] = self.RatioNominalToNumerical(X, y, categorical)
        self.metafeatures["DatasetRatio"] = self.DatasetRatio(X, y, categorical)
        self.metafeatures["LogDatasetRatio"] = self.LogDatasetRatio(X, y, categorical)
        self.metafeatures["InverseDatasetRatio"] = self.InverseDatasetRatio(X, y, categorical)
        self.metafeatures["LogInverseDatasetRatio"] = self.LogInverseDatasetRatio(X, y, categorical)
        self.metafeatures["ClassOccurences"] = self.ClassOccurences(X, y, categorical)
        self.metafeatures["ClassProbabilityMin"] = self.ClassProbabilityMin(X, y, categorical)
        self.metafeatures["ClassProbabilityMax"] = self.ClassProbabilityMax(X, y, categorical)
        self.metafeatures["ClassProbabilityMean"] = self.ClassProbabilityMean(X, y, categorical)
        self.metafeatures["ClassProbabilitySTD"] = self.ClassProbabilitySTD(X, y, categorical)
        self.metafeatures["NumSymbols"] = self.NumSymbols(X_transformed, y_transformed, categorical)
        self.metafeatures["Kurtosisses"] = self.Kurtosisses(X_transformed, y_transformed, categorical)
        self.metafeatures["Skewness"] = self.Skewness(X_transformed, y_transformed, categorical)
        self.metafeatures["PCA"] = self.PCA(X_transformed, y_transformed, categorical)
        self.metafeatures["SymbolsMin"] = self.SymbolsMin(X_transformed, y_transformed, categorical)
        self.metafeatures["SymbolsMax"] = self.SymbolsMax(X_transformed, y_transformed, categorical)
        self.metafeatures["SymbolsMean"] = self.SymbolsMean(X_transformed, y_transformed, categorical)
        self.metafeatures["SymbolsSTD"] = self.SymbolsSTD(X_transformed, y_transformed, categorical)
        self.metafeatures["SymbolsSum"] = self.SymbolsSum(X_transformed, y_transformed, categorical)
        self.metafeatures["KurtosisMin"] = self.KurtosisMin(X_transformed, y_transformed, categorical)
        self.metafeatures["KurtosisMax"] = self.KurtosisMax(X_transformed, y_transformed, categorical)
        self.metafeatures["KurtosisMean"] = self.KurtosisMean(X_transformed, y_transformed, categorical)
        self.metafeatures["KurtosisSTD"] = self.KurtosisSTD(X_transformed, y_transformed, categorical)
        self.metafeatures["SkewnessMin"] = self.SkewnessMin(X_transformed, y_transformed, categorical)
        self.metafeatures["SkewnessMax"] = self.SkewnessMax(X_transformed, y_transformed, categorical)
        self.metafeatures["SkewnessMean"] = self.SkewnessMean(X_transformed, y_transformed, categorical)
        self.metafeatures["SkewnessSTD"] = self.SkewnessSTD(X_transformed, y_transformed, categorical)
        self.metafeatures["ClassEntropy"] = self.ClassEntropy(X_transformed, y_transformed, categorical)
        self.metafeatures["NoiseSignalRatio"] = self.NoiseSignalRatio(X_transformed, y_transformed, categorical)
        self.metafeatures["OverlapVolumn"] = self.OverlapVolumn(X_transformed, y_transformed, categorical)
        self.metafeatures["LandmarkLDA"] = self.LandmarkLDA(X_transformed, y_transformed, categorical)
        self.metafeatures["LandmarkNaiveBayes"] = self.LandmarkNaiveBayes(X_transformed, y_transformed, categorical)
        self.metafeatures["LandmarkDecisionTree"] = self.LandmarkDecisionTree(X_transformed, y_transformed, categorical)
        self.metafeatures["LandmarkDecisionNodeLearner"] = self.LandmarkDecisionNodeLearner(X_transformed, y_transformed,
                                                                                  categorical)
        self.metafeatures["LandmarkRandomNodeLearner"] = self.LandmarkRandomNodeLearner(X_transformed, y_transformed, categorical)
        self.metafeatures["Landmark1NN"] = self.Landmark1NN(X_transformed, y_transformed, categorical)
        self.metafeatures["PCAFractionOfComponentsFor95PercentVariance"] = self.PCAFractionOfComponentsFor95PercentVariance(
            X_transformed, y_transformed, categorical)
        self.metafeatures["PCAKurtosisFirstPC"] = self.PCAKurtosisFirstPC(X_transformed, y_transformed, categorical)
        self.metafeatures["PCASkewnessFirstPC"] = self.PCASkewnessFirstPC(X_transformed, y_transformed, categorical)

#链接MongoDB数据库
def Connectdatabase():
    conn=pymongo.MongoClient(host="localhost",port=27017)
    db=conn.MongoDB_Data
    return db

def calculate(data_path):
    data = np.zeros(44)
    mf = Metafeatures()
    mf.fit(data_path)
    data[0] = mf.metafeatures["NumberOfInstances"]
    data[1] = mf.metafeatures["LogNumberOfInstances"]
    data[2] = mf.metafeatures["NumberOfClasses"]
    data[3] = mf.metafeatures["NumberOfFeatures"]
    data[4] = mf.metafeatures["LogNumberOfFeatures"]
    data[5] = mf.metafeatures["NumberOfOutliers"]
    data[6] = mf.metafeatures["PercentageOfInstancesWithOutliers"]
    data[7] = mf.metafeatures["NumberOfNumericFeatures"]
    data[8] = mf.metafeatures["NumberOfCategoricalFeatures"]
    data[9] = mf.metafeatures["RatioNumericalToNominal"]
    data[10] = mf.metafeatures["RatioNominalToNumerical"]
    data[11] = mf.metafeatures["DatasetRatio"]
    data[12] = mf.metafeatures["LogDatasetRatio"]
    data[13] = mf.metafeatures["InverseDatasetRatio"]
    data[14] = mf.metafeatures["LogInverseDatasetRatio"]
    data[15] = mf.metafeatures["ClassProbabilityMin"]
    data[16] = mf.metafeatures["ClassProbabilityMax"]
    data[17] = mf.metafeatures["ClassProbabilityMean"]
    data[18] = mf.metafeatures["ClassProbabilitySTD"]
    data[19] = mf.metafeatures["SymbolsMin"]
    data[20] = mf.metafeatures["SymbolsMax"]
    data[21] = mf.metafeatures["SymbolsMean"]
    data[22] = mf.metafeatures["SymbolsSTD"]
    data[23] = mf.metafeatures["SymbolsSum"]
    data[24] = mf.metafeatures["KurtosisMin"]
    data[25] = mf.metafeatures["KurtosisMax"]
    data[26] = mf.metafeatures["KurtosisMean"]
    data[27] = mf.metafeatures["KurtosisSTD"]
    data[28] = mf.metafeatures["SkewnessMin"]
    data[29] = mf.metafeatures["SkewnessMax"]
    data[30] = mf.metafeatures["SkewnessMean"]
    data[31] = mf.metafeatures["SkewnessSTD"]
    data[32] = mf.metafeatures["ClassEntropy"]
    data[33] = mf.metafeatures["NoiseSignalRatio"]
    data[34] = mf.metafeatures["OverlapVolumn"]
    data[35] = mf.metafeatures["LandmarkLDA"]
    data[36] = mf.metafeatures["LandmarkNaiveBayes"]
    data[37] = mf.metafeatures["LandmarkDecisionTree"]
    data[38] = mf.metafeatures["LandmarkDecisionNodeLearner"]
    data[39] = mf.metafeatures["LandmarkRandomNodeLearner"]
    data[40] = mf.metafeatures["Landmark1NN"]
    data[41] = mf.metafeatures["PCAFractionOfComponentsFor95PercentVariance"]
    data[42] = mf.metafeatures["PCAKurtosisFirstPC"]
    data[43] = mf.metafeatures["PCASkewnessFirstPC"]

    return data

def insert(data_path,file_name,user_name):
    mf_list = ['NumberOfInstances', 'LogNumberOfInstances', 'NumberOfFeatures',
               'LogNumberOfFeatures', 'NumberOfOutliers',
               'PercentageOfInstancesWithOutliers',
               'NumberOfNumericFeatures', 'NumberOfCategoricalFeatures', 'RatioNumericalToNominal',
               'RatioNominalToNumerical', 'DatasetRatio',
               'LogDatasetRatio', 'InverseDatasetRatio', 'LogInverseDatasetRatio','ClassProbabilityMin','ClassProbabilityMax',
               'ClassProbabilityMean','ClassProbabilitySTD'
               'SymbolsMin', 'SymbolsMax', 'SymbolsMean', 'SymbolsSTD', 'SymbolsSum',
               'KurtosisMin', 'KurtosisMax', 'KurtosisMean', 'KurtosisSTD', 'SkewnessMin', 'SkewnessMax',
               'SkewnessMean', 'SkewnessSTD', 'ClassEntropy', 'NoiseSignalRatio', 'OverlapVolumn',
               'LandmarkLDA', 'LandmarkNaiveBayes', 'LandmarkDecisionTree', 'LandmarkDecisionNodeLearner',
               'LandmarkRandomNodeLearner', 'Landmark1NN', 'PCAFractionOfComponentsFor95PercentVariance',
               'PCAKurtosisFirstPC', 'PCASkewnessFirstPC']
    meta_feature = calculate(data_path)

    date = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    file_name =file_name
    user_name = user_name

    df = pd.DataFrame(index=[file_name], columns=mf_list)
    for i in range(len(mf_list)):
        df.values[0][i] = meta_feature[i]

    df.to_csv('D:\\superlloy\\automl\\classifier\\selection\\meta-feature\\' +
                    file_name.split('.')[0] + '_' + 'MetaFeatures.csv')



    print('Insert DataBase:')
    db = Connectdatabase()
    db.MetaFeature.insert({
        "date": date,
        "mf_list": mf_list,
        "file_name": file_name,
        "user_name": user_name,
        "meta_feature": meta_feature.astype('object').tolist()
    })
    print('End Insert')


if __name__ == '__main__':

    # user_name = 'piki'
    # file_name = 'banknote-authentication.xls'
    # test_path = '/Users/buming/Documents/Super_Alloy/DataSet/datasets/train_done/cmc.xls'
    #
    # insert(test_path,file_name,user_name)

    #python .py file_name file_path user

    length_argv = len(sys.argv)
    print(length_argv)

    parameterlist = []
    for i in range(1, len(sys.argv)):
        para = sys.argv[i]
        parameterlist.append(para)
    print(parameterlist)

    insert(parameterlist[1], parameterlist[0], parameterlist[2])
    # insert(r'D:\gwhj\superlloy\data\piki\iris.xlsx', 'iris.xlsx', 'piki')
