import scanpy as sc
import scipy
from scipy.stats import pearsonr
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
import pandas as pd


def pearsonr_error(y, h):
    res = []
    if len(y.shape) < 2:
        y = y.reshape((1, -1))
        h = h.reshape((1, -1))

    for i in range(y.shape[0]):
        res.append(pearsonr(y[i], h[i])[0])
    return np.mean(res)


def cosine_similarity_score(y, h):
    if len(y.shape) < 2:
        y = y.reshape((1, -1))
        h = h.reshape((1, -1))
    cos = cosine_similarity(y, h)
    res = []
    for i in range(len(cos)):
        res.append(cos[i][i])
    return np.mean(res)


def kmeans(adata, n_clusters, use_rep=None):
    k_means = KMeans(n_clusters, n_init=20, random_state=0)
    y_pred = k_means.fit_predict(adata.obsm[use_rep])
    adata.obs['kmeans'] = y_pred
    adata.obs['kmeans'] = adata.obs['kmeans'].astype(str).astype('category')
    return adata


def louvain(adata, resolution=None, use_rep=None):
    sc.pp.neighbors(adata, use_rep=use_rep)
    sc.tl.louvain(adata, resolution=resolution)
    return adata


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.float64).astype(np.int64)
    y_pred = y_pred.astype(np.float64).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    return w[row_ind, col_ind].sum() * 1.0 / y_pred.size


def calculate_metric(pred, label):
    nmi = np.round(metrics.normalized_mutual_info_score(label, pred), 4)
    ari = np.round(metrics.adjusted_rand_score(label, pred), 4)
    return nmi, ari

#=============以下为yc发的三个插补指标
def impute_dropout(X, seed=1, rate=0.1):                  #用于模拟dropout现象
    """
    X: original testing set
    ========
    returns:
    X_zero: copy of X with zeros
    i, j, ix: indices of where dropout is applied
    首先检测X是稠密矩阵还是稀疏矩阵，并创建一个相同的副本X_zero。
    找出所有非零元素的索引。
    根据设定的rate随机选择部分非零元素并将它们设置为0。
    """

    # If the input is a dense matrix  判断是稠密还是稀疏矩阵
    if isinstance(X, np.ndarray):
        X_zero = np.copy(X)     #创建副本
        # select non-zero subset
        i, j = np.nonzero(X_zero)   #非零元素的索引
    # If the input is a sparse matrix
    else:
        X_zero = scipy.sparse.lil_matrix.copy(X)
        # select non-zero subset
        i, j = X_zero.nonzero()

    np.random.seed(seed)
    # changes here:随机选择 rate 百分比的非零元素索引，将它们的值设置为 0，模拟 dropout
    # choice number 1 : select 10 percent of the non zero values (so that distributions overlap enough)
    ix = np.random.choice(range(len(i)), int(
        np.floor(rate * len(i))), replace=False)
    # X_zero[i[ix], j[ix]] *= np.random.binomial(1, rate)
    X_zero[i[ix], j[ix]] = 0.0

    # choice number 2, focus on a few but corrupt binomially
    # ix = np.random.choice(range(len(i)), int(slice_prop * np.floor(len(i))), replace=False)
    # X_zero[i[ix], j[ix]] = np.random.binomial(X_zero[i[ix], j[ix]].astype(np.int), rate)
    return X_zero, i, j, ix


def imputation_error(X_mean, X,  i, j, ix):  #X_zero,
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset, does not need
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    median L1 distance between datasets at indices given
    """

    # If the input is a dense matrix
    if isinstance(X, np.ndarray):
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]      # 从X_mean和X中取出对应的索引值赋给x和y
        result = np.abs(x - y)                      # 计算x和y的绝对差
    # If the input is a sparse matrix
    else:
        all_index = i[ix], j[ix]
        x = X_mean[all_index[0], all_index[1]]
        y = X[all_index[0], all_index[1]]
        yuse = scipy.sparse.lil_matrix.todense(y)
        yuse = np.asarray(yuse).reshape(-1)
        result = np.abs(x - yuse)     #原理同上

    # 计算均方根误差
    rmse = sqrt(np.mean(np.square(y - x)))
    # return np.median(np.abs(x - yuse))
    # 返回结果：均值、中位数、最小值、最大值、均方根误差
    return np.mean(result), np.median(result), np.min(result), np.max(result), rmse

def imputation_cosine(X_mean, X, i, j, ix):  # X_zero,
    """
    X_mean: imputed dataset
    X: original dataset
    X_zero: zeros dataset, does not need
    i, j, ix: indices of where dropout was applied
    ========
    returns:
    cosine similarity between datasets at indices given
    """

    # If the input is a dense matrix
    if isinstance(X, np.ndarray):
        all_index = i[ix], j[ix]
        x, y = X_mean[all_index], X[all_index]
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        print(x)
        print(y)
        result = cosine_similarity(x, y)
    # If the input is a sparse matrix
    else:
        all_index = i[ix], j[ix]
        x = X_mean[all_index[0], all_index[1]]
        y = X[all_index[0], all_index[1]]
        yuse = scipy.sparse.lil_matrix.todense(y)
        yuse = np.asarray(yuse).reshape(-1)
        x = x.reshape(1, -1)
        yuse = yuse.reshape(1, -1)
        result = cosine_similarity(x, yuse)
    # return np.median(np.abs(x - yuse))
    return result[0][0]


# imputed_file = r'D:\Code\GraphSCI\zeisel_a5.csv'
# original_file = r"D:\Code\two1\data\zeisel\zeisel_count.csv"                    #是经过dropout处理的吗？
#
# X = pd.read_csv(original_file, header=0,index_col=0,sep=',')
# X = np.array(X).T
# X = reshapeX(X,k=3000 ,cou=1)  ####过滤高变基因，用我自己的就可以
#
# # "======================难道是用impute_dropout返回值存储为CSV文件？"
# i = pd.read_csv(r"D:\Code\two1\dropout\zeisel\i05.csv", header=0,index_col=0,sep=',')  #####这三个csv文件是怎么得来的？？？？？？？？？？？？？？
# i = np.array(i).reshape(i.shape[0], )
# j = pd.read_csv(r"D:\Code\two1\dropout\zeisel\j05.csv", header=0,index_col=0,sep=',')
# j = np.array(j).reshape(j.shape[0], )
# ix = pd.read_csv(r"D:\Code\two1\dropout\zeisel\ix05.csv", header=0,index_col=0,sep=',')
# ix = np.array(ix).reshape(ix.shape[0], )
#
#
#
#
# data_imputed = pd.read_csv(imputed_file, header=0, index_col=0, sep=',')
# X_imputed = np.array(data_imputed)#.T
#
# mean,median,min,max, rmse = imputation_error(X_imputed,X,i,j,ix)
# cosine_sim = imputation_cosine(X_imputed,X, i,j,ix)
#
# print("median:",median)
# print("cosine_sim:",cosine_sim)
# print("rmse:",rmse)