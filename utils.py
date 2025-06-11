import torch.utils.data as data
import os
import random
import dgl
import pandas as pd
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.functional import softplus
import scanpy as sc
from sklearn.cluster import KMeans
import scipy.sparse as sp
import copy

def metrics(uids, predictions, topk, test_labels):
    user_num = 0
    all_recall = 0
    all_ndcg = 0
    for i in range(len(uids)):
        uid = uids[i]
        prediction = list(predictions[i][:topk])
        label = test_labels[uid]
        if len(label)>0:
            hit = 0
            idcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(topk, len(label)))])
            dcg = 0
            for item in label:
                if item in prediction:
                    hit+=1
                    loc = prediction.index(item)
                    dcg = dcg + np.reciprocal(np.log2(loc+2))
            all_recall = all_recall + hit/len(label)
            all_ndcg = all_ndcg + dcg/idcg
            user_num+=1
    return all_recall/user_num, all_ndcg/user_num


def preprocess(adata,n_top, filter_min_counts=True, size_factors=True, normalize_input=False, logtrans_input=True,hvg_top = True):
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_cells=3)
        sc.pp.filter_cells(adata, min_genes=200)

    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['cs_factor'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['cs_factor'] = 1.0

    if logtrans_input:
        sc.pp.log1p(adata)

    gs_factor = np.max(adata.X, axis=0, keepdims=True)
    adata.var['gs_factor'] = gs_factor.reshape(-1)

    if normalize_input:
        sc.pp.scale(adata)

    if hvg_top:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_top)
        adata = adata[:, adata.var.highly_variable]

    return adata


def scipy_sparse_mat_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def sparse_dropout(mat, dropout):
    if dropout == 0.0:
        return mat
    indices = mat.indices()
    values = nn.functional.dropout(mat.values(), p=dropout)
    size = mat.size()
    return torch.sparse.FloatTensor(indices, values, size)

def spmm(sp, emb, device):
    sp = sp.coalesce()
    cols = sp.indices()[1]
    rows = sp.indices()[0]
    col_segs =  emb[cols] * torch.unsqueeze(sp.values(),dim=1)
    result = torch.zeros((sp.shape[0],emb.shape[1])).cuda(torch.device(device))
    result.index_add_(0, rows, col_segs)
    return result

def setup_seed(seed):          ##scBIG中用于删选数据的
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def sample(x, label, seed):
    x_sample = pd.DataFrame()
    for i in range(len(np.unique(label)) + 1):
        data = x[label == i,]
        data = pd.DataFrame(data)
        data = data.sample(frac=0.95, replace=False, weights=None, random_state=seed, axis=0)
        data['label'] = i
        x_sample = x_sample.append(data, ignore_index=True)

    y = np.asarray(x_sample['label'], dtype='int')
    x_sample = np.asarray(x_sample.iloc[:, :-1])
    return x_sample, y


def add_degree(graph, edge_types):
    def _calc_norm(x):
        x = x.numpy().astype('float32')
        x[x == 0.] = np.inf
        x = torch.FloatTensor(1. / np.sqrt(x))
        return x.unsqueeze(1)

    cell_ci, gene_ci = _calc_norm(graph['reverse-exp'].in_degrees()), _calc_norm(graph['exp'].in_degrees())
    cell_cj, gene_cj = _calc_norm(graph['exp'].out_degrees()), _calc_norm(graph['reverse-exp'].out_degrees())
    graph.nodes['cell'].data.update({'ci': cell_ci, 'cj': cell_cj})
    graph.nodes['gene'].data.update({'ci': gene_ci, 'cj': gene_cj})

    if 'co-exp' in edge_types:
        gene_cii, gene_cjj = _calc_norm(graph['co-exp'].in_degrees()), _calc_norm(graph['co-exp'].out_degrees())
        graph.nodes['gene'].data.update({'cii': gene_cii, 'cjj': gene_cjj})

def make_graph(adata, raw_exp=False, gene_similarity=False,):
    X = adata.X
    num_cells, num_genes = X.shape

    # Make expressioin/train graph
    num_nodes_dict = {'cell': num_cells, 'gene': num_genes}    #记录细胞和基因数
    exp_train_cell, exp_train_gene = np.where(X > 0)               # 记录边表达
    unexp_edges = np.where(X == 0)                                  # 非表达边

    # expression edges
    exp_edge_dict = {
        ('cell', 'exp', 'gene'): (exp_train_cell, exp_train_gene),                #细胞到基因的表达边
        ('gene', 'reverse-exp', 'cell'): (exp_train_gene, exp_train_cell)           #基因到细胞的表达边
    }

    # coexp_edges, uncoexp_edges = None, None
    # if gene_similarity:
    #     coexp_edges, uncoexp_edges = construct_gene_graph(X)            #调用construct_gene_graph方法计算基因之间的共表达边，并存储与下面的字典中
    #    exp_edge_dict[('gene', 'co-exp', 'gene')] = coexp_edges

    # expression encoder/decoder graph 这里的二部图是通过exp_edge_dict这个字典来构建的
    enc_graph = dgl.heterograph(exp_edge_dict, num_nodes_dict=num_nodes_dict)   #编码器图
    ###这里的编码图和解码图有什么区别，二者的作用分别是什么？？？？？？？
    exp_edge_dict.pop(('gene', 'reverse-exp', 'cell'))
    ##在解码器图中去除了反向表达边
    dec_graph = dgl.heterograph(exp_edge_dict, num_nodes_dict=num_nodes_dict)    #解码器图



    # add degree to cell/gene nodes
    add_degree(enc_graph, ['exp'] + (['co-exp'] if gene_similarity else []))      #为图中的节点添加度信息

    # If use ZINB decoder, add size factor to cell/gene nodes
    if raw_exp:
        Raw = pd.DataFrame(adata.raw.X, index=list(adata.raw.obs_names), columns=list(adata.raw.var_names))
        X = Raw[list(adata.var_names)].values
        exp_value = X[exp_train_cell, exp_train_gene].reshape(-1, 1)
        dec_graph.nodes['cell'].data['cs_factor'] = torch.Tensor(adata.obs['cs_factor']).reshape(-1, 1)
        dec_graph.nodes['gene'].data['gs_factor'] = torch.Tensor(adata.var['gs_factor']).reshape(-1, 1)

    else:
        ## Deflate the edge values of the bipartite graph to between 0 and 1 调整边的权重
        X = X / adata.var['gs_factor'].values
        exp_value = X[exp_train_cell, exp_train_gene].reshape(-1, 1)
    #边权重（exp_value）是通过细胞和基因之间的实际表达量计算而来的。通过将每个基因的表达值除以其最大值来规范化，然后提取特定细胞和基因对的表达值。
    # 这些权重反映了特定细胞如何“表达”特定基因，是量化细胞与基因之间交互强度的一种方式、、、
    #
    return adata, exp_value, enc_graph, dec_graph, unexp_edges

def make_graph_test(adata, raw_exp=False, gene_similarity=False,):
    X = adata.X
    num_cells, num_genes = X.shape

    # Make expressioin/train graph
    num_nodes_dict = {'cell': num_cells, 'gene': num_genes}    #记录细胞和基因数
    exp_train_cell, exp_train_gene = np.where(X > 0)               # 记录边表达
    unexp_edges = np.where(X == 0)                                  # 非表达边

    # expression edges
    exp_edge_dict = {
        ('cell', 'exp', 'gene'): (exp_train_cell, exp_train_gene),                #细胞到基因的表达边
        ('gene', 'reverse-exp', 'cell'): (exp_train_gene, exp_train_cell)           #基因到细胞的表达边
    }

    # coexp_edges, uncoexp_edges = None, None
    # if gene_similarity:
    #     coexp_edges, uncoexp_edges = construct_gene_graph(X)            #调用construct_gene_graph方法计算基因之间的共表达边，并存储与下面的字典中
    #    exp_edge_dict[('gene', 'co-exp', 'gene')] = coexp_edges

    # expression encoder/decoder graph 这里的二部图是通过exp_edge_dict这个字典来构建的
    enc_graph = dgl.heterograph(exp_edge_dict, num_nodes_dict=num_nodes_dict)   #编码器图
    ###这里的编码图和解码图有什么区别，二者的作用分别是什么？？？？？？？
    exp_edge_dict.pop(('gene', 'reverse-exp', 'cell'))
    ##在解码器图中去除了反向表达边
    dec_graph = dgl.heterograph(exp_edge_dict, num_nodes_dict=num_nodes_dict)    #解码器图



    # add degree to cell/gene nodes
    add_degree(enc_graph, ['exp'] + (['co-exp'] if gene_similarity else []))      #为图中的节点添加度信息

    # If use ZINB decoder, add size factor to cell/gene nodes
    if raw_exp:
        Raw = pd.DataFrame(adata.raw.X, index=list(adata.raw.obs_names), columns=list(adata.raw.var_names))
        X = Raw[list(adata.var_names)].values
        exp_value = X[exp_train_cell, exp_train_gene].reshape(-1, 1)
        dec_graph.nodes['cell'].data['cs_factor'] = torch.Tensor(adata.obs['cs_factor']).reshape(-1, 1)
        dec_graph.nodes['gene'].data['gs_factor'] = torch.Tensor(adata.var['gs_factor']).reshape(-1, 1)

    else:
        ## Deflate the edge values of the bipartite graph to between 0 and 1 调整边的权重
        X = X / adata.var['gs_factor'].values
        exp_value = X[exp_train_cell, exp_train_gene].reshape(-1, 1)
    #边权重（exp_value）是通过细胞和基因之间的实际表达量计算而来的。通过将每个基因的表达值除以其最大值来规范化，然后提取特定细胞和基因对的表达值。
    # 这些权重反映了特定细胞如何“表达”特定基因，是量化细胞与基因之间交互强度的一种方式、、、
    #
    return adata, exp_value, enc_graph, dec_graph, unexp_edges,exp_train_cell, exp_train_gene

###将二部图转换为coo格式存储####
def create_coo_matrix(graph, edge_type):
    # 提取特定类型边的源节点和目标节点索引
    src, dst = graph.edges(etype=edge_type)

    # 确保src和dst索引在CPU上
    if src.is_cuda:
        src = src.cpu()
    if dst.is_cuda:
        dst = dst.cpu()

    # 创建一个权重列表，这里简单地使用1来表示边的存在
    data = np.ones(len(src))

    # 提取节点数量
    num_src_nodes = graph.number_of_nodes('cell')  # 源节点数量
    num_dst_nodes = graph.number_of_nodes('gene')  # 目标节点数量

    # 创建COO矩阵
    coo_mat = sp.coo_matrix((data, (src.numpy(), dst.numpy())), shape=(num_src_nodes, num_dst_nodes))

    return coo_mat


class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.data = coomat.data
        self.dokmat = coomat.todok()  #这句话改变了坐标索引的值
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    # 这段代码是负采样（Negative
    # Sampling）的实现部分。在这段代码中，对于每一个用户项目对（u, i），通过随机选择负样本（i_neg）来训练模型。
    # 具体流程如下：
    # 对于数据中的每一个用户（u），循环进行以下操作。
    # 从可能的项目集合中随机选择一个负样本（i_neg）。
    # 检查（u, i_neg）是否在已有的正样本集合（self.dokmat）中，如果不在，则认为找到了一个合适的负样本，跳出循环；如果在，则需要重新选择负样本，直到找到一个不在正样本集合中的负样本。
    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx],self.data[idx]

class TrnData_test(data.Dataset):
    def __init__(self, exp_train_cell, exp_train_gene, exp_value, X):
        self.exp_train_cell = exp_train_cell
        self.exp_train_gene = exp_train_gene
        self.exp_value = exp_value
        self.n_pos_edges = len(exp_value)
        self.X = X  # 基因表达矩阵，稀疏矩阵格式
        self.num_genes = X.shape[1]

    def __len__(self):
        return self.n_pos_edges

    def __getitem__(self, idx):
        uid = self.exp_train_cell[idx]
        pos_gene = self.exp_train_gene[idx]
        pos_val = self.exp_value[idx]

        # 使用拒绝采样随机选择一个未表达的基因作为负样本
        while True:
            neg_gene = np.random.randint(self.num_genes)
            if self.X[uid, neg_gene] == 0:
                break

        return uid, pos_gene, neg_gene, pos_val

class TrnData_test1(data.Dataset):
    def __init__(self, coomat, exp_value):
        self.rows = coomat.row
        self.cols = coomat.col
        self.data = coomat.data
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)
        self.exp_value = exp_value

    def neg_sampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                i_neg = np.random.randint(self.dokmat.shape[1])
                if (u, i_neg) not in self.dokmat:
                    break
            self.negs[i] = i_neg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx], self.exp_value[idx]

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x) - 1., min=1e-5, max=1e6)


class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)


class ZINBDecoder(nn.Module):
    def __init__(self, feats_dim, gene_similarity=False):
        super().__init__()
        """ZINB decoder for link prediction
        predict link existence (not edge type)  
        """
        self.dec_mean = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_disp = nn.Linear(feats_dim, 1)
        self.dec_disp_act = DispAct()
        self.dec_pi = nn.Sequential(nn.Linear(feats_dim, 1), nn.Sigmoid())
        self.dec_mean_act = MeanAct()
        self.gene_similarity = gene_similarity

    def forward(self, graph, c_feat, g_feat, ckey='cell', gkey='gene'):
        """
        Paramters
        ---------
        graph : dgl.homograph
        c_feat : torch.FloatTensor
            cell features
        g_feat : torch.FloatTensor
            gene features
        g_last : torch.FloatTensor
            gene features of the last laye
        ckey, gkey : str
            target node types

        Returns
        -------
        mu : torch.FloatTensor
            the estimated mean of ZINB model shape : (n_edges, 1)
        disp : torch.FloatTensor
            the estimated dispersion of ZINB model shape : (n_edges, 1)
        pi : torch.FloatTensor
            the estimated dropout rate of ZINB model shape : (n_edges, 1)
        ge_score : torch.FloatTensor
            the predicted values of highly correlated gene edges when considering gene massage
        """
        ge_score = None

        with graph.local_scope():
            graph.nodes[ckey].data['h'], graph.nodes[gkey].data['h'] = c_feat, g_feat
            graph.nodes[ckey].data['one'] = torch.ones([c_feat.shape[0], 1], device=c_feat.device)
            graph.nodes[gkey].data['one'] = torch.ones([g_feat.shape[0], 1], device=g_feat.device)

            exp_graph = graph['cell', 'exp', 'gene'] if self.gene_similarity else graph

            exp_graph.apply_edges(fn.u_mul_v('h', 'h', 'h_d'))
            exp_graph.apply_edges(fn.u_mul_v('one', 'gs_factor', 'gs_factor'))
            exp_graph.apply_edges(fn.u_mul_v('cs_factor', 'one', 'cs_factor'))

            h_d = exp_graph.edata['h_d']
            mu_ = self.dec_mean(h_d)
            disp_ = self.dec_disp(h_d)
            pi = self.dec_pi(h_d)

            disp = self.dec_disp_act(exp_graph.edata['gs_factor'] * disp_)
            mu_ = exp_graph.edata['gs_factor'] * mu_
            mu = exp_graph.edata['cs_factor'] * self.dec_mean_act(mu_)

        return mu, disp, pi, ge_score


class NBLoss(nn.Module):
    def __init__(self):
        super(NBLoss, self).__init__()

    def forward(self, x, mean, disp, scale_factor=1.0):
        eps = 1e-10
        scale_factor = scale_factor[:, None]
        mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        result = t1 + t2

        result = torch.mean(result)
        return result


class ZINBLoss(nn.Module):
    def __init__(self):
        super().__init__()

    ###师姐发的ZINB，返回值为None
    def forward(self, mean, disp, pi, x=None, ridge_lambda=0.0,eps = 1e-10):
        # if x is None:
        #     zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        #     zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        #     return zero_case.mean()
        # scale_factor = scale_factor[:, None]
        # mean = mean * scale_factor

        t1 = torch.lgamma(disp + eps) + torch.lgamma(x + 1.0) - torch.lgamma(x + disp + eps)
        t2 = (disp + x) * torch.log(1.0 + (mean / (disp + eps))) + (x * (torch.log(disp + eps) - torch.log(mean + eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0 - pi + eps)
        zero_nb = torch.pow(disp / (disp + mean + eps), disp)
        zero_case = -torch.log(pi + ((1.0 - pi) * zero_nb) + eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)

        if ridge_lambda > 0:
            ridge = ridge_lambda * torch.square(pi)
            result += ridge

        result = torch.mean(result)
        return result

################################
class GaussianNoise(nn.Module):
    def __init__(self, sigma=0):
        super(GaussianNoise, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training:
            x = x + self.sigma * torch.randn_like(x)
        return x


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


def get_pos_value(adata, exp_value, uids, pos):
    # 假设 exp_train_cell 和 exp_train_gene 是通过 np.where(X > 0) 得到的
    exp_train_cell, exp_train_gene = np.where(adata.X > 0)
    # 它们分别对应非零表达值在细胞和基因上的索引

    # 将 uids 和 pos 转换为 numpy 数组以便与 exp_train_cell 和 exp_train_gene 进行比较
    uids_np = uids.cpu().numpy()
    pos_np = pos.cpu().numpy()

    # 初始化一个存储 pos_value 的张量
    pos_value = torch.zeros_like(uids, dtype=torch.float32)

    # 遍历每一个 (uid, pos) 对
    for i in range(len(uids_np)):
        # 找到对应的非零表达值在 exp_value 中的位置
        # 通过找到 exp_train_cell 中对应 uid 且 exp_train_gene 中对应 pos 的位置
        match_idx = np.where((exp_train_cell == uids_np[i]) & (exp_train_gene == pos_np[i]))[0]

        if len(match_idx) > 0:
            # 提取对应的表达值
            pos_value[i] = exp_value[match_idx[0]]

    return pos_value

# def sample(x, label, seed):
#     x_sample = pd.DataFrame()
#     for i in range(len(np.unique(label)) + 1):
#         data = x[label == i,]
#         data = pd.DataFrame(data)
#         data = data.sample(frac=0.95, replace=False, weights=None, random_state=seed, axis=0)
#         data['label'] = i
#         x_sample = x_sample.append(data, ignore_index=True)
#
#     y = np.asarray(x_sample['label'], dtype='int')
#     x_sample = np.asarray(x_sample.iloc[:, :-1])
#     return x_sample, y

def sample(x, label, seed):
    x_sample = pd.DataFrame()
    for i in range(len(np.unique(label)) + 1):
        data = x[label == i,]
        data = pd.DataFrame(data)
        data = data.sample(frac=0.8, replace=False, weights=None, random_state=seed, axis=0)
        data['label'] = i
        # x_sample = x_sample.append(data, ignore_index=True)
        x_sample = pd.concat([x_sample, data], ignore_index=True)

    y = np.asarray(x_sample['label'], dtype='int')
    x_sample = np.asarray(x_sample.iloc[:, :-1])
    return x_sample, y

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, stop_order='max'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.stop_order = stop_order
        self.val_loss_max = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.S_u_list= [None] * 3
        self.S_i_list= [None] * 3

    def __call__(self, val_loss, model):

        score = val_loss
        if self.stop_order == 'max':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
        elif self.stop_order == 'min':
            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score > self.best_score + self.delta:
                self.counter += 1
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Updation changed ({self.val_loss_max:.6f} --> {val_loss:.6f}).  Saving model ...')
        # torch.save(model, self.path)
        torch.save(model.state_dict(), self.path)

        self.val_loss_max = val_loss



class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def in_ipynb():  # pragma: no cover
    try:
        # noinspection PyUnresolvedReferences
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter




def empty_safe(fn, dtype):
    def _fn(x):
        if x.size:
            return fn(x)
        return x.astype(dtype)

    return _fn

decode = empty_safe(np.vectorize(lambda _x: _x.decode("utf-8")), str)


def read_clean(data):
    assert isinstance(data, np.ndarray)
    if data.dtype.type is np.bytes_:
        data = decode(data)
    if data.size == 1:
        data = data.flat[0]
    return data

def dict_from_group(group):
    assert isinstance(group, h5py.Group)
    d = dotdict()
    for key in group:
        if isinstance(group[key], h5py.Group):
            value = dict_from_group(group[key])
        else:
            value = read_clean(group[key][...])
        d[key] = value
    return d

import h5py
from scipy.sparse import csr_matrix


def read_data(filename, sparsify=False, skip_exprs=False):
    with h5py.File(filename, "r") as f:
        obs = pd.DataFrame(dict_from_group(f["obs"]), index=decode(f["obs_names"][...]))
        var = pd.DataFrame(dict_from_group(f["var"]), index=decode(f["var_names"][...]))
        uns = dict_from_group(f["uns"])
        if not skip_exprs:
            exprs_handle = f["exprs"]
            if isinstance(exprs_handle, h5py.Group):
                mat = csr_matrix((exprs_handle["data"][...], exprs_handle["indices"][...],
                                            exprs_handle["indptr"][...]), shape=exprs_handle["shape"][...])
            else:
                mat = exprs_handle[...].astype(np.float32)
                if sparsify:
                    mat = csr_matrix(mat)
        else:
            mat = csr_matrix((obs.shape[0], var.shape[0]))
    return mat, obs, var, uns