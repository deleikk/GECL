import warnings
import h5py
import numpy as np
import scanpy as sc
from utils import setup_seed,read_data
from GECL import run_GECL
from parser import args


warnings.filterwarnings('ignore')
#################数据读入及初步分析##################
setup_seed(0)
data_mat = h5py.File('data/'+args.data+'.h5')
X = np.array(data_mat['X']).astype(np.int_)
Y = np.array(data_mat['Y']).astype(np.int_).squeeze()
data_mat.close()

adata = sc.AnnData(X.astype('float'))
adata.obs['cl_type'] = Y
print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))

################## 读入h5ad数据#####################=======================================
# datamat = sc.read_h5ad(('data/rna_seq_baron.h5ad'))
# adata = datamat.X
# adata = sc.AnnData(adata.astype(float))
# Y = datamat.obs['cell_type1']
# Y = np.array(Y.array).squeeze()
# adata.X = adata.X.toarray()
# adata.obs['cl_type'] = Y
# print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
################################################################################################
# Adam
# mat, obs, var, uns = read_data(('data/' + args.data + '.h5'), sparsify=False,skip_exprs=False)
# X = np.array(mat.toarray())
# cell_name = np.array(obs["cell_type1"])
# cell_type, cell_label = np.unique(cell_name, return_inverse=True)
# Y = cell_label
# adata = sc.AnnData(X.astype('float'))
# adata.obs['cl_type'] = Y

##################读入csv数据#############
# 读取表达数据
# import pandas as pd
#
# expressions = pd.read_csv("data/Petropoulos/data.csv", header=1, index_col=0, sep=',')
# expressions = expressions.values.T  # 转置以匹配所需的格式
# expressions = np.array(expressions).astype(np.int_)
# adata = sc.AnnData(expressions.astype('float'))
#
# # # #====================读取标签数据
# labels_csv = pd.read_csv("data/Petropoulos/label.csv", header=0, index_col=0, sep=',')
# Y = labels_csv.iloc[:, 0].values  # 假设标签在第二列
# Y = np.array(Y).squeeze()
#
#
# adata.obs['cl_type']= Y
# print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
# print(adata)
# ########################################


#################预处理#######################

# Basic filtering
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.filter_cells(adata, min_genes=200)
# sc.pp.filter_genes(adata, min_counts=int(adata.shape[0] * 0.05))

adata.raw = adata.copy()

# Calculate the cell library size factor
sc.pp.normalize_per_cell(adata)
adata.obs['cs_factor'] = adata.obs.n_counts / np.median(adata.obs.n_counts)

# Log Normalization
sc.pp.log1p(adata)
# Calculate the gene size factor
adata.var['gs_factor'] = np.max(adata.X, axis=0, keepdims=True).reshape(-1)

sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=args.hvg_top)
adata = adata[:, adata.var.highly_variable]

print("Sparsity of after preprocessing: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
n_clusters = len(np.unique(Y))

print("number of cell type:{}".format(n_clusters))
print("数据集{}.".format(args.data))
print( 'lr',args.lr, 'lambda_1:', args.lambda1, 'lambda_2:', args.lambda2,
      'lambda_3:', args.lambda3, 'd:', args.d,  'batch_size:', args.inter_batch )
#################run#######################
adata ,record = run_GECL(adata,cl_type='cl_type',n_clusters=n_clusters,return_all=True)


print( 'lr:',args.lr, 'lambda_1:', args.lambda1, 'lambda_2:', args.lambda2,
      'lambda_3:', args.lambda3, 'd:', args.d,  'batch_size:', args.inter_batch,'HVG',args.hvg_top ,'svq_q:',args.q)

# Use cell embedding `feat` to perfrom Umap

#
# # 创建映射字典
# label_map = {
#     0: 'delta',
#     1: 'alpha',
#     2: 'activated_stellate',
#     3: 'ductal',
#     4: 'beta',
#     5: 'macrophage',
#     6: 'quiescent_stellate',
#     7: 'gamma',
#     8: 'endothelial',
#     9: 'epsilon',
#     10: 'schwann',
#     11: 'mast',
#     12: 'acinar',
#     13: 't_cell'
# }
#
# # 替换标签值
# adata.obs['cl_type'] = adata.obs['cl_type'].map(label_map)
#
# # 检查是否替换成功
# print(adata.obs['cl_type'].unique())
# adata.obs['cl_type'] = adata.obs['cl_type'].astype('category')



# sc.tl.umap(adata)
# adata.obs['cl_type'] = adata.obs['cl_type'].astype(str).astype('category')
# adata.obs['kmeans'] = adata.obs['kmeans'].astype(str).astype('category')
# adata.obs['louvain'] = adata.obs['louvain'].astype(str).astype('category')  ##是否可以改为用最好的那一轮去做可视化？
# sc.pl.umap(adata, color=['louvain', 'cl_type','kmeans'])
# sc.pl.umap(adata, color=['louvain'])
# sc.pl.umap(adata, color=[ 'cl_type'])
# sc.pl.umap(adata, color=['kmeans'])

