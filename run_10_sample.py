import warnings
from parser import args
import h5py
import numpy as np
import scanpy as sc
from GECL import run_GECL
from utils import setup_seed, sample


print("dataset:", args.data)
print('lr', args.lr, 'lambda_1:', args.lambda1, 'lambda_2:', args.lambda2,
      'lambda_3:', args.lambda3, 'd:', args.d, 'batch_size:', args.inter_batch, "dropout:", args.dropout)
def avg_score(list_of_scores):
    return sum(list_of_scores) / len(list_of_scores)


warnings.filterwarnings('ignore')

setup_seed(0)
# method = 'main5'

data_mat = h5py.File('data/' + args.data + '.h5')
X0 = np.array(data_mat['X'])
Y0 = np.array(data_mat['Y'])
X0 = np.ceil(X0).astype(np.int_)
Y0 = np.array(Y0).astype(np.int_).squeeze()

Final_ari_l, Final_nmi_l, N = [], [], []
times = 10

for t in range(times):
    print('----------------times: %d ----------------- ' % int(t + 1))
    adata = sc.AnnData(X0)
    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    ##sample
    seed = 10 * t
    X, Y = sample(X0, Y0, seed)
    adata = sc.AnnData(X.astype('float'))
    adata.obs['cl_type'] = Y
    n_clusters = len(np.unique(Y))

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

    print("Sparsity of after preprocessing: ",
          np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    n_clusters = len(np.unique(Y))

    print("number of cell type:{}".format(n_clusters))

    print("Sparsity: ", np.where(adata.X == 0)[0].shape[0] / (adata.X.shape[0] * adata.X.shape[1]))
    ###training
    adata, record = run_GECL(adata, cl_type='cl_type', n_clusters=n_clusters, return_all=True)
    # print(adata)

    final_ari_l, final_nmi_l = record['ari_l'][-1], record['nmi_l'][-1]
    n_pred = len(np.unique(np.array(adata.obs['louvain'])))
    N.append(n_pred)

    Final_ari_l.append(final_ari_l), Final_nmi_l.append(final_nmi_l)

## save results
# np.savez(os.path.join(dir0, "results/clustering/{}/result_{}_{}.npz".format(dataset, dataset, method)),
#          aril=Final_ari_l, nmil=Final_nmi_l)

avg_nmi_l = avg_score(Final_nmi_l)
avg_ari_l = avg_score(Final_ari_l)
print("dataset:", args.data)
print("Final_nmi_l", Final_nmi_l,"Average_Final_nmi_l:", avg_nmi_l)
print("Final_ari_l", Final_ari_l,"Average_Final_ari_l:", avg_ari_l)
print("Number of clusters identified by Louvain", N)

print('lr', args.lr, 'lambda_1:', args.lambda1, 'lambda_2:', args.lambda2,
      'lambda_3:', args.lambda3, 'd:', args.d, 'batch_size:', args.inter_batch, "dropout:", args.dropout)

# sc.pl.umap(adata, color=['louvain'])
# sc.pl.umap(adata, color=['cl_type'])



