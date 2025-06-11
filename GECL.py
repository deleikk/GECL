import time
import dgl
import numpy as np
import torch
import torch.utils.data as data
import gc
from metricetest import calculate_metric
from model import GECL
from parser import args
from utils import make_graph, create_coo_matrix, TrnData_test1, scipy_sparse_mat_to_torch_sparse_tensor, ZINBDecoder, \
    NBLoss, kmeans, louvain, EarlyStopping
from tqdm import tqdm

start_time = time.time()
device = 'cuda:' + args.cuda

# hyperparameters 参数设置
log_interval = args.log_interval
d = args.d
l = args.gnn_layer
temp = args.temp

epoch_no = args.epoch
max_samp = 40
lambda_1 = args.lambda1
lambda_2 = args.lambda2
lambda_3 = args.lambda3
lambda_4 = args.lambda4
dropout = args.dropout
lr = args.lr
decay = args.decay
svd_q = args.q
batch_size = args.inter_batch
patience = args.patience


def run_GECL(adata,
             cl_type=None,
             return_all: bool = False,
             n_clusters=None,
             gene_similarity: bool = False,
             log_interval: int = 5,
             use_rep: str = 'feat',
             verbose: bool = True,
             impute: bool = False,
             resolution: float = 1
             ):
    cell_type = adata.obs[cl_type].values if cl_type else None

    n_cells, n_genes = adata.shape
    # 构建二部图
    raw_exp = True

    print("making graph")
    adata, exp_value, enc_graph, dec_graph, unexp_edges = make_graph(adata, raw_exp, gene_similarity)
    enc_graph, dec_graph, exp_value = enc_graph.to(device), dec_graph.to(device), torch.tensor(exp_value, device=device)

    adataX = create_coo_matrix(enc_graph, ('cell', 'exp', 'gene'))  # 这里应该可以直接用dec_graph?
    adataX_csr = (adataX != 0).astype(np.float32)

    # 归一化二部图/邻接矩阵
    rowD = np.array(adataX.sum(1)).squeeze()
    colD = np.array(adataX.sum(0)).squeeze()
    for i in range(len(adataX.data)):
        adataX.data[i] = adataX.data[i] / pow(rowD[adataX.row[i]] * colD[adataX.col[i]], 0.5)
    # 对Train进行归一化之后值变成了不同的小数

    adataX = adataX.tocoo()
    adataX_data = TrnData_test1(adataX, exp_value)
    adataX_loader = data.DataLoader(adataX_data, batch_size=batch_size, shuffle=True, num_workers=0)

    adj_norm = scipy_sparse_mat_to_torch_sparse_tensor(
        adataX)  # 将train矩阵（假设它仍然是一个scipy的稀疏矩阵）转换为PyTorch的稀疏张量格式。PyTorch稀疏张量是为了高效地在PyTorch中处理稀疏矩阵而设计的
    adj_norm = adj_norm.coalesce().cuda(torch.device(device))
    print('Adj matrix normalized.')

    adj = scipy_sparse_mat_to_torch_sparse_tensor(adataX).coalesce().cuda(torch.device(device))
    print('Performing SVD...')
    svd_u, s, svd_v = torch.svd_lowrank(adj, q=svd_q)
    u_mul_s = svd_u @ (torch.diag(s))
    v_mul_s = svd_v @ (torch.diag(s))  # 将奇异值向量s 分别与左奇异值矩阵和右奇异值矩阵相乘 将其转化为对角矩阵
    del s
    del adj
    print('SVD done.')

    ###########模型初始化########

    model = GECL(adata, adj_norm.shape[0], adj_norm.shape[1], d, u_mul_s, v_mul_s, svd_u.T, svd_v.T,
                 adataX_csr, adj_norm, l, temp, lambda_1, lambda_2, dropout, device)

    model.cuda(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0, lr=lr)

    # ===================早停初始化=============
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    best_model_path = 'saved_model/best_model_{}.pth'.format(timestamp)

    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=0.001, path=best_model_path, stop_order='max')

    ##############记录值##################
    best_ari_k, best_ari_l = 0, 0
    best_nmi_k, best_nmi_l = 0, 0
    all_ari_k, all_ari_l = [], []
    all_nmi_k, all_nmi_l = [], []
    best_iter_k, best_iter_l = -1, -1
    loss_list = []
    loss_r_list = []
    loss_b_list = []
    loss_s_list = []
    ##########开始模型训练#############
    print(f"Start training on {device}...")
    for epoch in range(epoch_no):
        model.train()

        epoch_loss = 0
        epoch_loss_r = 0
        epoch_loss_s = 0
        epoch_loss_b = 0
        print("neg_sampling...")
        adataX_loader.dataset.neg_sampling()
        for i, batch in enumerate(tqdm(adataX_loader)):
            uids, pos, neg, pos_value = batch
            uids = uids.long().cuda(torch.device(device))
            pos = pos.long().cuda(torch.device(device))
            neg = neg.long().cuda(torch.device(device))
            iids = torch.concat([pos, neg], dim=0)
            pos_value = pos_value.to(device)
            optimizer.zero_grad()

            G_u_norm, G_i_norm, E_u_norm, E_i_norm = model.encode(adj_norm, flag=False)

            neg_score = torch.log(torch.exp(G_u_norm[uids] @ E_u_norm.T / temp).sum(1) + 1e-8).mean()
            neg_score += torch.log(torch.exp(G_i_norm[iids] @ E_i_norm.T / temp).sum(1) + 1e-8).mean()
            pos_score = (torch.clamp((G_u_norm[uids] * E_u_norm[uids]).sum(1) / temp, -5.0, 5.0)).mean() + (
                torch.clamp((G_i_norm[iids] * E_i_norm[iids]).sum(1) / temp, -5.0, 5.0)).mean()
            loss_s = -pos_score + neg_score
            loss_s = lambda_1 * loss_s

            # bpr
            u_emb = E_u_norm[uids]
            pos_emb = E_i_norm[pos]
            neg_emb = E_i_norm[neg]
            pos_scores = (u_emb * pos_emb).sum(-1)
            neg_scores = (u_emb * neg_emb).sum(-1)
            loss_b = -(pos_scores - neg_scores).sigmoid().log().mean()
            loss_b = lambda_2 * loss_b

            pos_edges = dec_graph.edge_ids(uids, pos, etype=('cell', 'exp', 'gene'))
            subgraph = dgl.edge_subgraph(dec_graph, pos_edges, preserve_nodes=True)
            pos_pre = model.decode(subgraph, E_u_norm, E_i_norm)  ###这里的嵌入传参是否应该也是要分批次的,
            loss_ZINB = NBLoss()
            loss_r = loss_ZINB(pos_pre[0], pos_pre[1], pos_pre[2], pos_value)
            loss_r = lambda_3 * loss_r

            # total loss
            loss = loss_s + loss_b + loss_r
            # loss = loss_b + loss_s

            loss.backward()
            optimizer.step()

            epoch_loss += loss.cpu().item()
            epoch_loss_r += loss_r.cpu().item()
            epoch_loss_b += loss_b.cpu().item()
            epoch_loss_s += loss_s.cpu().item()

            torch.cuda.empty_cache()
        # 计算平均损失
        batch_no = len(adataX_loader)
        epoch_loss = epoch_loss / batch_no
        epoch_loss_r = epoch_loss_r / batch_no
        epoch_loss_b = epoch_loss_b / batch_no
        epoch_loss_s = epoch_loss_s / batch_no

        loss_list.append(epoch_loss)
        loss_r_list.append(epoch_loss_r)
        loss_b_list.append(epoch_loss_b)
        loss_s_list.append(epoch_loss_s)
        if verbose or (epoch + 1) % log_interval == 0:
            print('Epoch:', epoch + 1, 'Loss:', epoch_loss, 'Loss_r:', epoch_loss_r, 'Loss_b:', epoch_loss_b, 'Loss_s:',
                  epoch_loss_s)

        if verbose or cell_type is not None and (epoch + 1) % (log_interval * 5) == 0:
            model.eval()
            with torch.no_grad():

                G_u_norm, G_i_norm, E_u_norm, E_i_norm = model.encode(adj_norm)
            model.train()

            adata.obsm['e0'] = model.E_u_0.data.cpu().numpy()  # Return initial cell embedding
            adata.obsm['feat'] = E_u_norm.cpu().detach().numpy()  # Return the weighted cell embeddings

            # kmeans
            adata = kmeans(adata, n_clusters,
                           use_rep=use_rep)  # 这里的use_rep表示feat,则函数中调用adata.obsm['feat'] 做了聚类，聚类的结果存储在adata.obs['kmeans']
            y_pred_k = np.array(adata.obs['kmeans'])

            # louvain
            adata = louvain(adata, resolution=resolution, use_rep=use_rep)
            y_pred_l = np.array(adata.obs['louvain'])
            print('Number of clusters identified by Louvain is {}'.format(len(np.unique(y_pred_l))))

            nmi_k, ari_k = calculate_metric(cell_type, y_pred_k)
            print('Clustering Kmeans %d:ARI= %.4f,NMI= %.4f,' % (epoch + 1, ari_k, nmi_k))

            nmi_l, ari_l = calculate_metric(cell_type, y_pred_l)
            print('Clustering Louvain %d: ARI= %.4f,NMI= %.4f' % (epoch + 1, ari_l, nmi_l))

            if ari_k > best_ari_k:
                best_ari_k = ari_k
                best_nmi_k = nmi_k
                best_iter_k = epoch + 1

            if ari_l > best_ari_l:
                best_ari_l = ari_l
                best_nmi_l = nmi_l
                best_iter_l = epoch + 1

            print(
                f'[END] For Kmeans, Best Iter : {best_iter_k} Best ARI : {best_ari_k:.4f}, Best NMI : {best_nmi_k:.4f}')
            print(
                f'[END] For Louvain, Best Iter : {best_iter_l} Best ARI : {best_ari_l:.4f}, Best NMI : {best_nmi_l:.4f}')
            ## End of training
            # ==============早停判断============#
            val_loss = ari_l
            early_stopping(val_loss, model)

            if early_stopping.early_stop:
                print("Early stopping...")
                break

    # 训练结束后加载最佳模型状态

    print("加载模型最好状态")
    model.load_state_dict(torch.load(best_model_path))
    # model = torch.load(best_model_path)
    model.eval()
    with torch.no_grad():

        G_u_norm, G_i_norm, E_u_norm, E_i_norm = model.encode(adj_norm, flag=False)

        adata.obsm['e0'] = model.E_u_0.data.cpu().numpy()  # Return initial cell embedding
        adata.obsm['feat'] = E_u_norm.cpu().detach().numpy()  # Return the weighted cell embeddings
        adata.varm['feat'] = E_i_norm.cpu().numpy()  # Return the final layer's gene embeddings

        # kmeans
        adata = kmeans(adata, n_clusters,
                       use_rep=use_rep)  # 这里的use_rep表示feat,则函数中调用adata.obsm['feat'] 做了聚类，聚类的结果存储在adata.obs['kmeans']
        y_pred_k = np.array(adata.obs['kmeans'])

        # louvain
        adata = louvain(adata, resolution=resolution, use_rep=use_rep)
        y_pred_l = np.array(adata.obs['louvain'])
        print('Number of clusters identified by Louvain is {}'.format(len(np.unique(y_pred_l))))

        nmi_k, ari_k = calculate_metric(cell_type, y_pred_k)
        print('Clustering Kmeans :ARI= %.4f,NMI= %.4f,' % (ari_k, nmi_k))

        nmi_l, ari_l = calculate_metric(cell_type, y_pred_l)
        print('Clustering Louvain: ARI= %.4f,NMI= %.4f' % (ari_l, nmi_l))
        # ==============================================================================================================

        if verbose and cl_type is not None:
            print(
                f'[END] For Kmeans, Best Iter : {best_iter_k} Best ARI : {best_ari_k:.4f}, Best NMI : {best_nmi_k:.4f}')
            print(
                f'[END] For Louvain, Best Iter : {best_iter_l} Best ARI : {best_ari_l:.4f}, Best NMI : {best_nmi_l:.4f}')
            # ====================
        all_ari_k.append(ari_k)
        all_ari_l.append(ari_l)
        all_nmi_k.append(nmi_k)
        all_nmi_l.append(nmi_l)

        record = None
        if return_all and cell_type is not None:
            print("记录值")

            record = {
                'all_loss': loss_list,
                'ari_k': all_ari_k,
                'ari_l': all_ari_l,
                'nmi_k': all_nmi_k,
                'nmi_l': all_nmi_l
            }

        #######################   Impute expression matrix (Optional) ########################
        if impute:
            print("正在执行插补...")
            # all_exp_cell, all_exp_gene = np.meshgrid(np.arange(n_cells), np.arange(n_genes))
            # all_exp_cell, all_exp_gene = all_exp_cell.reshape(-1), all_exp_gene.reshape(-1)
            #
            # all_dec_graph = dgl.heterograph({('cell', 'exp', 'gene'): (all_exp_cell, all_exp_gene)},
            #                                 num_nodes_dict={'cell': n_cells, 'gene': n_genes}).to(device)
            # all_dec_graph.nodes['cell'].data['cs_factor'] = dec_graph.nodes['cell'].data['cs_factor'].to(device)
            # all_dec_graph.nodes['gene'].data['gs_factor'] = dec_graph.nodes['gene'].data['gs_factor'].to(device)
            #
            # model.eval()
            #
            # with torch.no_grad():
            #     G_u_norm, G_i_norm, E_u_norm, E_i_norm = model.encode(adj_norm)  ##这里还是要把all_dec_graph放进去算嵌入
            #     mu, disp, pi, ge_score = model.decode(all_dec_graph, E_u_norm, E_i_norm)
            #
            # all_exp = mu.data.cpu().numpy().reshape(n_cells, n_genes)

            all_exp = G_u_norm @ G_i_norm.T
            all_exp = all_exp.cpu().numpy()
            adata.obsm['imputed'] = all_exp

            # 参考ALRA文章恢复生物零
            # ——ALRA Step 3: per-gene 0.1% quantile thresholding ——
            # p = 0.001
            # # axis=0 按 gene 维度求分位数，得到长度 n_genes 的阈值向量
            # thresh = np.quantile(all_exp, p, axis=0, method='lower')
            #
            # # 将所有低于该 gene 阈值的值置零
            # mask = all_exp < thresh  # (n_cells, n_genes) 的布尔矩阵
            # all_exp[mask] = 0.0

           # 进阶的：结合ZINB中的pi 来判断技术零
           #  pi_mat = pi.cpu().numpy().reshape(n_cells, n_genes)
           #  dropout_mask = pi_mat > 0.9  # 例如 π>0.9 的都当作本该为零
           #  all_exp[dropout_mask] = 0.0

            # adata.obsm['imputed'] = all_exp

        del model
        # del all_exp
        gc.collect()
        print("清理cuda缓存")
        torch.cuda.empty_cache()
        end_time = time.time()
        execution_time = end_time - start_time
        print("数据集{}.".format(args.data))
        print(f'执行时间：{execution_time / 60}分钟')
        if return_all:
            return adata, record
        # return adata
        return best_ari_l


# sc.tl.umap(adata)
# adata.obs['cl_type'] = adata.obs['cl_type'].astype(str).astype('category')
# adata.obs['kmeans'] = adata.obs['kmeans'].astype(str).astype('category')
# adata.obs['louvain'] = adata.obs['louvain'].astype(str).astype('category')  ##是否可以改为用最好的那一轮去做可视化？
# sc.pl.umap(adata, color=['louvain', 'cl_type','kmeans'])
# sc.pl.umap(adata, color=['louvain'])
# sc.pl.umap(adata, color=[ 'cl_type'])
# sc.pl.umap(adata, color=['kmeans'])