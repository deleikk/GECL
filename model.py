import torch
import torch.nn as nn
from utils import sparse_dropout, ZINBDecoder
import copy

class GECL(nn.Module,
           ):
    def __init__(self, adata, n_u, n_i, d, u_mul_s, v_mul_s, ut, vt, train_csr, adj_norm, l, temp, lambda_1, lambda_2,
                 dropout, device):
        # 用户数，物品数，嵌入维度，左右奇异对角矩阵，左右奇异值转置矩阵，训练集
        super(GECL, self).__init__()

        self.E_u_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_u, d)))
        self.E_i_0 = nn.Parameter(nn.init.xavier_uniform_(torch.empty(n_i, d)))
        self.adata = adata
        self.train_csr = train_csr
        self.adj_norm = adj_norm
        self.l = l
        self.E_u_list = [None] * (l + 1)
        self.E_i_list = [None] * (l + 1)
        self.E_u_list[0] = self.E_u_0
        self.E_i_list[0] = self.E_i_0
        self.Z_u_list = [None] * (l + 1)
        self.Z_i_list = [None] * (l + 1)
        self.G_u_list = [None] * (l + 1)
        self.G_i_list = [None] * (l + 1)
        # self.S_u_list = [None] * (l + 1)
        # self.S_i_list = [None] * (l + 1) #S 列表用于存储最好状态的初始嵌入
        self.G_u_list[0] = self.E_u_0
        self.G_i_list[0] = self.E_i_0
        self.temp = temp
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.act = nn.LeakyReLU(0.5)
        self.d = d
        self.E_u = None
        self.E_i = None
        self.u_mul_s = u_mul_s
        self.v_mul_s = v_mul_s
        self.ut = ut
        self.vt = vt

        #  self.linear=nn.Linear(64,2000)

        self.device = device
        self.decoder = ZINBDecoder(feats_dim=d)

    # def set_list(self,S_u_list,S_i_list):
    #     self.E_u_list = S_u_list
        # self.E_i_list = S_i_list

    def decode(self, dec_graph, u_norm, i_norm):
        mu, disp, pi, ge_score = self.decoder(dec_graph, u_norm, i_norm)
        return mu, disp, pi, ge_score

    def encode(self, adj_norm,flag = True):
        # if flag == True:
            # self.S_u_list = [t.detach().clone() for t in self.E_u_list]
            # self.S_i_list = [t.detach().clone() for t in self.E_i_list]
        for layer in range(1, self.l + 1):
            # GNN propagation
            # self.Z_u_list[layer] = (torch.spmm(sparse_dropout(self.adj_norm, self.dropout), self.E_i_list[layer - 1]))
            self.Z_u_list[layer] = (torch.spmm(sparse_dropout(adj_norm, self.dropout), self.E_i_list[layer - 1]))
            # 丢弃边缘以防止过拟合
            # self.Z_i_list[layer] = (
            # torch.spmm(sparse_dropout(self.adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1]))
            self.Z_i_list[layer] = (
                torch.spmm(sparse_dropout(adj_norm, self.dropout).transpose(0, 1), self.E_u_list[layer - 1]))
            # Z_u_list，Z_i_list原始图卷积嵌入
            # svd_adj propagation 首先将当前层的用户（或项目）节点特征矩阵与奇异值分解的结果进行乘积，得到更新后的特征矩阵。
            vt_ei = self.vt @ self.E_i_list[layer - 1]
            self.G_u_list[layer] = (self.u_mul_s @ vt_ei)
            # 重构的嵌入
            ut_eu = self.ut @ self.E_u_list[layer - 1]
            self.G_i_list[layer] = (self.v_mul_s @ ut_eu)
            # 使用先前的SVD结果(self.u_mul_s, self.v_mul_s, self.ut, self.vt)来更新嵌入

            # aggregate 在每一层的 GNN 传播和 SVD 传播后，会得到更新后的用户和项目的特征矩阵，存储在对应的列表中。
            self.E_u_list[layer] = self.Z_u_list[layer]
            self.E_i_list[layer] = self.Z_i_list[layer]
            # 把每层的更新合并，为了最终的嵌入表示

        # if flag:
        #     self.S_u_list = [t.detach().clone() for t in self.E_u_list]
        #     self.S_i_list = [t.detach().clone() for t in self.E_i_list]

        self.G_u = sum(self.G_u_list)  # 重构的最终的用户和项目的embedding矩阵 G_u 和 G_i
        self.G_i = sum(self.G_i_list)
        # 这段代码执行了模型的前向传播过程，包括了GNN和svd_adj的信息传播，以及信息的聚合操作。通过这些操作，模型学习到了用户和项目之间的交互信息，并生成了对应的embedding矩阵，用于后续的损失计算和反向传播。
        # aggregate across layers 每一层的用户和项目特征矩阵进行累加，以得到最终的用户和项目的 embedding 矩阵。 这些最终的 embedding 矩阵将用于后续的损失计算和模型训练过程中的反向传播。
        self.E_u = sum(self.E_u_list)

        self.E_i = sum(self.E_i_list)

        return self.G_u, self.G_i, self.E_u, self.E_i
        # return self.G_u, self.G_i, self.E_u, self.E_i,self.S_u_list, self.S_i_list

    # def forward(self, dec_graph, adj_norm=None):
    #     G_u_norm, G_i_norm, E_u_norm, E_i_norm,_,_ = self.encode(adj_norm)
    #
    #     return self.decode(dec_graph, G_u_norm, G_i_norm)