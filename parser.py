import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Model Params')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')#1e-3
    parser.add_argument('--decay', default=0.99, type=float, help='learning rate')
    parser.add_argument('--inter_batch', default=1024, type=int, help='batch size')#1024
    parser.add_argument('--note', default=None, type=str, help='note')
    parser.add_argument('--hvg_top', default=2000, type=int, help='number of top genes')  # 2k
    parser.add_argument('--lambda1', default=0.01, type=float, help='weight of cl loss')  # 0.01
    parser.add_argument('--lambda2', default=1, type=float, help='weight of bpr loss')
    parser.add_argument('--lambda3', default=0.3, type=float, help='weight of ZINB loss') #0.3
    parser.add_argument('--lambda4', default=1e-7, type=float, help='l2 reg weight')
    parser.add_argument('--epoch', default=50, type=int, help='number of epochs')
    parser.add_argument('--d', default=64, type=int, help='embedding size')#64
    parser.add_argument('--q', default=5, type=int, help='rank')
    parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')#2
    parser.add_argument('--data', default='Macosko_mouse_retina', type=str, help='name of dataset')
    parser.add_argument('--dropout', default=0.00, type=float, help='rate for edge dropout')#0.01
    parser.add_argument('--temp', default=0.2, type=float, help='temperature in cl loss')
    parser.add_argument('--cuda', default='1', type=str, help='the gpu to use')
    parser.add_argument('--log_interval', default='5', type=int, help='how many iterations to wait before logging training status.')
    parser.add_argument('--sample_rate', default='0.1', type=float, help='The edge sampling rate of bipartite graph in decoder')
    parser.add_argument('--patience', default='10', type=int, help='early stopping patience')
    return parser.parse_args()

args = parse_args()
