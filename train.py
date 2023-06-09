import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn, optim
import datetime
from torch.distributions.categorical import Categorical
import time
from torch.optim import lr_scheduler


# distance matrix and knn matrix
def Generate_Two_Matrix(data, k):
    dis_matrix = torch.cdist(data, data, p=2)
    dis_matrix01 = torch.where(dis_matrix <= torch.topk(dis_matrix, k + 1, largest=False)[0][..., -1, None], 1, 0)
    return dis_matrix, dis_matrix01


# compute tour length with greedy search
def compute_tour_length(x, tour):
    """
    x:(batch,nb_nodes,2)
    tour:(batch, nb_node)
    return: L(batch, 1)
    """
    bsz, nb_node = x.shape[0], x.shape[1]
    index = tour.unsqueeze(2).expand(bsz, nb_node, 2)
    order_seq = x.gather(dim=1, index=index)
    rolled_seq = order_seq.roll(dims=1, shifts=-1)
    segment_length = ((order_seq - rolled_seq) ** 2).sum(2).sqrt()
    travel_distances = segment_length.sum(1)
    return travel_distances


# multiple FC layers
class decoder_MLP(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, L: int):
        assert L > 1
        super(decoder_MLP, self).__init__()
        self.linear1 = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(L - 1)])
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.L = L
        self.activation = nn.ReLU()

    def forward(self, x):
        for i in range(self.L - 1):
            x = self.activation(self.linear1[i](x))
        y = self.linear2(x).squeeze(-1)
        return y


# GAT layer
class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, nb_heads: int, leaky_relu_negative_slope: float = 0.2):
        super(GraphAttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.nb_heads = nb_heads
        self.n_hidden = hidden_dim // nb_heads
        self.linear = nn.Linear(hidden_dim, self.n_hidden * nb_heads, bias=False)
        self.linear_e = nn.Linear(hidden_dim, self.n_hidden * nb_heads, bias=False)
        self.attn = nn.Linear(self.n_hidden * 3, 1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=leaky_relu_negative_slope)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, dis_mat: torch.Tensor):
        bsz, nb_node = h.shape[0], h.shape[1]
        # embedding
        g = self.linear(h).view(bsz, nb_node, self.nb_heads, self.n_hidden)
        g_repeat = g.repeat(1, nb_node, 1, 1)
        g_repeat_interleave = g.repeat_interleave(nb_node, dim=1)
        dis_mat = self.linear_e(dis_mat).view(bsz, nb_node * nb_node, self.nb_heads, self.n_hidden)
        g_concat = torch.cat([g_repeat_interleave, g_repeat, dis_mat], dim=-1)
        g_concat = g_concat.view(bsz, nb_node, nb_node, self.nb_heads, 3 * self.n_hidden)
        e = self.activation(self.attn(g_concat)).squeeze(-1)
        # mask
        e = e.masked_fill(adj_mat == 0, float('-inf'))
        # softmax
        a = self.softmax(e)
        attn_res = torch.einsum('bijh,bjhf->bihf', a, g)
        return attn_res.reshape(bsz, nb_node, self.nb_heads * self.n_hidden)


# Edge feature transition
class EdgeTransition(nn.Module):
    def __init__(self, hidden_dim: int):
        super(EdgeTransition, self).__init__()
        self.Eweight1 = nn.Linear(hidden_dim, hidden_dim)
        self.Hweight1 = nn.Linear(hidden_dim, hidden_dim)
        self.Hweight2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, h: torch.Tensor, dis_mat: torch.Tensor):
        dis_mat_emb = self.Eweight1(dis_mat)
        h_emb = self.Hweight1(h.unsqueeze(1)) + self.Hweight2(h.unsqueeze(2))
        return h_emb + dis_mat_emb


# BN for node
class NodeBN(nn.Module):
    def __init__(self, hidden_dim):
        super(NodeBN, self).__init__()
        self.batch_norm = nn.BatchNorm1d(hidden_dim, track_running_stats=False)

    def forward(self, x):
        if len(x[0]) == 1:
            return x
        else:
            x_trans = x.transpose(1, 2).contiguous()  # Reshape input: (batch_size, hidden_dim, num_nodes)
            x_trans_bn = self.batch_norm(x_trans)
            x_bn = x_trans_bn.transpose(1, 2).contiguous()  # Reshape to original shape
            del x_trans, x_trans_bn
            return x_bn


# BN for edge
class EdgeBN(nn.Module):
    def __init__(self, hidden_dim):
        super(EdgeBN, self).__init__()
        self.batch_norm = nn.BatchNorm2d(hidden_dim, track_running_stats=False)

    def forward(self, e):
        if len(e[0]) == 1:
            return e
        else:
            e_trans = e.permute(0, 3, 1,
                                2).contiguous()  # Reshape input: (batch_size, num_nodes, num_nodes, hidden_dim)
            e_trans_bn = self.batch_norm(e_trans)
            e_bn = e_trans_bn.permute(0, 2, 3, 1).contiguous()  # Reshape to original
            del e_trans, e_trans_bn
            return e_bn


# GNN layer
class GraphAttentionEncoder(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int):
        super(GraphAttentionEncoder, self).__init__()
        assert hidden_dim % num_heads == 0
        self.GAT_layer = nn.ModuleList([GraphAttentionLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        self.EdgeTransitionLayer = nn.ModuleList(EdgeTransition(hidden_dim) for _ in range(num_layers))
        self.NodeBN1 = nn.ModuleList([NodeBN(hidden_dim) for _ in range(num_layers)])
        self.EdgeBN1 = nn.ModuleList([EdgeBN(hidden_dim) for _ in range(num_layers)])
        self.nb_layers = num_layers
        self.MHA_layer = nn.ModuleList([nn.MultiheadAttention(hidden_dim, num_heads) for _ in range(num_layers)])
        self.att_norm_layer = nn.ModuleList([NodeBN(hidden_dim) for _ in range(num_layers)])
        self.activation1 = nn.Sigmoid()

    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, dis_mat: torch.Tensor, h_start: torch.Tensor):
        for i in range(self.nb_layers):
            h_rc = h
            h = self.GAT_layer[i](h, adj_mat, dis_mat) + h_rc
            h = self.NodeBN1[i](h)
            dis_mat_rc = dis_mat
            dis_mat = self.activation1(self.EdgeTransitionLayer[i](h, dis_mat)) + dis_mat_rc
            dis_mat = self.EdgeBN1[i](dis_mat)
            h_start_rc = h_start
            h_transpose = h.transpose(0, 1).contiguous()
            if i == self.nb_layers - 1:
                _, score = self.MHA_layer[i](h_start, h_transpose, h_transpose)
            else:
                h_start, _ = self.MHA_layer[i](h_start, h_transpose, h_transpose)
                h_start = self.att_norm_layer[i](h_start + h_start_rc)
        return dis_mat, score.squeeze(1)


# NAR4TSP
class PointGnnModel(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_encoder_layers: int, num_decoder_layers: int):
        super(PointGnnModel, self).__init__()
        self.nodes_coord_embedding = nn.Linear(2, hidden_dim)
        self.edges_values_embedding = nn.Linear(1, hidden_dim)
        self.encoder = GraphAttentionEncoder(hidden_dim, num_heads, num_encoder_layers)
        self.mlp_edges = decoder_MLP(hidden_dim, 1, num_decoder_layers)
        self.starting_symbol = nn.Parameter(torch.randn(hidden_dim))

    def forward(self, x_edges, x_edges_values, x_nodes_coord):
        bsz = x_edges.shape[0]
        # [0, 1, ... bsz-1]
        h = self.nodes_coord_embedding(x_nodes_coord)
        # v_h
        h_start = self.starting_symbol.repeat(1, bsz, 1)
        dis_mat = self.edges_values_embedding(x_edges_values.unsqueeze(-1))
        dis_mat, score = self.encoder(h, x_edges.unsqueeze(-1), dis_mat, h_start)
        y_pred_edges = self.mlp_edges(dis_mat)  # B×V×V
        return score, y_pred_edges


# return tour and its corresponding probability
def return_tour_pro(score, y_pred_edges, deterministic):
    bsz, nb_nodes = score.shape[0], score.shape[1]
    zero_to_bsz = torch.arange(bsz)
    tours, sumLogProOfActions, = [], []
    if deterministic:
        # greedy
        node_begin = torch.argmax(score, dim=1)
    else:
        # Bernoulli
        node_begin = Categorical(score).sample()
    # mask
    mask_visited_nodes = torch.zeros(bsz, nb_nodes).bool().to(score.device)
    mask_visited_nodes[zero_to_bsz, node_begin] = True

    idx = node_begin
    tours.append(node_begin)
    ProbOfChoices = score[zero_to_bsz, idx]
    sumLogProOfActions.append(torch.log(ProbOfChoices + 1e-10))

    for node in range(nb_nodes - 1):
        prob = torch.softmax(y_pred_edges[zero_to_bsz, idx].masked_fill(mask_visited_nodes, float('-inf')), dim=-1)
        if deterministic:
            idx = torch.argmax(prob, dim=1)
        else:
            idx = Categorical(prob).sample()
        # update mask
        mask_visited_nodes = mask_visited_nodes.clone()
        mask_visited_nodes[zero_to_bsz, idx] = True
        ProbOfChoices = prob[zero_to_bsz, idx]
        sumLogProOfActions.append(torch.log(ProbOfChoices + 1e-10))
        tours.append(idx)
    sumLogProOfActions = torch.stack(sumLogProOfActions, dim=1).sum(dim=1)
    tours = torch.stack(tours, dim=1)
    return tours, sumLogProOfActions


# 定义超参数
class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)


args = DotDict()

args.device = torch.device('cuda')
args.learning_rate = 1e-4
args.num_nodes = 50
args.num_k_neigh = args.num_nodes // 5
# only for training 50 and 100
args.num_epochs = 1000 if args.num_nodes == 50 else 2000
args.num_batch_size = 64
args.num_batch_per_epoch = 2500
args.hidden_dim = 128
args.num_heads = 8
args.num_encoder_layers = 6
args.num_decoder_layers = 2

# val set
x_10k = torch.rand(10000, args.num_nodes, 2, device=args.device)
dis_matrix_10k, dis_matrix01_10k = Generate_Two_Matrix(x_10k, args.num_k_neigh)


args.nb_batch_eval = (10000 + args.num_batch_size - 1) // args.num_batch_size


Model_Train = nn.DataParallel(
    PointGnnModel(args.hidden_dim, args.num_heads, args.num_encoder_layers, args.num_decoder_layers))
Model_Train.to(args.device)
total = sum([param.nelement() for param in Model_Train.parameters()])
print("Number of parameter: %.2fM" % (total / 1e6))
# 优化器
optimizer = optim.Adam(Model_Train.parameters(), lr=args.learning_rate)

min_len = 100
for epoch in range(args.num_epochs):
    Model_Train.train()
    start_time = time.time()
    for _ in range(args.num_batch_per_epoch):
        x = torch.rand(args.num_batch_size, args.num_nodes, 2, device=args.device)
        dis_matrix, dis_matrix01 = Generate_Two_Matrix(x, args.num_k_neigh)
        score, y_pred_edges = Model_Train(dis_matrix01, dis_matrix, x)
        tours_train, sumLogProOfActions = return_tour_pro(score, y_pred_edges, False)
        tours_baseline, _ = return_tour_pro(score, y_pred_edges, True)
        L_train = compute_tour_length(x, tours_train)
        L_baseline = compute_tour_length(x, tours_baseline)

        gap = torch.mean(L_train - L_baseline)
        loss = torch.mean((L_train - L_baseline - gap) * sumLogProOfActions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    time_one_epoch = time.time() - start_time
    Model_Train.eval()
    mean_tour_length_val_t, mean_tour_length_val_b = 0, 0
    tours_length_t = torch.zeros(size=(0, 1), device=args.device)
    tours_length_b = torch.zeros(size=(0, 1), device=args.device)
    with torch.no_grad():
        for step in range(args.nb_batch_eval):
            x = x_10k[step * args.num_batch_size:(step + 1) * args.num_batch_size, :, :]
            dis_matrix = dis_matrix_10k[step * args.num_batch_size:(step + 1) * args.num_batch_size, :, :]
            dis_matrix01 = dis_matrix01_10k[step * args.num_batch_size:(step + 1) * args.num_batch_size, :, :]

            score, y_pred_edges = Model_Train(dis_matrix01, dis_matrix, x)
            tours_val_t, _ = return_tour_pro(score, y_pred_edges, False)
            tours_val_b, _ = return_tour_pro(score, y_pred_edges, True)
            L_val_t = compute_tour_length(x, tours_val_t)
            L_val_b = compute_tour_length(x, tours_val_b)

            tours_length_t = torch.cat((tours_length_t, L_val_t[:, None]), dim=0)
            tours_length_b = torch.cat((tours_length_b, L_val_b[:, None]), dim=0)
        mean_tour_length_val_b = torch.mean(tours_length_b, dim=0).item()
        mean_tour_length_val_t = torch.mean(tours_length_t, dim=0).item()
    mystring_min = 'Epoch: {:d}, L_train: {:.3f}, L_base: {:.3f}, epoch time:{:.3f}min'.format(
        epoch, mean_tour_length_val_t, mean_tour_length_val_b, time_one_epoch / 60)
    print(mystring_min)
    if min_len > mean_tour_length_val_b:
        min_len = mean_tour_length_val_b
        # 保存检查点
        checkpoint_dir = os.path.join('checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save({
            'parameter': Model_Train.state_dict(),},
                '{}.pkl'.format('checkpoint' + '-n{}'.format(args.num_nodes)))
