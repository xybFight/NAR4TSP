import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn, optim
import time


# distance matrix and knn matrix
def Generate_Two_Matrix(data, k):
    dis_matrix = torch.cdist(data, data, p=2)
    dis_matrix01 = torch.where(dis_matrix <= torch.topk(dis_matrix, k + 1, largest=False)[0][..., -1, None], 1, 0)
    return dis_matrix, dis_matrix01


# compute tour length with beam search
def compute_tour_length_all_b(x, tour):
    """
    x:(batch,nb_nodes,2)
    tour:(batch,b_width,nb_node)
    return:L(batch, b_width)
    """
    bsz, nb_node, b_width = x.shape[0], x.shape[1], tour.shape[1]
    index = tour.unsqueeze(3).expand(bsz, -1, nb_node, 2)
    seq_expand = x[:, None, :, :].expand(bsz, b_width, nb_node, 2)
    order_seq = seq_expand.gather(dim=2, index=index)
    rolled_seq = order_seq.roll(dims=2, shifts=-1)
    segment_length = ((order_seq - rolled_seq) ** 2).sum(3).sqrt()
    travel_distances = segment_length.sum(2)
    return travel_distances


# compute tour length with greedy search
def compute_tour_length_single(x, tour):
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


# greedy search for a pointer and a score matrix
def greedy_search(score, y_pred_edges):
    bsz, nb_nodes = score.shape[0], score.shape[1]
    # [0, 1, ... bsz-1]
    zero_to_bsz = torch.arange(bsz)
    tours = []
    node_begin = torch.argmax(score, dim=1)
    # mask
    mask_visited_nodes = torch.zeros(bsz, nb_nodes).bool().to(score.device)
    mask_visited_nodes[zero_to_bsz, node_begin] = True
    idx = node_begin
    tours.append(node_begin)
    for node in range(nb_nodes - 1):
        # mask with $-inf$
        prob = torch.softmax(y_pred_edges[zero_to_bsz, idx].masked_fill(mask_visited_nodes, float('-inf')), dim=-1)
        idx = torch.argmax(prob, dim=1)
        # update mask
        mask_visited_nodes = mask_visited_nodes.clone()
        mask_visited_nodes[zero_to_bsz, idx] = True
        tours.append(idx)
    tours = torch.stack(tours, dim=1)
    return tours


# beam search for a pointer and a score matrix
def beam_search(score, y_pred_edges, beam_width=100):
    bsz, nb_nodes = score.shape[0], score.shape[1]
    assert beam_width < nb_nodes ** 2
    # [0, 1, ... bsz-1]
    zero_to_bsz = torch.arange(bsz, device=score.device)
    zero_to_B = torch.arange(beam_width, device=score.device)
    # mask
    mask_visited_nodes = torch.zeros(bsz, nb_nodes, device=score.device).bool()
    for t in range(nb_nodes):
        if t == 0:
            B_t0 = min(beam_width, nb_nodes)
            # 进行log
            score_t = torch.log(score)
            sum_scores = score_t
            top_val, top_idx = torch.topk(sum_scores, B_t0, dim=1)
            sum_scores = top_val
            zero_to_B_t0 = torch.arange(B_t0, device=score.device)
            zero_to_bsz_idx = zero_to_bsz[:, None].expand(bsz, B_t0)
            zero_to_B_t0_idx = zero_to_B_t0[None, :].expand(bsz, B_t0)
            mask_visited_nodes = mask_visited_nodes.unsqueeze(1)  # size(mask_visited_nodes)=(bsz, 1, nb_nodes)
            mask_visited_nodes = torch.repeat_interleave(mask_visited_nodes, B_t0, dim=1)  # (bsz, B_t0, nb_nodes)
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_t0_idx, top_idx] = True
            tours = torch.zeros(bsz, B_t0, nb_nodes, device=score.device, dtype=torch.long)
            tours[:, :, 0] = top_idx
        elif t == 1:
            top_idx_expand = top_idx.unsqueeze(2).expand(-1, -1, nb_nodes)
            prob_next_node = y_pred_edges.gather(dim=1, index=top_idx_expand)
            score_t = torch.log_softmax(prob_next_node.masked_fill(mask_visited_nodes, float('-inf')), dim=-1)

            sum_scores = score_t + sum_scores.unsqueeze(2)  # (bsz, Bt0, nodes)

            sum_scores_flatten = sum_scores.view(bsz, -1)
            # top-k
            top_val, top_idx = torch.topk(sum_scores_flatten, beam_width, dim=1)
            idx_top_beams = torch.div(top_idx, nb_nodes, rounding_mode='trunc')
            idx_in_beams = top_idx % nb_nodes
            sum_scores = top_val
            # load mask
            mask_visited_nodes_tmp = mask_visited_nodes.clone()
            mask_visited_nodes = torch.zeros(bsz, beam_width, nb_nodes, device=x.device).bool()
            zero_to_bsz_idx = zero_to_bsz[:, None].expand(bsz, beam_width)
            zero_to_B_idx = zero_to_B[None, :].expand(bsz, beam_width)
            # update mask for previous
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx] = mask_visited_nodes_tmp[zero_to_bsz_idx, idx_top_beams]
            # update mask for now
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx, idx_in_beams] = True
            # update tour
            tours_tmp = tours.clone()
            tours = torch.zeros(bsz, beam_width, nb_nodes, device=score.device, dtype=torch.long)
            # tour index for time 0
            tours[zero_to_bsz_idx, zero_to_B_idx] = tours_tmp[zero_to_bsz_idx, idx_top_beams]
            # tour index for time 1
            tours[:, :, t] = idx_in_beams
        else:
            idx_in_beams_expand = idx_in_beams.unsqueeze(2).expand(-1, -1, nb_nodes)
            prob_next_node = y_pred_edges.gather(dim=1, index=idx_in_beams_expand)
            score_t = torch.log_softmax(prob_next_node.masked_fill(mask_visited_nodes, float('-inf')), dim=-1)

            sum_scores = score_t + sum_scores.unsqueeze(2)  # (bsz, B, nodes)
            sum_scores_flatten = sum_scores.view(bsz, -1)
            # top-k
            top_val, top_idx = torch.topk(sum_scores_flatten, beam_width, dim=1)
            idx_top_beams = torch.div(top_idx, nb_nodes, rounding_mode='trunc')
            idx_in_beams = top_idx % nb_nodes
            sum_scores = top_val
            # mask
            mask_visited_nodes_tmp = mask_visited_nodes.clone()
            # update mask for previous
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx] = mask_visited_nodes_tmp[zero_to_bsz_idx, idx_top_beams]
            # update mask for now
            mask_visited_nodes[zero_to_bsz_idx, zero_to_B_idx, idx_in_beams] = True
            tours_tmp = tours.clone()
            # update tour
            tours[zero_to_bsz_idx, zero_to_B_idx] = tours_tmp[zero_to_bsz_idx, idx_top_beams]
            tours[:, :, t] = idx_in_beams
    tours_beamsearch = tours
    return tours_beamsearch



class DotDict(dict):
    def __init__(self, **kwds):
        super().__init__()
        self.update(kwds)


args = DotDict()

args.device = torch.device('cuda')
args.learning_rate = 1e-4

args.hidden_dim = 128
args.num_heads = 8
args.num_encoder_layers = 6
args.num_decoder_layers = 2


# model
Model = nn.DataParallel(
    PointGnnModel(args.hidden_dim, args.num_heads, args.num_encoder_layers, args.num_decoder_layers))

args.num_nodes = 100
args.num_k_neigh = args.num_nodes // 5
args.num_batch_size = 500 if args.num_nodes == 50 else 128
f = open('checkpoint/checkpoint-n'+str(args.num_nodes)+'.pkl', 'rb')
record = torch.load(f)
Model.to(args.device)
Model.eval()
Model.load_state_dict(record['parameter'])

x_10k = torch.load('data/10k_TSP'+str(args.num_nodes)+'.pt').to(args.device)
dis_matrix_10k, dis_matrix01_10k = Generate_Two_Matrix(x_10k, args.num_k_neigh)
args.nb_batch_eval = (10000 + args.num_batch_size - 1) // args.num_batch_size

greedy = True

B = 1000
tours_length = torch.zeros(size=(0, 1), device=args.device)
start_time = time.time()
for step in range(args.nb_batch_eval):
    with torch.no_grad():
        x = x_10k[step * args.num_batch_size:(step + 1) * args.num_batch_size, :, :]
        dis_matrix = dis_matrix_10k[step * args.num_batch_size:(step + 1) * args.num_batch_size, :, :]
        dis_matrix01 = dis_matrix01_10k[step * args.num_batch_size:(step + 1) * args.num_batch_size, :, :]
        score, y_pred_edges = Model(dis_matrix01, dis_matrix, x)
        if greedy:
            # greedy search
            tours_greedy = greedy_search(score, y_pred_edges)
            L_greedy = compute_tour_length_single(x, tours_greedy)
            tours_length = torch.cat((tours_length, L_greedy[:, None]), dim=0)
        else:
            # beamsearch
            tours_beamsearch = beam_search(score, y_pred_edges, B)
            L_beamsearch = compute_tour_length_all_b(x, tours_beamsearch)
            L_beamsearch, idx_min = L_beamsearch.min(dim=1)
            tours_length = torch.cat((tours_length, L_beamsearch[:, None]), dim=0)

total_time = time.time() - start_time
print('time:{:.2f}s, mean_length:{}, std:{}'.format(total_time, torch.mean(tours_length, dim=0),
                                                      torch.std(tours_length, unbiased=False, dim=0)))
