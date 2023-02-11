import torch as th
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import dgl.function as fn
from cal import ccorr


def rotate(h, r):
    d = h.shape[-1]
    h_re, h_im = th.split(h, d // 2, -1)
    r_re, r_im = th.split(r, d // 2, -1)
    return th.cat([h_re * r_re - h_im * r_im, h_re * r_im + h_im * r_re], dim=-1)


class CompGraphConv(nn.Module):
    """One layer of CompGCN."""

    def __init__(self,
                 in_dim,
                 out_dim,
                 comp_fn='rotate',
                 batchnorm=True,
                 dropout=0.1):
        super(CompGraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = th.tanh
        self.batchnorm = batchnorm

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        # define in/out/loop transform layer
        self.W_O = nn.Linear(self.in_dim, self.out_dim)
        self.W_I = nn.Linear(self.in_dim, self.out_dim)
        self.W_S = nn.Linear(self.in_dim, self.out_dim)

        # define relation transform layer
        self.W_R = nn.Linear(self.in_dim, self.out_dim)

        # self loop embedding
        self.loop_rel = nn.Parameter(th.Tensor(1, self.in_dim))
        nn.init.xavier_normal_(self.loop_rel)

    def forward(self, g, n_in_feats, r_feats):

        with g.local_scope():
            # Assign values to source nodes. In a homogeneous graph, this is equal to
            # assigning them to all nodes.
            g.srcdata['h'] = n_in_feats
            # append loop_rel embedding to r_feats
            r_feats = th.cat((r_feats, self.loop_rel), 0)
            # Assign features to all edges with the corresponding relation embeddings
            g.edata['h'] = r_feats[g.edata['etype']] * g.edata['norm']

            # Compute composition function in 4 steps
            # Step 1: compute composition by edge in the edge direction, and store results in edges.
            if self.comp_fn == 'sub':
                g.apply_edges(fn.u_sub_e('h', 'h', out='comp_h'))
            elif self.comp_fn == 'mul':
                g.apply_edges(fn.u_mul_e('h', 'h', out='comp_h'))
            elif self.comp_fn == 'ccorr':
                g.apply_edges(lambda edges: {'comp_h': ccorr(edges.src['h'], edges.data['h'])})
            elif self.comp_fn == 'rotate':
                g.apply_edges(lambda edges: {'comp_h': rotate(edges.src['h'], edges.data['h'])})
            else:
                raise Exception('Only supports sub, mul, and ccorr')

            # Step 2: use extracted edge direction to compute in and out edges
            comp_h = g.edata['comp_h']

            in_edges_idx = th.nonzero(g.edata['in_edges_mask'], as_tuple=False).squeeze()
            out_edges_idx = th.nonzero(g.edata['out_edges_mask'], as_tuple=False).squeeze()

            comp_h_O = self.W_O(comp_h[out_edges_idx])
            comp_h_I = self.W_I(comp_h[in_edges_idx])

            new_comp_h = th.zeros(comp_h.shape[0], self.out_dim).to(comp_h.device)
            new_comp_h[out_edges_idx] = comp_h_O
            new_comp_h[in_edges_idx] = comp_h_I

            g.edata['new_comp_h'] = new_comp_h

            # Step 3: sum comp results to both src and dst nodes
            g.update_all(fn.copy_e('new_comp_h', 'm'), fn.sum('m', 'comp_edge'))

            # Step 4: add results of self-loop
            if self.comp_fn == 'sub':
                comp_h_s = n_in_feats - r_feats[-1]
            elif self.comp_fn == 'mul':
                comp_h_s = n_in_feats * r_feats[-1]
            elif self.comp_fn == 'ccorr':
                comp_h_s = ccorr(n_in_feats, r_feats[-1])
            elif self.comp_fn == 'rotate':
                comp_h_s = rotate(n_in_feats, r_feats[-1])
            else:
                raise Exception('Only supports sub, mul, and ccorr')

            # Sum all of the comp results as output of nodes and dropout
            n_out_feats = (self.W_S(comp_h_s) + self.dropout(g.ndata['comp_edge'])) * (1 / 3)

            # Compute relation output
            r_out_feats = self.W_R(r_feats)

            # Batch norm
            if self.batchnorm:
                n_out_feats = self.bn(n_out_feats)

            # Activation function
            if self.actvation is not None:
                n_out_feats = self.actvation(n_out_feats)

        return n_out_feats, r_out_feats[:-1]
