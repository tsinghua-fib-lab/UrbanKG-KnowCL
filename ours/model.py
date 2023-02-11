import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
from compGCN import CompGraphConv
import torch.nn as nn


def get_resnet(name, pretrained=False):
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


''' 
Load state_dict in pre_model to model
Solve the problem that model and pre_model have some different keys
'''
import collections


def load_pretrain(model, pre_s_dict):
    s_dict = model.state_dict()
    # remove fc weights and bias
    pre_s_dict.pop('projector.0.weight')
    pre_s_dict.pop('projector.2.weight')
    # use new dict to store states, record missing keys
    missing_keys = []
    new_state_dict = collections.OrderedDict()
    for key in s_dict.keys():
        simclr_key = 'encoder.' + key
        if simclr_key in pre_s_dict.keys():
            new_state_dict[key] = pre_s_dict[simclr_key]
        else:
            new_state_dict[key] = s_dict[key]
            missing_keys.append(key)
    print('{} keys are not in the pretrain model:'.format(len(missing_keys)), missing_keys)
    # load new s_dict
    model.load_state_dict(new_state_dict)
    return model


class Pair_CLIP_SI(nn.Module):
    def __init__(self, node_emb, rel_emb, layer_size, layer_dropout, **kwargs):
        super(Pair_CLIP_SI, self).__init__()
        d = kwargs['d']
        self.g = kwargs['g']
        self.layer_size = layer_size
        self.layer_dropout = layer_dropout
        self.num_layer = len(layer_size)

        model_simCLR_resnet18_path = "../data/model_pretrain/checkpoint_100.tar"
        resnet18_pretrain = torch.load(model_simCLR_resnet18_path)
        mlp_pretrain = torch.load(model_simCLR_resnet18_path)
        self.si_encoder = get_resnet(name='resnet18', pretrained=False)
        self.si_encoder.fc = Identity()
        self.si_encoder = load_pretrain(self.si_encoder, resnet18_pretrain)
        self.projector_si = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 64, bias=False),
        )
        self.projector_si[0].weight.data = mlp_pretrain['projector.0.weight']
        self.projector_si[2].weight.data = mlp_pretrain['projector.2.weight']

        # CompGCN layers
        self.layers = nn.ModuleList()
        self.layers.append(CompGraphConv(64, self.layer_size[0]))
        for i in range(self.num_layer - 1):
            self.layers.append(CompGraphConv(self.layer_size[i], self.layer_size[i + 1]))

        # Initial relation embeddings
        self.rel_embds = nn.Embedding.from_pretrained(rel_emb, freeze=True)
        # Node embeddings
        self.n_embds = nn.Embedding.from_pretrained(node_emb, freeze=True)
        # Dropout after compGCN layers
        self.dropouts = nn.ModuleList()
        for i in range(self.num_layer):
            self.dropouts.append(nn.Dropout(self.layer_dropout[i]))
        # CompGCN +mlp_projector
        self.projector_ent = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
        )

        torch.nn.init.xavier_normal_(self.projector_ent[0].weight.data)
        torch.nn.init.xavier_normal_(self.projector_ent[2].weight.data)

    def forward(self, si, kg_idx):
        si = self.si_encoder(si)
        si_features = self.projector_si(si)

        n_feats = self.n_embds.weight
        r_feats = self.rel_embds.weight
        for layer, dropout in zip(self.layers, self.dropouts):
            n_feats, r_feats = layer(self.g, n_feats, r_feats)
            n_feats = dropout(n_feats)
        kge_features = n_feats[kg_idx, :]
        kge_features = self.projector_ent(kge_features)

        # calculate loss for kge-sv-si contrastive
        score = torch.einsum('ai, ci->ac', kge_features, si_features)
        # [nb, nb]
        score_1 = F.softmax(score, dim=1)
        diag_1 = torch.diag(score_1)
        loss_1 = -torch.log(diag_1 + 1e-10).sum()
        # [nb, nb]
        score_2 = F.softmax(score, dim=0)
        diag_2 = torch.diag(score_2)
        loss_2 = -torch.log(diag_2 + 1e-10).sum()
        return loss_1 + loss_2

    def get_feature(self, si):
        si_features = self.si_encoder(si)
        return si_features


class Pair_CLIP_SV(nn.Module):
    def __init__(self, node_emb, rel_emb, layer_size, layer_dropout, **kwargs):
        super(Pair_CLIP_SV, self).__init__()
        d = kwargs['d']
        self.g = kwargs['g']
        self.layer_size = layer_size
        self.layer_dropout = layer_dropout
        self.num_layer = len(layer_size)

        model_simCLR_resnet18_path = "../data/model_pretrain/checkpoint_100.tar"
        resnet18_pretrain_sv = torch.load(model_simCLR_resnet18_path)
        mlp_pretrain_sv = torch.load(model_simCLR_resnet18_path)
        self.sv_encoder = get_resnet(name='resnet18', pretrained=False)
        self.sv_encoder.fc = Identity()
        self.sv_encoder = load_pretrain(self.sv_encoder, resnet18_pretrain_sv)
        self.projector_sv = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(),
            nn.Linear(512, 64, bias=False),
        )
        self.projector_sv[0].weight.data = mlp_pretrain_sv['projector.0.weight']
        self.projector_sv[2].weight.data = mlp_pretrain_sv['projector.2.weight']

        # CompGCN layers
        self.layers = nn.ModuleList()
        self.layers.append(CompGraphConv(64, self.layer_size[0]))
        for i in range(self.num_layer - 1):
            self.layers.append(CompGraphConv(self.layer_size[i], self.layer_size[i + 1]))

        # Initial relation embeddings
        self.rel_embds = nn.Embedding.from_pretrained(rel_emb, freeze=True)
        # Node embeddings
        self.n_embds = nn.Embedding.from_pretrained(node_emb, freeze=True)
        # Dropout after compGCN layers
        self.dropouts = nn.ModuleList()
        for i in range(self.num_layer):
            self.dropouts.append(nn.Dropout(self.layer_dropout[i]))
        # CompGCN +mlp_projector
        self.projector_ent = nn.Sequential(
            nn.Linear(64, 64, bias=False),
            nn.ReLU(),
            nn.Linear(64, 64, bias=False),
        )

        torch.nn.init.xavier_normal_(self.projector_ent[0].weight.data)
        torch.nn.init.xavier_normal_(self.projector_ent[2].weight.data)

    def forward(self, sv, kg_idx):
        sv = sv.reshape(len(kg_idx) * 10, 3, 224, 224)
        sv = self.sv_encoder(sv)
        sv_features = sv.reshape(len(kg_idx), 10, 512)

        sv_features = torch.mean(sv_features, dim=1)

        sv_features = self.projector_sv(sv_features)

        n_feats = self.n_embds.weight
        r_feats = self.rel_embds.weight
        for layer, dropout in zip(self.layers, self.dropouts):
            n_feats, r_feats = layer(self.g, n_feats, r_feats)
            n_feats = dropout(n_feats)
        kge_features = n_feats[kg_idx, :]
        kge_features = self.projector_ent(kge_features)

        score = torch.einsum('ai, bi->ab', kge_features, sv_features)
        # [nb, nb]
        score_1 = F.softmax(score, dim=1)
        diag_1 = torch.diag(score_1)
        loss_1 = -torch.log(diag_1 + 1e-10).sum()
        # [nb, nb]
        score_2 = F.softmax(score, dim=0)
        diag_2 = torch.diag(score_2)
        loss_2 = -torch.log(diag_2 + 1e-10).sum()
        return loss_1 + loss_2

    def get_feature(self, sv):
        batch_size = sv.shape[0]
        sv = sv.reshape(batch_size * 10, 3, 224, 224)
        sv = self.sv_encoder(sv)
        sv_features = sv.reshape(batch_size, 10, 512)
        sv_features = torch.mean(sv_features, dim=1)

        return sv_features
