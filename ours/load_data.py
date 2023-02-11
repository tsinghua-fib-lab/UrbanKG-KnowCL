from tqdm import tqdm
import dgl
from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torch


class Data_CompGCN(object):
    def __init__(self, kg_dir, dataset_name):
        # kg_dir: directory of kg_data
        self.dataset_name = dataset_name
        self.ents, self.rels, self.ent2id, self.rel2id, self.kg_data, self.region_kg_idxs = self.load_kg(kg_dir=kg_dir)
        self.num_ent, self.num_rel = len(self.ent2id), len(self.rel2id) // 2
        self.id2ent = {v: k for k, v in self.ent2id.items()}
        rels = [x[1] for x in self.kg_data]
        src = [x[0] for x in self.kg_data]
        dst = [x[2] for x in self.kg_data]

        self.g = dgl.graph((src, dst), num_nodes=self.num_ent)
        self.g.edata['etype'] = torch.Tensor(rels).long()
        # identify in and out edges
        in_edges_mask = [True] * (self.g.num_edges() // 2) + [False] * (self.g.num_edges() // 2)
        out_edges_mask = [False] * (self.g.num_edges() // 2) + [True] * (self.g.num_edges() // 2)
        self.g.edata['in_edges_mask'] = torch.Tensor(in_edges_mask)
        self.g.edata['out_edges_mask'] = torch.Tensor(out_edges_mask)

    def load_kg(self, kg_dir):
        facts_str = []
        print('loading knowledge graph...')
        with open(kg_dir, 'r') as f:
            for line in tqdm(f.readlines()):
                x = line.strip().split('\t')
                facts_str.append([x[0], x[1], x[2]])
        origin_rels = sorted(list(set([x[1] for x in facts_str])))
        all_rels = sorted(origin_rels + [x + '_rev' for x in origin_rels])
        all_ents = sorted(list(set([x[0] for x in facts_str] + [x[2] for x in facts_str])))

        # ny
        if 'new_york' in self.dataset_name:
            region_ents = [x for x in all_ents if '360' in x]
            other_ents = [x for x in all_ents if '360' not in x]
            region_ents = sorted(region_ents, key=lambda y: int(y))
            other_ents = sorted(other_ents)
        ents = region_ents + other_ents
        ent2id, rel2id = dict([(x, i) for i, x in enumerate(ents)]), dict([(x, i) for i, x in enumerate(all_rels)])
        kg_data = [[ent2id[x[0]], rel2id[x[1]], ent2id[x[2]]] for x in facts_str] + [
            [ent2id[x[2]], rel2id[x[1] + '_rev'], ent2id[x[2]]] for x in facts_str]  # reverse relations & facts added
        region_kg_idxs = [ent2id[x] for x in region_ents]
        return ents, all_rels, ent2id, rel2id, kg_data, region_kg_idxs


class region_dataset(Dataset):
    def __init__(self, region_kg_idxs, id2ent, si_root_dir, sv_root_dir, streetview_region_dict):
        self.region_kg_idxs = region_kg_idxs
        self.id2ent = id2ent
        self.si_root_dir = si_root_dir
        self.sv_root_dir = sv_root_dir
        self.streetview_region = streetview_region_dict

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.ToTensor(),
            ]
        )
        self.transform = self.transform

    def __len__(self):
        return len(self.region_kg_idxs)

    def __getitem__(self, kg_idx):
        streetview_list = []
        kg_idx = self.region_kg_idxs[kg_idx]
        for img_file_name in self.streetview_region[self.id2ent[kg_idx]]:
            image = Image.open(self.sv_root_dir + self.id2ent[kg_idx] + '/' + img_file_name)
            if self.transform:
                image = self.transform(image)
            streetview_list.append(image)
        k_streetview = torch.stack(streetview_list, 0)

        satellite_image = Image.open(self.si_root_dir + self.id2ent[kg_idx] + '.png')
        if self.transform:
            satellite_image = self.transform(satellite_image)
        return satellite_image, k_streetview, kg_idx
