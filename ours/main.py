import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from load_data import Data_CompGCN, region_dataset
from torch.utils.data import DataLoader
from torch import nn
import torch
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
import argparse
from model import Pair_CLIP_SI, Pair_CLIP_SV
import json
from cal import in_out_norm


def trainer(args, model, train_loader, optimizer, epoch):
    loss_epoch = []
    model.train()
    for step, batch in enumerate(train_loader):
        optimizer.zero_grad()
        if args.model_name == 'Pair_CLIP_SI':
            si, kg_idx = batch[0].to(args.device), batch[2].to(args.device)
            loss = model(si, kg_idx)
        elif args.model_name == 'Pair_CLIP_SV':
            sv, kg_idx = batch[1].to(args.device), batch[2].to(args.device)
            loss = model(sv, kg_idx)
        else:
            print('Wrong model!!!')
        loss.backward()
        optimizer.step()

        loss_epoch.append(loss.item())
        if step % 24 == 0:
            print(f"TrainStep [{step}/{len(train_loader)}]\t train_loss: {loss.item()}")
        args.global_step += 1
    print(f"TrainEpoch [{epoch}/{args.epochs}\t train_loss_epoch:{np.mean(loss_epoch)}")
    return np.mean(loss_epoch)


def extract_feature(args, model, extract_loader):
    # load pre-trained model from checkpoint
    model.eval()
    for load_epoch in [args.epochs]:
        print("epoch:" + str(load_epoch))
        model_fp = os.path.join(args.model_path, "checkpoint_{}.tar".format(load_epoch))
        model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
        model = model.to(args.device)
        feature_vector = []
        for step, batch in enumerate(extract_loader):
            si, sv, kg_idx = batch[0].to(args.device), batch[1].to(args.device), batch[2].to(args.device)
            if args.model_name == 'Pair_CLIP_SI':
                input_images = si
            elif args.model_name == 'Pair_CLIP_SV':
                input_images = sv
            with torch.no_grad():
                image_features = model.get_feature(input_images)
                feature_vector.extend(image_features.cpu().numpy())
            if step % 100 == 0:
                print(f"Step [{step}/{len(extract_loader)}]\t Computing features...")
        feature_vector = np.array(feature_vector)
        print("Features shape {}".format(feature_vector.shape))
        print("### Creating features from pre-trained context model ###")
        save_path = args.model_path + "/" + str(args.model_name).split('_')[2] + "_feature_vector"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        np.savetxt(save_path + '/region_' + str(load_epoch) + '.txt', feature_vector, fmt="%f")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ours")
    parser.add_argument("--dataset", type=str, default="new_york", nargs="?", help="Dataset")
    parser.add_argument("--start_epoch", type=int, default=0, nargs="?", help="Start epoch")
    parser.add_argument("--current_epoch", type=int, default=0, nargs="?", help="Current epoch")
    parser.add_argument("--global_step", type=int, default=0, nargs="?", help="global_step")
    parser.add_argument("--epochs", type=int, default=100, nargs="?", help="Epochs")
    parser.add_argument("--batch_size", type=int, default=16, nargs="?", help="Batch size.")
    parser.add_argument("--lr", type=float, default=0.0003, nargs="?", help="Learning rate.")
    parser.add_argument("--seed", type=int, default=61, nargs="?", help="random seed.")
    parser.add_argument("--model_name", type=str, default="Pair_CLIP_SV", help="Pair_CLIP_SI, Pair_CLIP_SV")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--kg_path", type=str, default="")
    parser.add_argument("--si_path", type=str, default="")
    parser.add_argument("--sv_path", type=str, default="")
    parser.add_argument('--layer_size', nargs='?', default=[], help='List of output size for each compGCN layer')
    parser.add_argument("--n_gcn_layer", type=int, default=2, nargs="?", help="n_gcn_layer.")
    parser.add_argument('--layer_dropout', nargs='?', default=[],
                        help='List of dropout value after each compGCN l,ayer')
    parser.add_argument("--TuckER_pretrain_path", type=str, default=".npz")
    parser.add_argument("--poi_streetview_filename_path", type=str, default="")

    args = parser.parse_args()

    for n_gcn in range(args.n_gcn_layer):
        args.layer_size.append(64)
        args.layer_dropout.append(0.3)

    args.kg_path = "../data/" + args.dataset + "/" + args.dataset + "_0715.txt"
    args.sv_path = "../data/" + args.dataset + "/streetview_image/Region/"
    args.si_path = "../data/" + args.dataset + "/satellite_image/zl15_224/"

    args.TuckER_pretrain_path = "../data/" + args.dataset + "/whole_ER_" + args.dataset + "_TuckER64.npz"

    args.poi_streetview_filename_path = "../data/" + args.dataset + "/streetview_image/region_5_10_poi_image_filename.json"

    args.model_path = "./works/" + args.model_name + "/" + args.dataset + "/check_" \
                      + str(args.epochs) + "_lr_" + str(args.lr) + "_gcn_" + str(args.n_gcn_layer) + "_bs_" + str(
        args.batch_size)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    with open(args.poi_streetview_filename_path, 'r') as f_json:
        region_poi_streetview_filename = json.load(f_json)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(False)

    d = Data_CompGCN(args.kg_path, args.dataset)

    region_poi_streetview_ents = sorted(list(region_poi_streetview_filename.keys()), key=lambda y: int(y))

    region_poi_streetview_idxs = [d.ent2id[x] for x in region_poi_streetview_ents]
    train_dataset = region_dataset(region_poi_streetview_idxs, d.id2ent, args.si_path, \
                                   args.sv_path, dict(region_poi_streetview_filename))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    print('len of train_dataset:', len(train_dataset))

    extract_dataset = region_dataset(region_poi_streetview_idxs, d.id2ent, args.si_path, \
                                     args.sv_path, dict(region_poi_streetview_filename))
    extract_loader = DataLoader(extract_dataset, batch_size=16, shuffle=False, num_workers=8)
    print('len of extract_dataset:', len(extract_dataset))

    pretrain_emb = np.load(args.TuckER_pretrain_path)
    node_emb = torch.FloatTensor(pretrain_emb['E_pretrain'])
    rel_emb = torch.FloatTensor(pretrain_emb['R_pretrain'])

    g = in_out_norm(d.g.to(args.device))
    kwargs = {'d': d, 'g': g}
    if args.model_name == 'Pair_CLIP_SI':
        model = Pair_CLIP_SI(node_emb, rel_emb, args.layer_size, args.layer_dropout, **kwargs)
    elif args.model_name == 'Pair_CLIP_SV':
        model = Pair_CLIP_SV(node_emb, rel_emb, args.layer_size, args.layer_dropout, **kwargs)

    model = model.to(args.device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8)

    for epoch in range(args.start_epoch, args.epochs):
        loss_epoch = trainer(args, model, train_loader, opt, epoch)
        if epoch in range(0, args.epochs, 20):
            out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))
            torch.save(model.state_dict(), out)
        args.current_epoch += 1
    out = os.path.join(args.model_path, "checkpoint_{}.tar".format(args.current_epoch))
    torch.save(model.state_dict(), out)

    extract_feature(args, model, extract_loader)
