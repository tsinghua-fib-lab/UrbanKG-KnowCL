import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch
from torch import nn
import math
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from torch.optim.lr_scheduler import ExponentialLR
import argparse
import numpy as np
import pandas as pd
import os
from torch.autograd import Variable
import warnings
from sklearn import metrics
import json


def weights_init_1(m):
    seed = 20
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight, gain=1)


def weights_init_2(m):
    seed = 20
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.nn.init.xavier_uniform_(m.weight, gain=1)
    torch.nn.init.constant_(m.bias, 0)


class linearRegression(torch.nn.Module):
    def __init__(self, inputSize, outputSize, drop_out=0.5):
        super(linearRegression, self).__init__()
        self.linear1 = torch.nn.Linear(inputSize, inputSize, bias=True)
        self.act1 = nn.ReLU()
        self.linear2 = torch.nn.Linear(inputSize, outputSize, bias=True)
        self.dropout = nn.Dropout(drop_out)
        weights_init_2(self.linear1)
        weights_init_2(self.linear2)

    def forward(self, x1):
        out = self.linear1(x1)
        out = self.dropout(out)
        out = self.act1(out)
        out = self.linear2(out)
        return out


def mape_loss_func(preds, labels):
    mask = labels != 0
    return np.fabs((labels[mask] - preds[mask]) / labels[mask]).mean()


def train_eval(args):
    with open(args.region2allinfo_path, 'r') as f_json:
        region_info_json = json.load(f_json)
    warnings.filterwarnings('ignore')
    train_data_index = list(pd.read_csv(args.train_img_csv_index_path, header=0)['csvindex'])
    val_data_index = list(pd.read_csv(args.val_img_csv_index_path, header=0)['csvindex'])
    test_data_index = list(pd.read_csv(args.test_img_csv_index_path, header=0)['csvindex'])
    all_data_image_name = list(pd.read_csv(args.region_img_name_path, header=0)['imagename'])

    indicator = []
    for img_name in all_data_image_name:
        indicator.append(math.log(region_info_json[str(img_name)][args.indicator] + 1, math.e))

    feature_all = np.loadtxt(args.feature_vector_path)
    print(feature_all.shape)

    x_train = feature_all[train_data_index, :]
    y_train = []
    for idx in train_data_index:
        y_train.append(indicator[idx])
    x_val = feature_all[val_data_index, :]
    y_val = []
    for idx in val_data_index:
        y_val.append(indicator[idx])
    x_test = feature_all[test_data_index, :]
    y_test = []
    for idx in test_data_index:
        y_test.append(indicator[idx])

    x_train = torch.as_tensor(x_train, dtype=torch.float32).cuda()
    y_train = np.array(y_train)
    y_train = torch.as_tensor(y_train.reshape((-1, 1)), dtype=torch.float32).cuda()

    x_val = torch.as_tensor(x_val, dtype=torch.float32).cuda()
    y_val = np.array(y_val)
    y_val = torch.as_tensor(y_val.reshape((-1, 1)), dtype=torch.float32)

    x_test = torch.as_tensor(x_test, dtype=torch.float32).cuda()
    y_test = np.array(y_test)
    y_test = torch.as_tensor(y_test.reshape((-1, 1)), dtype=torch.float32)

    global_step = 0
    model = linearRegression(feature_all.shape[1], 1, drop_out=args.drop_out)
    model.cuda()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    if args.dr:
        scheduler = ExponentialLR(optimizer, args.dr)
    best_epoch = 0
    best_val_r2 = -10000
    best_loss = 0
    best_train_r2 = 0
    best_test_r2 = 0
    best_RMSE = 0
    best_MAE = 0
    best_MAPE = 0
    for epoch in range(args.epochs):
        model.train()
        outputs = model(x_train)
        optimizer.zero_grad()
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if args.dr:
            scheduler.step()
        with torch.no_grad():
            model.eval()
            predicted = model(x_train).cpu().data.numpy()
            train_r2 = r2_score(list(y_train.cpu().data.numpy()), list(predicted))

        with torch.no_grad():
            model.eval()
            predicted = model(x_val).cpu().numpy()
            val_loss = criterion(torch.from_numpy(predicted), y_val)

            val_r2 = r2_score(list(y_val.cpu().numpy()), list(predicted))

        with torch.no_grad():
            model.eval()
            predicted = model(x_test).cpu().numpy()
            test_loss = criterion(torch.from_numpy(predicted), y_test)
            r2 = r2_score(list(y_test.cpu().numpy()), list(predicted))
            RMSE = np.sqrt(mean_squared_error(list(y_test.cpu().numpy()), list(predicted)))
            MAPE = mape_loss_func(predicted, y_test.cpu().numpy())
            MAE = metrics.mean_absolute_error(list(y_test.cpu().numpy()), list(predicted))

        global_step = global_step + 1
        if best_val_r2 < val_r2:
            best_val_r2 = val_r2
            best_loss = val_loss
            best_epoch = epoch
            best_train_r2 = train_r2
            best_test_r2 = r2
            best_RMSE = RMSE
            best_MAE = MAE
            best_MAPE = MAPE

            print('bestepoch:', best_epoch)
            print('Epoch:', epoch, 'Train loss:', float(loss))
            print('Train_R2: ', train_r2)
            print('Val_R2: ', val_r2)
            print('Test_R2: ', r2)
            print('RMSE', RMSE)
            print('MAE ', MAE)
            print('MAPE ', MAPE)
        if epoch - best_epoch > args.patience:
            print('=========== Final Results ===========')
            print('Best Epoch: %d\n' % (best_epoch))
            break
        elif epoch == args.epochs - 1:
            print('r2 didn\'t increase for %d epochs, Best Epoch=%d, Best Loss=%.8f' % (
            epoch - best_epoch, best_val_r2, best_loss))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="new_york", nargs="?", help="Dataset")

    parser.add_argument("--epochs", type=int, default=1000, nargs="?", help="MLP train epochs.")
    parser.add_argument("--lr", type=float, default=0.003, nargs="?", help="Learning rate.")
    parser.add_argument("--dr", type=float, default=0.0, nargs="?", help="Decay rate.")
    parser.add_argument("--drop_out", type=float, default=0.1, nargs="?", help="Drop out.")
    parser.add_argument("--wd", type=float, default=0.0, nargs="?", help="Weight Decay")
    parser.add_argument("--indicator", type=str, default="edu", nargs="?", help="Indicator name.")
    parser.add_argument("--patience", type=int, default=300, nargs="?", help="valid patience.")
    parser.add_argument("--seed", type=int, default=42, nargs="?", help="random seed.")
    parser.add_argument("--model_name", type=str, default="Pair_CLIP_SI")

    parser.add_argument("--region2allinfo_path", type=str, default="_region2allinfo.json")
    parser.add_argument("--train_img_csv_index_path", type=str, default="_train_img_csv_index.csv")
    parser.add_argument("--val_img_csv_index_path", type=str, default="_val_img_csv_index.csv")
    parser.add_argument("--test_img_csv_index_path", type=str, default="_test_img_csv_index.csv")
    parser.add_argument("--region_img_name_path", type=str, default="_region_img_name.csv")
    parser.add_argument("--feature_vector_path", type=str, default="")

    parser.add_argument("--KnowCLlr", type=float, default=0.0003)
    parser.add_argument("--KnowCLwd", type=float, default=0.0, nargs="?", help="KnowCL model wd.")
    parser.add_argument("--KnowCLgcn", type=int, default=2, nargs="?", help="KnowCL_n_gcn_layers")
    parser.add_argument("--KnowCLbatchsize", type=int, default=128, nargs="?", help="KnowCL batchsize")
    parser.add_argument("--KnowCLepoch", type=int, default=100, nargs="?", help="KnowCL epoch")

    args = parser.parse_args()

    dataset_path = "../data/" + args.dataset + "/" + args.dataset
    args.region2allinfo_path = dataset_path + args.region2allinfo_path

    args.model_path = "./works/" + args.model_name + "/" + args.dataset + "/check_" \
                      + str(args.KnowCLepoch) + "_lr_" + str(args.KnowCLlr) + "_gcn_" + str(
        args.KnowCLgcn) + "_bs_" + str(args.KnowCLbatchsize)

    args.feature_vector_path = args.model_path + "/" + str(args.model_name).split('_')[
        2] + "_feature_vector/region_" + str(args.KnowCLepoch) + ".txt"

    dataset_path = "../data/" + args.dataset + "/" + args.dataset + "_zl15"
    args.train_img_csv_index_path = dataset_path + args.train_img_csv_index_path
    args.val_img_csv_index_path = dataset_path + args.val_img_csv_index_path
    args.test_img_csv_index_path = dataset_path + args.test_img_csv_index_path
    args.region_img_name_path = dataset_path + args.region_img_name_path

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.use_deterministic_algorithms(False)
    print('Loading data....')
    train_eval(args)
