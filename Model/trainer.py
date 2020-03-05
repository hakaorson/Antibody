import torch
from sklearn import metrics
import torch.nn as nn


class myloss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, predict):
        pos_log = torch.log(target)
        neg_log = torch.log(1.0-target)
        digits = pos_log*predict+neg_log*(1.0-predict)
        dig_sum = torch.mean(digits)
        return -dig_sum


def train(args, model: torch.nn.Module, data):
    '''
    loss_fnc = torch.nn.BCELoss()  # 这个是问题的关键，有些样本无正样本，就会直接崩溃
    # loss_fnc = torch.nn.MSELoss()
    '''
    # loss_weight = torch.tensor([1, 10], dtype=torch.float32)
    loss_fnc = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learn_rate)
    model.train()
    epoch_loss = 0
    for batch_data in data:
        batch_loss = 0
        for item in batch_data:
            primary_node_target = item.antibody_dgl.ndata['label'].long(
            ).reshape(-1)
            pred_primary = model(
                item.antibody_dgl, item.antibody_matrix, item.antigen_dgl, item.antigen_matrix)
            loss = loss_fnc(pred_primary, primary_node_target)
            batch_loss += loss
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        epoch_loss += batch_loss
    print(epoch_loss.detach().numpy())
    return model


def test(model, data):
    predict = []
    target = []
    for item in data:
        proba = model(item.dgl_data, item.mol_feat).detach().numpy()
        pred = 0 if proba < 0.5 else 1
        predict.append(pred)
        target.append(int(item.target[0]))
    auc = metrics.roc_auc_score(target, predict)
    print(auc)


if __name__ == '__main__':
    pass
