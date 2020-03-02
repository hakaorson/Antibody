import torch
from sklearn import metrics


def train(args, model: torch.nn.Module, data):
    loss_fnc = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for batch_data in data:
        loss = 0
        for item in batch_data:
            match_target = item.label.reshape(-1)
            antibody_node_target = item.antibody.ndata['label']
            antigen_node_target = item.antigen.ndata['label']
            logits = model(item.antibody, item.antigen)
            loss += loss_fnc(logits, match_target)
            print(logits.detach().numpy(), match_target.detach().numpy())
        print(loss.detach().numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
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
