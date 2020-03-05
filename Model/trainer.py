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
        for graph_item in batch_data:
            primary_node_target = graph_item.antibody_dgl.ndata['label'].long(
            ).reshape(-1)
            pred_primary = model(
                graph_item.antibody_dgl, graph_item.antibody_matrix, graph_item.antigen_dgl, graph_item.antigen_matrix)
            loss = loss_fnc(pred_primary, primary_node_target)
            if torch.isnan(loss):
                print(graph_item.antibody_name)
            else:
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
    for graph_item in data:
        graph_proba = model(graph_item.antibody_dgl, graph_item.antibody_matrix,
                            graph_item.antigen_dgl, graph_item.antigen_matrix).detach().numpy()
        for node_item in graph_proba:
            pred = 0 if node_item[0] > node_item[1] else 1
            predict.append(pred)
        graph_target = list(
            graph_item.antibody_dgl.ndata['label'].detach().numpy())
        target.extend(graph_target)
    auc = metrics.roc_auc_score(target, predict)
    print(auc)


if __name__ == '__main__':
    pass
