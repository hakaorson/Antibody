from torch import nn
import torch
import dgl
from dgl.nn.pytorch import GraphConv


class SingleGCN(nn.Module):
    def __init__(self, in_feats, out_feats, activate, dropout, bias=True):
        super().__init__()
        self.dropout = torch.nn.Dropout() if dropout else None
        self.activation = torch.nn.Sigmoid() if activate else None
        self.gcn = GraphConv(
            in_feats, out_feats, activation=self.activation, bias=bias)
        self.gcn_weight = nn.Linear(in_feats, out_feats)

    def forward(self, dgl_data: dgl.DGLGraph):
        hidden = dgl_data.ndata['hidden']
        # print(hidden.detach().numpy()[:3])
        if self.dropout:
            hidden = self.dropout(hidden)
        dgl_data.ndata['hidden'] = self.gcn(dgl_data, hidden)
        return dgl_data


class MultiGCN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(SingleGCN(
            args.gcn_input_size, args.gcn_hidden_size, args.activate, args.dropout, args.bias))
        for index in range(args.hidden_layer):
            self.gcn_layers.append(SingleGCN(
                args.gcn_hidden_size, args.gcn_hidden_size, args.activate, args.dropout, args.bias))
        self.gcn_layers.append(SingleGCN(
            args.gcn_hidden_size, args.gcn_output_size, args.activate, args.dropout, args.bias))

    def forward(self, dgl_data):
        for layer in self.gcn_layers:
            dgl_data = layer(dgl_data)
        return dgl_data


class ReadOut(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.full_conn = nn.Linear(args.gcn_output_size, args.graph_feat_size)
        self.activate = nn.Sigmoid()

    def forward(self, dgl_data: dgl.DGLGraph):
        graph_data = torch.mean(dgl_data.ndata['hidden'], 0)
        graph_feature = self.full_conn(graph_data)
        graph_feat_act = self.activate(graph_feature)
        return graph_feat_act


class NodeClass(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.full_conn = nn.Linear(args.gcn_output_size, 1)
        self.activate = nn.Sigmoid()

    def forward(self, dgl_data: dgl.DGLGraph):
        nodes_data = dgl_data.ndata['hidden']
        digits = self.full_conn(nodes_data)
        result = self.activate(digits)
        return result


class GraphFeatMatch(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.full_conn_1 = nn.Linear(
            args.graph_feat_size*2, args.graph_feat_size)
        self.full_conn_2 = nn.Linear(args.graph_feat_size, 1)
        self.activate = nn.Sigmoid()

    def forward(self, feat_p, feat_s):
        concat_feat = torch.cat((feat_p, feat_s), dim=-1)
        digits = self.full_conn_1(concat_feat)
        digits = self.full_conn_2(digits)
        result = self.activate(digits)
        return result


class SimpleModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gcn_process_primary = MultiGCN(args)
        self.gcn_process_secondary = MultiGCN(args)
        self.read_out_primary = ReadOut(args)
        self.read_out_secondary = ReadOut(args)
        self.node_primary = NodeClass(args)
        self.node_secondary = NodeClass(args)
        self.match_value = GraphFeatMatch(args)

    def forward(self, dgl_primary, dgl_secondary):
        # 注意要初始化，因为hidden已经更改了
        dgl_primary.ndata['hidden'] = dgl_primary.ndata['feature']
        dgl_secondary.ndata['hidden'] = dgl_secondary.ndata['feature']
        gcn_primary = self.gcn_process_primary(dgl_primary)
        gcn_secondary = self.gcn_process_secondary(dgl_secondary)

        feat_primary = self.read_out_primary(gcn_primary)
        feat_secondaty = self.read_out_secondary(gcn_secondary)
        distance = self.match_value(feat_primary, feat_secondaty)
        # distance = torch.nn.CosineSimilarity(feat_primary, feat_secondaty,0)

        node_pred_primary = self.node_primary(gcn_primary)
        node_pred_secondary = self.node_secondary(gcn_secondary)
        return distance, node_pred_primary, node_pred_secondary


if __name__ == '__main__':
    pass
