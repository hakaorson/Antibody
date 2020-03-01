from torch import nn
import torch
import dgl


class SingleGCN(nn.Module):
    def __init__(self, in_feats, out_feats, activate, dropout, bias=True):
        super().__init__()
        self.full_conn = torch.nn.Linear(in_feats, out_feats, bias=bias)
        self.dropout = torch.nn.Dropout() if dropout else None
        self.activation = torch.nn.ReLU() if activate else None

    def gcn_msg(self, edge):
        msg_data = edge.src['hidden']
        return {'message': msg_data}

    def gcn_reduce(self, node):
        reduce_data = torch.mean(node.mailbox['message'], 1)
        return {'reduce': reduce_data}

    def gcn_node(self, node):
        node_data = node.data['reduce']
        update_data = self.full_conn(node_data)
        return {'hidden': update_data}

    def forward(self, dgl_data: dgl.DGLGraph):
        hidden = dgl_data.ndata['hidden']
        if self.dropout:
            hidden = self.dropout(hidden)
        dgl_data.update_all(self.gcn_msg, self.gcn_reduce, self.gcn_node)
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
        self.gcn_process = MultiGCN(args)
        self.read_out = ReadOut(args)
        self.match_value = GraphFeatMatch(args)

    def forward(self, dgl_primary, dgl_secondary):
        # 注意要初始化，因为hidden已经更改了
        dgl_primary.ndata['hidden'] = dgl_primary.ndata['feature']
        dgl_secondary.ndata['hidden'] = dgl_secondary.ndata['feature']
        gcn_primary = self.gcn_process(dgl_primary)
        gcn_secondary = self.gcn_process(dgl_secondary)
        feat_primary = self.read_out(gcn_primary)
        feat_secondaty = self.read_out(gcn_secondary)
        distance = self.match_value(feat_primary, feat_secondaty)
        return distance


if __name__ == '__main__':
    pass