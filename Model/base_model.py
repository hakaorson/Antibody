from torch import nn
import torch
import dgl
from dgl.nn.pytorch import GraphConv


class SingleGCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()
        self.acti = nn.ReLU()
        # self.gcn = GraphConv(in_feats, out_feats, activation=self.acti)
        self.gcn_weight = nn.Linear(in_feats, out_feats)

    def msg_gcn(self, edge):
        msg = edge.src['hidden']
        return {'msg': msg}

    def reduce_gcn(self, node):
        reduce = torch.sum(node.mailbox['msg'], 1)
        return {'reduce': reduce}

    def forward(self, dgl_data: dgl.DGLGraph, matrix):
        digits = dgl_data.ndata['hidden']
        neibors = dgl_data.ndata['neibors']
        rsqt_neibors = torch.rsqrt(neibors)
        digits = digits*rsqt_neibors
        digits = torch.mm(matrix, digits)
        digits = digits*rsqt_neibors
        digits = self.gcn_weight(digits)
        dgl_data.ndata['hidden'] = digits
        dgl_data.ndata['stack'] = torch.cat(
            (dgl_data.ndata['hidden'], dgl_data.ndata['stack']), -1)
        '''
        hidden = dgl_data.ndata['hidden']
        rsqt_neibors = torch.rsqrt(dgl_data.ndata['neibors'])

        digits = hidden*rsqt_neibors
        dgl_data.ndata['hidden'] = digits
        dgl_data.update_all(self.msg_gcn, self.reduce_gcn)
        digits = dgl_data.ndata['reduce']
        digits = hidden*rsqt_neibors

        digits = self.gcn_weight(digits)
        dgl_data.ndata['hidden'] = digits

        dgl_data.ndata['stack'] = torch.cat(
            (dgl_data.ndata['hidden'], dgl_data.ndata['stack']), -1)
        temp_0 = hidden.detach().numpy()
        temp_1 = dgl_data.ndata['hidden'].detach().numpy()
        temp_2 = dgl_data.ndata['stack'].detach().numpy()
        '''
        return dgl_data


class GCNProcess(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.GCNlayers = nn.ModuleList()
        self.GCNlayers.append(
            SingleGCN(args.gcn_input_size, args.gcn_hidden_size))
        self.GCNlayers.append(
            SingleGCN(args.gcn_hidden_size, args.gcn_output_size))
        args.gcn_stack_size = args.gcn_input_size+args.gcn_hidden_size * \
            (args.hidden_layer+1)+args.gcn_output_size
        # self.GCNlayers.append(SingleGCN(32, 32))

    def forward(self, dgl_data, matrix):
        dgl_data = self.init_dgl_data(dgl_data)
        for model in self.GCNlayers:
            dgl_data = model(dgl_data, matrix)
        return dgl_data

    def init_dgl_data(self, dgl_data):
        dgl_data.ndata['hidden'] = dgl_data.ndata['feature']
        dgl_data.ndata['stack'] = dgl_data.ndata['feature']
        return dgl_data


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.attention_process = nn.Linear(
            args.gcn_stack_size, args.gcn_stack_size)
        self.attention_acti = torch.nn.ReLU()

    def forward(self, node_prim, node_second):
        prim = self.attention_process(node_prim)
        second = self.attention_process(node_second)
        matrix = self.attention_acti(torch.mm(
            prim, torch.transpose(second, 1, 0)))
        matrix_sqsq = matrix.pow(2).sum(1).sqrt()

        matrix_alpha = torch.transpose(
            torch.div(torch.transpose(matrix, 1, 0), matrix_sqsq), 1, 0)

        # matrix_alpha = torch.div(matrix, matrix_sqsq, 0)
        result = torch.mm(matrix_alpha, second)
        return result


class Predict(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.predictor = nn.Linear(
            args.gcn_stack_size*2, args.label_num, bias=True)
        # self.acti = nn.Sigmoid()

    def forward(self, prim, contex):
        digits = self.predictor(torch.cat((prim, contex), -1))
        # digits = self.acti(digits)
        # print('digits', digits.detach().numpy()[0])
        return digits


'''
class Predict_no_att(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.predictor = nn.Linear(92, 2, bias=True)
        # self.acti = nn.Sigmoid()

    def forward(self, prim):
        digits = self.predictor(prim)
        # digits = self.acti(digits)
        # print('digits', digits.detach().numpy()[0])
        return digits
'''


class SimpleModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.gcn_process = GCNProcess(args)
        self.attention = Attention(args)
        self.predict = Predict(args)

    def forward(self, dgl_primary, mat_prim, dgl_secondary, mat_send):
        node_primary = self.gcn_process(
            dgl_primary, mat_prim).ndata['stack']
        node_secondary = self.gcn_process(
            dgl_secondary, mat_send).ndata['stack']
        node_context = self.attention(node_primary, node_secondary)
        node_predict = self.predict(node_primary, node_context)
        # node_predict = self.pred_no_att(node_primary)
        return node_predict


if __name__ == '__main__':
    pass
