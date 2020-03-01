import random
import dgl
import os
import networkx as nx
import torch


class BatchGenerator():
    def __init__(self, data, batch_size):
        self.data = data
        random.shuffle(self.data)
        self.batch_size = batch_size if batch_size != -1 else len(self.data)
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        try:
            self.data[self.index+self.batch_size-1]  # 用于检查是否越界
            b_data = self.data[self.index:self.index+self.batch_size]
        except IndexError:
            raise StopIteration()
        self.index += self.batch_size
        return b_data


class SingleSample():
    def __init__(self, data):
        self.antibody = data[0]
        self.antigen = data[1]
        self.label = torch.tensor(data[2], dtype=torch.float32)


class StructData():
    def __init__(self, args):
        self.pos_data = self.read_file(args.process_path)
        self.neg_data = self.expand_neg_data(self.pos_data, args.neg_rate)
        self.all_data_with_label = [
            pos+[1] for pos in self.pos_data]+[neg+[0] for neg in self.neg_data]
        self.all_structed_data = [SingleSample(
            data)for data in self.all_data_with_label]
        self.train_data, self.valid_data, self.test_data = self.split(
            self.all_structed_data, args.split_rate)

    def read_file(self, path):
        result = []
        names = os.listdir(path)
        for name in names:
            nxgraph_antibody = nx.read_gpickle(
                os.path.join(path, name, 'antibody.gpickle'))
            nxgraph_antigen = nx.read_gpickle(
                os.path.join(path, name, 'antigen.gpickle'))
            dgl_antibody = self.nx_to_dgl(nxgraph_antibody)
            dgl_antigen = self.nx_to_dgl(nxgraph_antigen)
            result.append([dgl_antibody, dgl_antigen])
        return result

    def expand_neg_data(self, pos_data, rate):
        size_pos = len(pos_data)
        size_neg = int(rate*size_pos)
        neg_data = []
        for _ in range(size_neg):
            choose_antibody = random.randint(0, size_pos-1)
            choose_antigen = random.randint(0, size_pos-2)
            if choose_antigen >= choose_antibody:
                choose_antigen += 1
            neg_data.append([pos_data[choose_antibody][0],
                             pos_data[choose_antigen][1]])
        return neg_data

    def nx_to_dgl(self, nx_graph: nx.Graph):
        dgl_graph = dgl.DGLGraph()
        for node in nx_graph.nodes:
            data = nx_graph.nodes[node]
            data['feature'] = torch.tensor(
                data['feature'], dtype=torch.float32).reshape(1, -1)
            data['label'] = torch.tensor(
                data['label'], dtype=torch.float32).reshape(1, -1)
            dgl_graph.add_nodes(1, data)
        for v0, v1 in nx_graph.edges:
            dgl_graph.add_edge(v0, v1)
            dgl_graph.add_edge(v1, v0)
        # print(dgl_graph.ndata['hidden'].shape)
        return dgl_graph

    def split(self, data, split_rate):
        nums = list(map(int, split_rate.split(' ')))
        sums = sum(nums)
        rates = [item/sums for item in nums]
        random.shuffle(data)  # 必须做初始的随机，因为原始数据分布不均匀
        cut1 = int(rates[0]*len(data))
        cut2 = int((rates[0]+rates[1])*len(data))
        return data[:cut1], data[cut1:cut2], data[cut2:]
