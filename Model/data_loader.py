import random
import dgl
import os
import networkx as nx
import torch
import numpy as np
import multiprocessing as multi
multi.set_start_method('spawn', True)


def single_read(path, name):
    nxgraph_antibody = nx.read_gpickle(
        os.path.join(path, name, 'antibody.gpickle'))
    
    nxgraph_antigen = nx.read_gpickle(
        os.path.join(path, name, 'antibody.gpickle'))
    # TODO 这里改了，试试是不是attention不同网络的问题
    for sub_node in nx.connected_components(nxgraph_antibody.graph_data):
        pass
    info = [[sum(nxgraph_antibody.node_label), len(nxgraph_antibody.node_label)], [sum(
        nxgraph_antigen.node_label), len(nxgraph_antigen.node_label)]]
    if info[0][0] == 0 or info[1][0] == 0 or info[0][0] == info[0][1] or info[1][0] == info[1][1]:
        print('ignore graph:', name, info)
        return None
    else:
        dgl_antibody = nx_to_dgl(nxgraph_antibody.graph_data)
        dgl_antigen = nx_to_dgl(nxgraph_antigen.graph_data)
        print('read graph:', name, info)
        return [[dgl_antibody, nxgraph_antibody.matrix, nxgraph_antibody.name], [dgl_antigen, nxgraph_antigen.matrix, nxgraph_antigen.name]]


def nx_to_dgl(nx_graph: nx.Graph):
    dgl_graph = dgl.DGLGraph()
    for node in nx_graph.nodes:
        nx_node_data = nx_graph.nodes[node]
        dgl_node_data = {}
        ner_num = len(list(nx_graph.neighbors(node)))
        dgl_node_data['feature'] = torch.tensor(
            nx_node_data['feature'], dtype=torch.float32).reshape(1, -1)
        dgl_node_data['label'] = torch.tensor(
            nx_node_data['label'], dtype=torch.float32).reshape(1, -1)
        dgl_node_data['neibors'] = torch.tensor(
            ner_num, dtype=torch.float32).reshape(1, -1)
        dgl_graph.add_nodes(1, dgl_node_data)
        dgl_graph.add_edge(node, node)  # 添加自环
    for v0, v1 in nx_graph.edges:
        if v0 < v1:
            dgl_graph.add_edge(v0, v1)
            dgl_graph.add_edge(v1, v0)
    # print(dgl_graph.ndata['hidden'].shape)
    return dgl_graph


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
        self.antibody_dgl = data[0][0]
        self.antigen_dgl = data[1][0]
        self.antibody_matrix = torch.tensor(data[0][1], dtype=torch.float32)
        self.antigen_matrix = torch.tensor(data[1][1], dtype=torch.float32)
        self.antibody_name = data[0][2]
        self.antigen_name = data[1][2]
        self.label = torch.tensor(data[-1], dtype=torch.float32)


class StructData():
    def __init__(self, args, datapath):
        self.pos_data = self.multi_read(datapath)
        self.neg_data = self.expand_neg_data(self.pos_data, args.neg_rate)
        self.pos_stuct_data = self.get_structed_data(self.pos_data, 1)
        self.neg_stuct_data = self.get_structed_data(self.neg_data, 0)
        self.all_structed_data = self.pos_stuct_data+self.neg_stuct_data
        self.train_data, self.valid_data, self.test_data = self.split(
            self.all_structed_data, args.split_rate)
        self.show_infos(self.pos_data)

    def multi_read(self, path):
        anti_names = os.listdir(path)
        result = []
        pool = multi.Pool(processes=10)
        for name in anti_names:
            result.append(pool.apply_async(single_read, (path, name,)))
        pool.close()
        pool.join()
        result_get = []
        for item in result:
            if item.get():
                result_get.append(item.get())
        return result_get

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

    def split(self, data, split_rate):
        nums = list(map(int, split_rate.split(' ')))
        sums = sum(nums)
        rates = [item/sums for item in nums]
        random.shuffle(data)  # 必须做初始的随机，因为原始数据分布不均匀
        cut1 = int(rates[0]*len(data))
        cut2 = int((rates[0]+rates[1])*len(data))
        return data[:cut1], data[cut1:cut2], data[cut2:]

    def get_structed_data(self, data, label):
        digits = [item+[label] for item in data]
        return [SingleSample(item)for item in digits]

    def show_infos(self, all_graphs):
        antibody_infos, antigen_infos = [], []
        for item in all_graphs:
            antibody, antigen = item[0][0], item[1][0]
            antibody_label = antibody.ndata['label'].detach().numpy()
            antigen_label = antigen.ndata['label'].detach().numpy()
            antibody_infos.append([int(sum(antibody_label)), len(
                antibody_label), antibody.number_of_edges()])
            antigen_infos.append([int(sum(antigen_label)), len(
                antigen_label), antigen.number_of_edges()])
        antibody_infos = np.sum(np.array(antibody_infos), axis=0)
        antigen_infos = np.sum(np.array(antigen_infos), axis=0)
        print(antibody_infos)
        print(antigen_infos)


if __name__ == '__main__':
    path = 'D:\\abs_sample_gaopan\\Data\\processed_data'
    name = '1A3R_L'
    single_read(path, name)
