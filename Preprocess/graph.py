import networkx as nx
from Preprocess import feature
# from matplotlib import pyplot as plt
import numpy as np


class NXGraph():
    def __init__(self, name, data):
        self.name = name
        self.res_list = data[0]
        self.feature = feature.Feature(self.res_list)
        self.edges = data[1]
        self.node_label = self.get_node_label(data[2], len(self.res_list))
        self.graph_data, self.matrix = self.generate_graph(
            name, self.feature.all_feature, self.node_label, self.edges)
        self.basic_infos = [sum(self.node_label), len(
            self.node_label), self.graph_data.number_of_edges()]  # 反应自环+单向边

    def get_node_label(self, indexs, length):
        result = [0 for _ in range(length)]
        for index in indexs:
            result[index] = 1
        return result

    def generate_graph(self, name, node_feature, node_label, edges):
        assert len(node_feature) == len(node_label)
        graph = nx.Graph()
        matrix = np.zeros((len(node_label), len(node_label)))
        for index in range(len(node_feature)):
            graph.add_node(
                index, feature=node_feature[index], label=node_label[index])
            graph.add_edge(index, index)  # 添加自环
            matrix[index][index] = 1
        for v0, v1 in edges:
            graph.add_edge(v0, v1)  # 无向图，加两次无效
            # graph.add_edge(v1, v0)
            matrix[v0][v1] = 1
            matrix[v1][v0] = 1
        # 可以看出图很密集
        '''
        nx.draw(graph)
        plt.show()
        '''
        return graph, matrix


if __name__ == '__main__':
    pass
