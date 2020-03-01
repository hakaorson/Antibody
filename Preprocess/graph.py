import networkx as nx
from Preprocess import feature
from matplotlib import pyplot as plt


class NXGraph():
    def __init__(self, args, name, data):
        self.name = name
        self.res_list = data[0]
        self.feature = feature.Feature(self.res_list)
        self.edges = data[1]
        self.node_label = self.get_node_label(data[2], len(self.res_list))
        self.graph_data = self.generate_graph(
            name, self.feature.all_feature, self.node_label, self.edges)

    def get_node_label(self, indexs, length):
        result = [0 for _ in range(length)]
        for index in indexs:
            result[index] = 1
        return result

    def generate_graph(self, name, node_feature, node_label, edges):
        assert len(node_feature) == len(node_label)
        graph = nx.Graph()
        for index in range(len(node_feature)):
            graph.add_node(
                index, feature=node_feature[index], label=node_label[index])
        for v0, v1 in edges:
            graph.add_edge(v0, v1)
        # 可以看出图很密集
        '''
        nx.draw(graph)
        plt.show()
        '''
        return graph


if __name__ == '__main__':
    pass
