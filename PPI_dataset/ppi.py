import numpy as np
import os
import json
import networkx as nx
from networkx.readwrite import json_graph


def storation(path, graph_antibody, graph_antigen):
    os.makedirs(path, exist_ok=True)
    nx.write_gpickle(graph_antibody, os.path.join(path, 'antibody.gpickle'))
    nx.write_gpickle(graph_antigen, os.path.join(path, 'antigen.gpickle'))


class NXGraph():
    def __init__(self, name, data):
        self.name = name
        self.origin_data = data
        self.nodes_map = self.get_nodemap(list(self.origin_data.nodes()))
        self.graph_data, self.matrix = self.new_graph(
            self.origin_data, self.nodes_map)

    def get_nodemap(self, node_list):
        node_set = set(node_list)
        assert len(node_set) == len(node_list)
        map_dict = {}
        for index, node in enumerate(node_list):
            map_dict[node] = index
        return map_dict

    def new_graph(self, old_graph, node_map):
        new_graph = nx.Graph()
        matrix = np.zeros((old_graph.number_of_nodes(),
                           old_graph.number_of_nodes()))
        for node in old_graph.nodes():
            node_index = node_map[node]
            new_graph.add_node(node_index)
            new_graph.nodes[node_index]['feature'] = old_graph.nodes[node]['feature']
            new_graph.nodes[node_index]['all_label'] = old_graph.nodes[node]['all_label']
            new_graph.nodes[node_index]['label'] = old_graph.nodes[node]['label']
            new_graph.add_edge(node_index, node_index)
            matrix[node_index][node_index] = 1
        for v0, v1 in old_graph.edges:
            v0_index = node_map[v0]
            v1_index = node_map[v1]
            new_graph.add_edge(v0_index, v1_index)
            matrix[v0_index][v1_index] = 1
            matrix[v1_index][v0_index] = 1
        return new_graph, matrix


def main():
    dataset_dir = 'Data/ppi'
    process_ppi_dir = 'Data/process_ppi'
    G = json_graph.node_link_graph(
        json.load(open(dataset_dir + "/ppi-G.json")))
    all_nodes = G.nodes()
    labels = json.load(open(dataset_dir + "/ppi-class_map.json"))
    feats = np.load(dataset_dir + "/ppi-feats.npy")
    for node in all_nodes:
        G.nodes[node]['feature'] = feats[node]
        G.nodes[node]['all_label'] = labels[str(node)]
        G.nodes[node]['label'] = labels[str(node)][0]
    sub_graphs = []
    for con in nx.connected_components(G):
        if len(con) > 10:
            temp_graph = G.subgraph(con)
            sub_graphs.append(temp_graph)
    for index, graph in enumerate(sub_graphs):
        nx_graph_0 = NXGraph(str(index), graph)
        nx_graph_1 = NXGraph(str(index), graph)
        storation(process_ppi_dir+'/'+str(index), nx_graph_0, nx_graph_1)
        print(str(index))


if __name__ == '__main__':
    pass
