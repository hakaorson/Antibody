from Preprocess import pdb
from Preprocess import graph
import os
import networkx as nx


def get_complex_names(path):
    file_names = os.listdir(path)
    names = set()
    for file_name in file_names:
        if 'ab_train_descriptors_N5' in file_name:
            names.add(file_name.split('ab_train_descriptors_N5')[0])
    return names


def storation(path, graph_antibody, graph_antigen):
    os.makedirs(path, exist_ok=True)
    nx.write_gpickle(graph_antibody, os.path.join(path, 'antibody.gpickle'))
    nx.write_gpickle(graph_antigen, os.path.join(path, 'antigen.gpickle'))


def generate_dataset(args):
    o_path = args.origin_path
    p_path = args.process_path
    complex_names = get_complex_names(o_path)
    os.makedirs(p_path, exist_ok=True)
    for complex_name in complex_names:
        single_name = complex_name.split('.')[0]
        path_antibody = os.path.join(o_path, complex_name+'ab.pdb')
        path_antigen = os.path.join(o_path, complex_name+'ag.pdb')
        antibody_data, antigen_data = pdb.pair_process(
            args, path_antibody, path_antigen)
        antibody_graph = graph.NXGraph(
            args, single_name+'_antibody', antibody_data)
        antigen_graph = graph.NXGraph(
            args, single_name+'_antigen', antigen_data)
        storation(os.path.join(p_path, single_name),
                  antibody_graph.graph_data, antigen_graph.graph_data)
        print(single_name)


if __name__ == '__main__':
    pass
