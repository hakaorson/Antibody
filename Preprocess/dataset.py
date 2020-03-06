from Preprocess import pdb
from Preprocess import graph
import os
import networkx as nx
import numpy as np
import multiprocessing as multi
import shutil
multi.set_start_method('spawn', True)


def get_complex_names(path):
    file_names = os.listdir(path)
    names = set()
    for file_name in file_names:
        if 'ab.pdb' in file_name:
            names.add(file_name.split('ab.pdb')[0])
    return names


def storation(path, graph_antibody, graph_antigen):
    os.makedirs(path, exist_ok=True)
    nx.write_gpickle(graph_antibody, os.path.join(path, 'antibody.gpickle'))
    nx.write_gpickle(graph_antigen, os.path.join(path, 'antigen.gpickle'))


def generate_dataset(args):
    o_path = args.origin_path
    p_path = args.process_path
    complex_names = get_complex_names(o_path)
    shutil.rmtree(p_path)
    os.makedirs(p_path, exist_ok=True)
    result = []
    pool = multi.Pool(processes=10)
    for name in complex_names:
        result.append(pool.apply_async(single_process, (args, name,)))
    pool.close()
    pool.join()
    antibody_infos, antigen_infos = [], []
    for item in result:
        value = item.get()
        if value is not None:
            antibody_infos.append(value[0])
            antigen_infos.append(value[1])
    antibody_infos = np.sum(np.array(antibody_infos), axis=0)
    antigen_infos = np.sum(np.array(antigen_infos), axis=0)
    print(antibody_infos)
    print(antigen_infos)


def single_process(args, name):
    single_name = name.split('.')[0]
    path_antibody = os.path.join(args.origin_path, name+'ab.pdb')
    path_antigen = os.path.join(args.origin_path, name+'ag.pdb')
    antibody_data, antigen_data = pdb.pair_process(
        args, path_antibody, path_antigen)
    if antibody_data and antigen_data:
        antibody_graph = graph.NXGraph(single_name+'_antibody', antibody_data)
        antigen_graph = graph.NXGraph(single_name+'_antigen', antigen_data)
        storation(os.path.join(args.process_path, single_name),
                  antibody_graph, antigen_graph)
        print(single_name, '-Success')
        return [antibody_graph.basic_infos, antigen_graph.basic_infos]
    else:
        print(single_name, '-False')
        return None


if __name__ == '__main__':
    pass
