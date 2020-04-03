from Bio.PDB import PDBParser
from Bio.PDB import NeighborSearch
from Bio.PDB import Selection


def read_pdb(path):
    parse = PDBParser()
    pdb_models = parse.get_structure('<name>', path)  # 这个name好像是无关紧要的
    single_model = pdb_models[0]
    chain_keys = list(single_model.child_dict.keys())
    single_chain = single_model[chain_keys[0]]
    atom_list = Selection.unfold_entities(single_chain, 'A')
    res_list = Selection.unfold_entities(single_chain, 'R')
    search = NeighborSearch(atom_list)
    return res_list, search


def get_graph_edges(res_list, search, dist):
    result = set()
    res_dict = {}
    for index, res in enumerate(res_list):
        res_dict[res] = index
    for index, res in enumerate(res_list):
        for atom in res.get_unpacked_list():
            nerbor_res_list = search.search(atom.coord, dist, 'R')
            for nerbor_res in nerbor_res_list:
                if nerbor_res in res_dict.keys() and res_dict[nerbor_res] > index:
                    result.add((index, res_dict[nerbor_res]))
    return result


def main():
    pdb_file_path = 'Data/origin_data/1A3R_L.H_P_ag.pdb'
    res_list, search = read_pdb(pdb_file_path)
    edges = get_graph_edges(res_list, search, 8)
    return res_list, edges


if __name__ == '__main__':
    res, edges = main()
    pass
