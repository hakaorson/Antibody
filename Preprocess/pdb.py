from Bio.PDB import PDBParser
from Bio.PDB import NeighborSearch
from Bio.PDB import Selection
CDRS = {'L': [[24, 34], [50, 56], [89, 97]],
        'H': [[26, 32], [52, 56], [95, 102]]}


def read_pdb(pdb_file_path):
    parse = PDBParser()
    model = parse.get_structure('<name>', pdb_file_path)
    return model[0]


def get_chains_antibody(path):
    pdb_model = read_pdb(path)
    dict_keys = list(pdb_model.child_dict.keys())
    if 'H' in dict_keys and 'L' in dict_keys:
        chain_h = pdb_model['H']
        chain_l = pdb_model['L']
        return chain_h, chain_l
    else:
        return None, None


def get_chain_antigen(path):
    pdb_model = read_pdb(path)
    dict_keys = list(pdb_model.child_dict.keys())
    assert(len(dict_keys) > 0)
    chain_a = pdb_model[dict_keys[0]]
    return chain_a


def extract_chain(chain):
    atom_list = Selection.unfold_entities(chain, 'A')
    res_list = Selection.unfold_entities(chain, 'R')
    search = NeighborSearch(atom_list)
    return res_list, search


def get_cdr_id():
    cdr_ids = {}
    for key in CDRS.keys():
        cdr_ids[key] = []
        for low, high in CDRS[key]:
            for index in range(low-1, high+2):  # 两边阔一个
                cdr_ids[key].append(index)
    return cdr_ids


def get_resbyid(res_list, id_list):
    result = []
    for res in res_list:
        if res.id[1] in id_list:
            result.append(res)
    return result


def get_append_res_list(res_list, search_list, dist):
    append_res = set()
    for res in res_list:
        for search in search_list:
            for atom in res.get_unpacked_list():
                nerbor_res = search.search(atom.coord, dist, 'R')
                append_res = append_res | set(nerbor_res)
    return append_res


def get_graph_edges(res_list, search_list, dist):
    result = set()
    res_dict = {}
    for index, res in enumerate(res_list):
        res_dict[res] = index
    for index, res in enumerate(res_list):
        for search in search_list:
            for atom in res.get_unpacked_list():
                nerbor_res_list = search.search(atom.coord, dist, 'R')
                for nerbor_res in nerbor_res_list:
                    if nerbor_res in res_dict.keys() and res_dict[nerbor_res] > index:
                        result.add((index, res_dict[nerbor_res]))
    return result


def get_antibody_graph(args, chain_h, chain_l):
    all_cdr_ids = get_cdr_id()
    res_list_h, search_h = extract_chain(chain_h)
    res_list_l, search_l = extract_chain(chain_l)
    cdr_res_list = get_resbyid(
        res_list_h, all_cdr_ids['H'])+get_resbyid(res_list_l, all_cdr_ids['L'])
    append_res_list = get_append_res_list(
        cdr_res_list, [search_h, search_l], args.expand_dist)
    all_res_list = list(set(cdr_res_list) | append_res_list)
    all_edges = get_graph_edges(
        all_res_list, [search_h, search_l], args.graph_dist)
    return all_res_list, all_edges, [search_h, search_l]


def get_antigen_graph(args, chain):
    res_list, search = extract_chain(chain)
    edges = get_graph_edges(res_list, [search], args.graph_dist)
    return res_list, edges, [search]


def find_interface_between_ag(args, res_list, searchs):
    result = set()
    for index, res in enumerate(res_list):
        for search in searchs:
            for atom in res.get_unpacked_list():
                if(len(search.search(atom.coord, args.interface_dist, 'R'))):
                    result.add(index)
                    break
    return result


def pair_process(args, pdb_path_antibody, pdb_path_antigen):
    chain_h, chain_l = get_chains_antibody(pdb_path_antibody)
    chain_a = get_chain_antigen(pdb_path_antigen)
    if chain_h and chain_l:
        res_list_antibody, edges_antibody, searchs_antibody = get_antibody_graph(
            args, chain_h, chain_l)
        res_list_antigen, edges_antigen, searchs_antigen = get_antigen_graph(
            args, chain_a)
        inter_of_antibody = find_interface_between_ag(
            args, res_list_antibody, searchs_antigen)
        inter_of_antigen = find_interface_between_ag(
            args, res_list_antigen, searchs_antibody)
        return (res_list_antibody, edges_antibody, inter_of_antibody), (res_list_antigen, edges_antigen, inter_of_antigen)
    else:
        return None, None


if __name__ == '__main__':
    pass
