import json
def gen_uniprot_dict_all_new():
    uniprot_dict_all = json.load(open('./output_info/uniprot_dict_all.json','r'))
    my_dict = dict() # key1:uniprot id key2:pdb_id value:chain_id
    for pdb_id, diction in uniprot_dict_all.items():
        for chain,uniprot_id in diction.items():
            if uniprot_id not in my_dict.keys():
                my_dict[uniprot_id] = dict()
            my_dict[uniprot_id][pdb_id] = chain
            
    ec_dict = json.load(open('./output_info/enzyme_commission_new_uniprots.json','r'))
    for pdb_id,uniprot_id in ec_dict.items():
        chain = pdb_id[-1]
        if len(uniprot_id) == 0:
            continue
        if uniprot_id[0] not in my_dict.keys():
            my_dict[uniprot_id[0]] = dict()
        my_dict[uniprot_id[0]][pdb_id] = chain
        
    json.dump(my_dict,open('./output_info/uniprot_pdb_chain_dict.json','w'))

if __name__ == "__main__":
    gen_uniprot_dict_all_new()