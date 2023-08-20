import requests
from bs4 import BeautifulSoup
# from selenium import webdriver
import time
import random
import re
import os
from tqdm import tqdm
import json
def get_uniprots(url_root, pdbs, chain : bool = False):
    uniprot_dict = {}
    for i in tqdm(range(len(pdbs))):
        if i == 'index' or i == 'readme':
            continue
        uniprot_dict[pdbs[i].strip()] = []
        time.sleep(random.uniform(0.1,1))
        chain_id = ''
        if chain:
            cmplx, chain_id = pdbs[i].strip().split('-')
            url = url_root + cmplx
        else:
            url = url_root + str(pdbs[i])
        e = ''
        while e == '':
            try:
                e = requests.get(url)
                break
            except:
                print("Connection refused by the server..")
                print("Let me sleep for 5 seconds")
                print("ZZzzzz...")
                time.sleep(5)
                print("Was a nice sleep, now let me continue...")
                continue
        # e = requests.get(url)
        soup = BeautifulSoup(e.text, 'html.parser')
        if chain:
            tables = soup.find_all('table')
            for table in tables:
                s_t = BeautifulSoup(str(table), 'html.parser')
                hyperlinks = s_t.find_all('a')
                chains = []
                for hyp in hyperlinks:
                    if 'href' in hyp.attrs and "/sequence/" + cmplx in hyp.attrs['href'] and hyp.text != 'Expand':
                        chains.append(hyp.text)
                    if 'href' in hyp.attrs and "https://www.uniprot.org/uniprot/" in hyp.attrs['href']:
                        chains = list(set(chains))
                        for item in chains:
                            if 'auth' in item:
                                auth_id = item.split('[')[1].split(' ')[1][:-1]
                                if chain_id == auth_id:
                                    uniprot_dict[pdbs[i].strip()].append(hyp.text)
                            else:
                                if chain_id == item:
                                    uniprot_dict[pdbs[i].strip()].append(hyp.text)
            # break
        else:
            hyperlinks = soup.find_all('a')
            for hyp in hyperlinks:
                if 'href' in hyp.attrs and "https://www.uniprot.org/uniprot/" in hyp.attrs['href']:
                    id = hyp.text
                    uniprot_dict[pdbs[i]].append(id)
    return uniprot_dict
def parse_info_PDBBind():
    query_url = 'http://www.rcsb.org/structure/'
    protein_ligand_dir = './data/PDBBind/refined-set/'
    protein_protein_dir = './data/PDBBind/PP'

    pl_pdbs = os.listdir(protein_ligand_dir)
    pp_pdbs = os.listdir(protein_protein_dir)
    print("Protein Ligand dataset size:", len(pl_pdbs) - 2)
    print("Protein Protein dataset size:", len(pp_pdbs) - 2)
    pp_pdbs = [i.removesuffix('.ent.pdb') for i in pp_pdbs]
    protein_ligand_dict = get_uniprots(query_url, pl_pdbs)
    protein_protein_dict = get_uniprots(query_url, pp_pdbs) 
    with open('./output_info/protein_ligand_uniprots.json','w') as f:
        json.dump(protein_ligand_dict, f)
    with open('./output_info/protein_protein_uniprots.json','w') as f:
        json.dump(protein_protein_dict, f)   

def read_file(file_dir):
    with open(file_dir, 'r') as f:
        lines = f.readlines()
    return lines

def parse_info_EnzymeCommission():
    query_url = 'http://www.rcsb.org/structure/'
    train_chains= read_file('./datasets/EnzymeCommissionNew/nrPDB-EC_train.txt')
    val_chains = read_file('./datasets/EnzymeCommissionNew/nrPDB-EC_valid.txt')
    test_chains = read_file('./datasets/EnzymeCommissionNew/nrPDB-EC_test.txt')
    chains_total = list(set(train_chains + val_chains + test_chains))
    print("Total length of EnzymeCommission:", len(chains_total))
    enzymecommission_dict = get_uniprots(query_url, chains_total, chain=True)
    with open('./output_info/enzyme_commission_new_uniprots.json', 'w') as f:
        json.dump(enzymecommission_dict, f)
def parse_info_fold():
    query_url = 'http://www.rcsb.org/structure/'
    filenames = ['./datasets/HomologyTAPE/train.txt','./datasets/HomologyTAPE/val.txt','./datasets/HomologyTAPE/test.txt']
    raw_chains = []
    chains_total = []
    for filename in filenames:
        f =open(filename,'r')
        for line in f.readlines():
            item = line.split('\t')[0]
            raw_chains.append(item)
            chains_total.append(item[1:5].upper()+"-"+item[-2].upper())
    
    chains_total = list(set(chains_total))
    print("chains total:",chains_total)
    print("Total length of Homology:", len(chains_total))
    fold_dict = get_uniprots(query_url, chains_total, chain=True)
    fold_raw_dict = {}
    print("fold raw dict:",fold_dict)
    for raw_chain in raw_chains:
        chain = raw_chain[1:5].upper()+"-"+raw_chain[-2].upper()
        if chain in fold_dict.keys():
            fold_raw_dict[raw_chain] = fold_dict[chain]
    with open('./output_info/Homology_uniprots.json', 'w') as f:
        json.dump(fold_raw_dict, f)
    
def parse_info_GeneOntology():
    query_url = 'http://www.rcsb.org/structure/'
    train_chains= read_file('./data/GeneOntology/nrPDB-GO_train.txt')
    val_chains = read_file('./data/GeneOntology/nrPDB-GO_valid.txt')
    test_chains = read_file('./data/GeneOntology/nrPDB-GO_test.txt')
    chains_total = list(set(train_chains + val_chains + test_chains))
    print("Total length of GeneOntology:", len(chains_total))
    enzymecommission_dict = get_uniprots(query_url, chains_total, chain=True)
    with open('./output_info/gene_ontology_uniprots.json', 'w') as f:
        json.dump(enzymecommission_dict, f)

def gen_info_list(json_dir):
    with open(json_dir,'r') as f:
        info_dict = json.load(f)
    info_list = []
    for k,v in info_dict.items():
        for id in v:
            info_list.append(id)
    # print(info_list)
    return info_list

if __name__ == '__main__':
    parse_info_fold()
    # if not os.path.exists('./output_info/protein_ligand_uniprots.json') or not os.path.exists('./output_info/protein_protein_uniprots.json'):
    #     print("Parsing PDBBind from PDBbank...")
    #     parse_info_PDBBind()
    #     print("Done!")
    # else:
    #     print("PDBBind info already exists, skip parsing...")
    # if not os.path.exists('./output_info/enzyme_commission_uniprots.json'):
    #     print("Parsing EnzymeCommission from PDBbank...")
    #     parse_info_EnzymeCommission()
    #     print("Done!")
    # else:
    #     print("EC info already exists, skip parsing...")
    # if not os.path.exists('./output_info/gene_ontology_uniprots.json'):
    #     print("Parsing GeneOntology from PDBbank...")
    #     parse_info_GeneOntology()
    #     print("Done!")
    # else:
    #     print("GO info already exists, skip parsing...")
        
    # pl_list = gen_info_list('./output_info/protein_ligand_uniprots.json')
    # pp_list = gen_info_list('./output_info/protein_protein_uniprots.json')
    # ec_list = gen_info_list('./output_info/enzyme_commission_uniprots.json')
    # ge_list = gen_info_list('./output_info/gene_ontology_uniprots.json')
    # print("Number of proteins in protein-ligand binding:", len(pl_list))
    # print("Number of proteins in enzyme_commision prediction:", len(ec_list))    
    # print("Number of proteins in protein-protein interaction:", len(pp_list))
    # print("Number of proteins in gene_ontology interaction:", len(ge_list))
    # intersection_pl_ec = list(set(pl_list) & set(ec_list))
    # intersection_pp_ec = list(set(pp_list) & set(ec_list))
    # intersection_pl_ge = list(set(pl_list) & set(ge_list))
    # intersection_pp_ge = list(set(pp_list) & set(ge_list))
    # intersection_ec_ge = list(set(ec_list) & set(ge_list))
    # intersection_pl_pp = list(set(pl_list) & set(pp_list))
    # intersection_pl_pp_ec = list(set(pl_list) & set(pp_list) & set(ec_list))
    # intersection_pl_pp_ge = list(set(pl_list) & set(pp_list) & set(ge_list))
    # intersection_pl_pp_ge_ec = list(set(pl_list) & set(pp_list) & set(ec_list) & set(ge_list))
    # # print("Size of pl set:", len(set(pl_list)))
    # # print("Size of pp set:", len(set(pp_list)))
    # # intersection = list(set(pl_list) & set(pp_list))
    # print("Calculating shared PDB ids of these datasets...")
    # print("Number of proteins in common(pl&EC):", len(intersection_pl_ec))
    # print("Number of proteins in common(pp&EC):", len(intersection_pp_ec))
    # print("Number of proteins in common(pl&GO):", len(intersection_pl_ge))
    # print("Number of proteins in common(pp&GO):", len(intersection_pp_ge))
    # print("Number of proteins in common(EC&GO):", len(intersection_ec_ge))
    # print("Number of proteins in common(pl&pp):", len(intersection_pl_pp))
    # print("Number of proteins in common(pl&pp&EC):", len(intersection_pl_pp_ec))
    # print("Number of proteins in common(pl&pp&GO):", len(intersection_pl_pp_ge))
    # print("Number of proteins in common(pl&pp&GO&EC):", len(intersection_pl_pp_ge_ec))