import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import time
import random
import re
import os
from tqdm import tqdm
import json
import pandas as pd
from Bio.PDB import PDBParser
class PDBWebParser(object):
    def __init__(self, root_dir: str = './data/MultiTask'):
        super(PDBWebParser, self).__init__()
        self.root_dir = root_dir    
        self.ec_root = './datasets/EnzymeCommission/all'
        self.go_root = './datasets/GeneOntology/all'
        self.lba_root = './datasets/PDBbind/refined-set'
        self.pp_root = './datasets/PDBbind/PP'
        self.ec_files = os.listdir(self.ec_root)
        self.go_files = os.listdir(self.go_root)
        self.lba_files = os.listdir(self.lba_root)
        self.pp_files = os.listdir(self.pp_root)
    def find_structure(self, item):
        if '-' in item:
            if item+'.pdb' in self.ec_files:
                return os.path.join(self.ec_root, item+'.pdb'), -1
            elif item+'.pdb' in self.go_files:
                return os.path.join(self.go_root, item+'.pdb'), -1
        else:
            if item + '.ent.pdb' in self.pp_files:
                return os.path.join(self.pp_root, item+'.ent.pdb'), -1
            elif item in self.lba_files:
                protein_dir = os.path.join(self.lba_root, item, item + "_protein.pdb")
                ligand_dir = os.path.join(self.lba_root, item, item + '_ligand.mol2')
                return protein_dir, ligand_dir
        print(item)
        return -1, -1
    def get_uniprots(self, url_root, pdbs):
        uniprot_dict = {}
        for i in tqdm(range(len(pdbs))):
            uniprot_dict[pdbs[i].strip()] = {}
            time.sleep(random.uniform(0.1,1))
            if '-' in pdbs[i]:
                cmplx, _ = pdbs[i].strip().split('-')
                url = url_root + cmplx
            else:
                cmplx = str(pdbs[i]).upper()
                url = url_root + cmplx
            protein_dir, _ = self.find_structure(pdbs[i])
            p = PDBParser(QUIET=True)
            try:
                structure = p.get_structure(pdbs[i], protein_dir)
            except:
                continue
            model = structure[0]
            chains = list(model.get_chains())
            chain_ids = [chain.id for chain in chains if chain.id != ' ']
            # print(pdb_ids[i], chain_ids)
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
            soup = BeautifulSoup(e.text, 'html.parser')
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
                                if auth_id in chain_ids:
                                    uniprot_dict[pdbs[i].strip()][auth_id] = hyp.text
                            else:
                                if item in chain_ids:
                                    uniprot_dict[pdbs[i].strip()][item] = hyp.text
        return uniprot_dict

if __name__ == '__main__':
    with open('./datasets/MultiTask/train_all.txt', 'r') as f:
        train_pdbs = f.readlines()
        train_pdbs = [i.strip() for i in train_pdbs]
    with open('./datasets/MultiTask/val.txt', 'r') as f:
        val_pdbs = f.readlines()
        val_pdbs = [i.strip() for i in val_pdbs]
    with open('./datasets/MultiTask/test.txt', 'r') as f:
        test_pdbs = f.readlines()
        test_pdbs = [i.strip() for i in test_pdbs]
        
    pdb_ids = list(train_pdbs + val_pdbs + test_pdbs)
    query_url = 'http://www.rcsb.org/structure/'
    parser = PDBWebParser()
    uniprot_dict = parser.get_uniprots(query_url, pdb_ids)
    if not os.path.exists('./output_info/uniprot_dict_all.json'):
        with open('./output_info/uniprot_dict_all.json', 'w') as f:
            json.dump(uniprot_dict, f)
    else:
        print("Uniprot dict already exists, skip saving...")