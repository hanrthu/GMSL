# This script is for processing the dataset to form a new multi-task dataset with a unform labeling code
# Datasets PDBBind(Protein-Ligand, Protein-Protein), EnzymeCommission, GeneOntology
import os
import json
from typing import List
import random
import pandas as pd

def gen_domain_info():
    input_file = open('./output_info/scop.txt','r')
    output_file = open('./output_info/domain_info.txt','w')
    domain_dict = dict()
    for line in input_file.readlines():
        if line.startswith('>'):
            fold_item = line.split(' ')[0].strip('>')
            domain = line.split(' ')[2].strip('(').strip(')')
            domain_dict[fold_item] = domain
    json.dump(domain_dict,output_file)        
            

def cal_complex_all(json_dirs):
    complex_list = []
    for json_dir in json_dirs:
        with open(json_dir, 'r') as f:
            info_dict = json.load(f)
        for i in info_dict:
            if '-' in i:
                id = i.split('-')[0].lower()
            complex_list.append(id)
    print(len(list(set(complex_list))))

def gen_train_test_ids_new_new(complex_dict,ec_dict,go_dict,reaction_dict,fold_dict):
    train_list = []
    # print(complex_dict)
    all_list = []
    for k, v in complex_dict.items():
        all_list.extend(v) # k is uniprot id v is pdb ids
    all_list = list(set(all_list))
    for k, v in complex_dict.items():
        # print(k, v)
        # 由于每个蛋白可能有多个Uniprot，可以根据某一个Uniprot不在的情况找出训练集的样本
        if k not in ec_dict or k not in go_dict or k not in reaction_dict or k not in fold_dict:
            train_list.extend(v) # train_list:哪些pdb_id中有uniprot缺少ec或者go标注
    #取反即可获得标注完整的数据集
    test_list = [i for i in all_list if i not in train_list] # test_list:哪些pdb_id中所有uniprot有完整的ec和go标注
    test_uniprots = []
    for k, v in complex_dict.items():
        for id in v:
            if id in test_list and k not in test_uniprots:
                test_uniprots.append(k)
    # print("Uniprots:", len(test_uniprots))
    # print("Train:", len(train_list), len(list(set(train_list))))
    # print("Test:", len(test_list), len(list(set(test_list))))
    # print("All:", len(list(set(test_list + train_list))))
    # print(list(set(train_list)))
    return list(set(train_list)), list(set(test_list)), list(set(test_uniprots)) 

def gen_train_test_ids_new(complex_dict,ec_dict,go_dict,reaction_dict):
    train_list = []
    # print(complex_dict)
    all_list = []
    for k, v in complex_dict.items():
        all_list.extend(v) # k is uniprot id v is pdb ids
    all_list = list(set(all_list))
    for k, v in complex_dict.items():
        # print(k, v)
        # 由于每个蛋白可能有多个Uniprot，可以根据某一个Uniprot不在的情况找出训练集的样本
        if k not in ec_dict or k not in go_dict or k not in reaction_dict:
            train_list.extend(v) # train_list:哪些pdb_id中有uniprot缺少ec或者go标注
    #取反即可获得标注完整的数据集
    test_list = [i for i in all_list if i not in train_list] # test_list:哪些pdb_id中所有uniprot有完整的ec和go标注
    test_uniprots = []
    for k, v in complex_dict.items():
        for id in v:
            if id in test_list and k not in test_uniprots:
                test_uniprots.append(k)
    # print("Uniprots:", len(test_uniprots))
    # print("Train:", len(train_list), len(list(set(train_list))))
    # print("Test:", len(test_list), len(list(set(test_list))))
    # print("All:", len(list(set(test_list + train_list))))
    # print(list(set(train_list)))
    return list(set(train_list)), list(set(test_list)), list(set(test_uniprots))        

def gen_train_test_ids(complex_dict, ec_dict, go_dict):
    train_list = []
    # print(complex_dict)
    all_list = []
    for k, v in complex_dict.items():
        all_list.extend(v) # k is uniprot id v is pdb ids
    all_list = list(set(all_list))
    for k, v in complex_dict.items():
        # print(k, v)
        # 由于每个蛋白可能有多个Uniprot，可以根据某一个Uniprot不在的情况找出训练集的样本
        if k not in ec_dict or k not in go_dict:
            train_list.extend(v) # train_list:哪些pdb_id中有uniprot缺少ec或者go标注
    #取反即可获得标注完整的数据集
    test_list = [i for i in all_list if i not in train_list] # test_list:哪些pdb_id中所有uniprot有完整的ec和go标注
    test_uniprots = []
    for k, v in complex_dict.items():
        for id in v:
            if id in test_list and k not in test_uniprots:
                test_uniprots.append(k)
    # print("Uniprots:", len(test_uniprots))
    # print("Train:", len(train_list), len(list(set(train_list))))
    # print("Test:", len(test_list), len(list(set(test_list))))
    # print("All:", len(list(set(test_list + train_list))))
    # print(list(set(train_list)))
    return list(set(train_list)), list(set(test_list)), list(set(test_uniprots))
        
            
def gen_protein_property_uniprots(json_dir: str, single=True):
    with open(json_dir, 'r') as f:
        info_dict = json.load(f)
    uniprot_dict = {}
    # print(len(info_dict))
    info_dict_new = {}
    for k, v in info_dict.items():
        if single:
            if len(v) != 1:
                continue
            else:
                uniprot_id = v[0]
                info_dict_new[k] = [uniprot_id]   
                # 只记录同一个uniprot id的所有pdb_id
                if uniprot_id not in uniprot_dict:     
                    uniprot_dict[uniprot_id] = k
                # else:
                #     uniprot_dict[uniprot_id].append(k)
        else:
            # Remove items whose uniprot ids are larger than 3 or equal zero.
            if len(v) == 0 or len(v) > 3:
                continue
            else:
                info_dict_new[k] = v
                for uniprot_id in v:
                    if uniprot_id not in uniprot_dict:
                        uniprot_dict[uniprot_id] = [k]
                    else:
                        uniprot_dict[uniprot_id].append(k)
    # print("Unitest:", len(uniprot_dict), len(info_dict_new))
    return uniprot_dict, info_dict_new

def gen_reaction_uniprots():
    "生成uniprot_dict, info_dict"
    input_json_dir = "./output_info/enzyme_commission_new_uniprots.json"
    output_json_dir = "./output_info/reaction_uniprots.json"
    all_uniprots_dict = json.load(open(input_json_dir))
    new_uniprots_dict = {}
    chain_function_file = open('./datasets/ProtFunc/chain_functions.txt','r')
    for line in chain_function_file.readlines():
        pdbid = line.split(',')[0]
        pdbid = pdbid.replace('.','-').upper()
        try:
            new_uniprots_dict[pdbid] = all_uniprots_dict[pdbid]
        except:
            print("{} not found".format(pdbid))
    json.dump(new_uniprots_dict,open(output_json_dir,'w'))
    return gen_protein_property_uniprots(output_json_dir)
    
def save_dataset_info(dir, complex_list):
    with open(dir, 'w') as f:
        for complex in complex_list:
            f.write(complex + '\n')
        f.close()

def gen_ec_labels():
    root_dir = './datasets/EnzymeCommission/nrPDB-EC_annot.tsv'
    with open(root_dir, 'r') as f:
        lines = f.readlines()
    ec_classes = lines[1].strip().split('\t')
    label_dict = {}
    pdb_annot_dict = {}
    label_id = 0
    for label in ec_classes:
        if label not in label_dict:
            label_dict[label] = label_id
            label_id += 1
    for item in lines[3:]:
        pdb_id, annotations = item.split('\t')
        annotations_list = annotations.strip().split(',')
        pdb_annot_dict[pdb_id] = [label_dict[annot] for annot in annotations_list]
    return pdb_annot_dict
    # print("Number of classes in task {} is {}".format('EnzymeCommission', label_id))
def gen_reaction_labels():
    root_dir = './datasets/ProtFunc/chain_functions.txt'
    pdb_uniprot_dict = json.load(open('./output_info/reaction_uniprots.json'))
    f = open(root_dir)
    lines = f.readlines()
    pdb_annot_dict = {}
    for line in lines:
        pdbid,label = line.split(',')
        pdbid = pdbid.replace('.','-').upper()
        if(pdbid in pdb_uniprot_dict.keys()):
            uniprot_ids = pdb_uniprot_dict[pdbid]
            pdb_annot_dict[pdbid] = [int(label.strip('\n'))]
    return pdb_annot_dict
def gen_fold_labels():
    root_dir = "./datasets/HomologyTAPE/"
    label_map_file = open(root_dir+'class_map.txt')
    label_map = dict()
    pdb_annot_dict = {}
    pdb_uniprot_dict = json.load(open('./output_info/Homology_uniprots.json'))
    for line in label_map_file.readlines():
        label_map[line.split('\t')[0]] = int(line.split('\t')[-1].strip('\n'))
    for filename in ['train.txt','test_family.txt','test_superfamily.txt','test_fold.txt','val.txt']:
        file = open(root_dir+filename)
        lines = file.readlines()
        for line in lines:
            item = line.split('\t')[0]
            label = line.strip('\n').split('\t')[-1]
            if item in pdb_uniprot_dict.keys():
                pdb_annot_dict[item] = label_map[label]
    return pdb_annot_dict
            
def process_reaction_labels():
    "生成reaction单任务对应的uniformed_labels.json"
    root_dir = './datasets/ProtFunc/chain_functions.txt'
    pdb_uniprot_dict = json.load(open('./output_info/reaction_uniprots.json'))
    f = open(root_dir)
    lines = f.readlines()
    pdb_annot_dict = {}
    for line in lines:
        chain,label = line.split(',')
        chain = chain.replace('.','-').upper()
        if(chain in pdb_uniprot_dict.keys()):
            uniprot_ids = pdb_uniprot_dict[chain]
            pdb_annot_dict[chain] = {"uniprots":uniprot_ids,",ec":[-1],"reaction":[[int(label.strip('\n'))]],"go":[-1],"ppi":-1,"lba":-1}
    json.dump(pdb_annot_dict,open('./datasets/ProtFunc/uniformed_labels.json','w'))
# def get_full_annotation(go_uniprot_dict):
#     root_dir = './datasets/GeneOntology/nrPDB-GO_annot.tsv'
#     with open(root_dir, 'r') as f:
#         lines = f.readlines()
#     go_classes_molecular_functions = lines[1].strip().split('\t')
#     go_classes_biological_process = lines[5].strip().split('\t')
#     go_classes_cellular_component = lines[9].strip().split('\t')
#     for k, v in go_uniprot_dict.items():
        

def gen_go_labels(go_uniprot_dict):
    root_dir = './datasets/GeneOntology/nrPDB-GO_annot.tsv'
    with open(root_dir, 'r') as f:
        lines = f.readlines()
    go_classes_molecular_functions = lines[1].strip().split('\t')
    go_classes_biological_process = lines[5].strip().split('\t')
    go_classes_cellular_component = lines[9].strip().split('\t')
    
    label_dict = {'molecular_functions':{}, 'biological_process':{}, 'cellular_component':{}}
    pdb_annot_dict = {}
    label_id_molecular = 0
    label_id_biological = 0
    label_id_cellular = 0
    full_ids = []
    full_uniprot_dict = {}
    for label in go_classes_molecular_functions:
        if label not in label_dict['molecular_functions']:
            label_dict['molecular_functions'][label] = label_id_molecular
            label_id_molecular += 1
    for label in go_classes_biological_process:
        if label not in label_dict['biological_process']:
            label_dict['biological_process'][label] = label_id_biological
            label_id_biological += 1
    for label in go_classes_cellular_component:
        if label not in label_dict['cellular_component']:
            label_dict['cellular_component'][label] = label_id_cellular
            label_id_cellular += 1
    for item in lines[13:]:
        pdb_id, molecular, biological, cellular = item.split('\t')
        molecular_list  = molecular.strip().split(',')
        biological_list = biological.strip().split(',')
        cellular_list = cellular.strip().split(',')
        # print(molecular_list)
        # 列表中会包含一些空的信息
        pdb_annot_dict[pdb_id] = {'molecular_functions':[label_dict['molecular_functions'][annot] for annot in molecular_list if annot != ''],
                                  'biological_process':[label_dict['biological_process'][annot] for annot in biological_list if annot != ''],
                                  'cellular_component':[label_dict['cellular_component'][annot] for annot in cellular_list if annot != '']}
        if pdb_annot_dict[pdb_id]['molecular_functions'] != [] and pdb_annot_dict[pdb_id]['biological_process'] != [] and pdb_annot_dict[pdb_id]['cellular_component'] != []:
            full_ids.append(pdb_id)
    for k, v in go_uniprot_dict.items():
        if v in full_ids:
            full_uniprot_dict[k] = v
    print("Number of classes in task {} is {}".format('molecular_functions', label_id_molecular+1))
    print("Number of classes in task {} is {}".format('biological_process', label_id_biological+1))
    print("Number of classes in task {} is {}".format('cellular_component', label_id_cellular+1))
    return pdb_annot_dict, full_uniprot_dict
    # print(pdb_annot_dict)


def gen_lba_labels():
    root_dir = './datasets/PDBbind/refined-set/index/INDEX_general_PL_data.2020'
    res = {}
    with open(root_dir) as f:
        for line in f:
            if '#' in line:
                continue
            cont = line.strip().split()
            if len(cont) < 5:
                continue
            code, pk = cont[0], cont[3]
            res[code] = float(pk)
    # print("LBA:", len(res))
    return res

def gen_ppi_labels():
    root_dir = './datasets/protein_protein/pp_affinity.xlsx'
    pp_info = pd.read_excel(root_dir, header=1)
    pdb_codes = pp_info['PDB code']
    res = {}
    for i, code in enumerate(pdb_codes):
        res[code] = pp_info['pKd pKi pIC50'][i]
    # print("PPI:", len(res))
    return res
def gen_label_new_new(pdb_ids, ec_labels, go_labels,reaction_labels,fold_labels, ppi_labels, lba_labels, ec_uniprot_dict, go_uniprot_dict,reaction_uniprot_dict, fold_uniprot_dict,ec_info_dict, go_info_dict,reaction_info_dict,fold_info_dict, pp_info_dict, pl_info_dict):
    uniform_labels = {}
    for pdb_id in pdb_ids:
        if '-' in pdb_id:
            uniprots = ""
            if pdb_id in ec_labels:
                uniprots = ec_info_dict[pdb_id]
                single_ec_label = [ec_labels[pdb_id]]
            else:
                single_ec_label = [-1]
            if pdb_id in go_labels:
                uniprots = go_info_dict[pdb_id]
                single_go_label = [go_labels[pdb_id]]
            else:
                single_go_label = [-1]
            if pdb_id in reaction_labels and pdb_id in reaction_info_dict:
                uniprots = reaction_info_dict[pdb_id] 
                single_reaction_label = [reaction_labels[pdb_id]]
            else:
                single_reaction_label = [-1]
            
            if len(uniprots) == 1 :
                # print(uniprots)
                # 查找这条链是否有fold label
                single_fold_label = [-1]
                if (uniprots[0] in fold_uniprot_dict):
                    fold_items_list = fold_uniprot_dict[uniprots[0]]
                    if (len(fold_items_list)>0):
                        fold_label_dict = dict()
                        for fold_item in fold_items_list:
                            fold_label_dict[fold_item] = fold_labels[fold_item]
                        single_fold_label = [fold_label_dict]
        elif len(pdb_id) == 7:
            pass        
        else:
            single_ec_label = []
            single_go_label = []
            single_reaction_label = []
            single_fold_label = []
            if pdb_id in pp_info_dict:
                uniprots = pp_info_dict[pdb_id]
            elif pdb_id in pl_info_dict:
                uniprots = pl_info_dict[pdb_id]
            else:
                raise NotImplementedError
            for uniprot_id in uniprots:
                if uniprot_id in ec_uniprot_dict:
                    ec_label = ec_labels[ec_uniprot_dict[uniprot_id]]
                else:
                    ec_label = -1
                if uniprot_id in go_uniprot_dict:
                    go_label = go_labels[go_uniprot_dict[uniprot_id]]
                else:
                    go_label = -1
                if uniprot_id in reaction_uniprot_dict:
                    reaction_label = reaction_labels[reaction_uniprot_dict[uniprot_id]]
                else:
                    reaction_label = -1
                if uniprot_id in fold_uniprot_dict:
                    fold_item_list = fold_uniprot_dict[uniprot_id]
                    fold_label = {}
                    for fold_item in fold_item_list:
                        fold_label[fold_item] = fold_labels[fold_item]
                else:
                    fold_label = -1
                single_ec_label.append(ec_label)
                single_go_label.append(go_label)
                single_reaction_label.append(reaction_label)
                single_fold_label.append(fold_label)
        if pdb_id in ppi_labels:
            ppi_label = ppi_labels[pdb_id]
        else:
            ppi_label = -1
        if pdb_id in lba_labels:
            lba_label = lba_labels[pdb_id]
        else:
            lba_label = -1
        uniform_labels[pdb_id] = {"uniprots":uniprots, "ec": single_ec_label, "go": single_go_label, "reaction":single_reaction_label,"fold:":single_fold_label,"ppi": ppi_label, "lba": lba_label}
    return uniform_labels

def gen_label_new(pdb_ids, ec_labels, go_labels,reaction_labels, ppi_labels, lba_labels, ec_uniprot_dict, go_uniprot_dict,reaction_uniprot_dict, ec_info_dict, go_info_dict,reaction_info_dict, pp_info_dict, pl_info_dict):
    uniform_labels = {}
    for pdb_id in pdb_ids:
        if '-' in pdb_id:
            if pdb_id in ec_labels:
                uniprots = ec_info_dict[pdb_id]
                single_ec_label = [ec_labels[pdb_id]]
            else:
                single_ec_label = [-1]
            if pdb_id in go_labels:
                uniprots = go_info_dict[pdb_id]
                single_go_label = [go_labels[pdb_id]]
            else:
                single_go_label = [-1]
            if pdb_id in reaction_labels and pdb_id in reaction_info_dict:
                uniprots = reaction_info_dict[pdb_id] 
                single_reaction_label = [reaction_labels[pdb_id]]
            else:
                single_reaction_label = [-1]
                
        else:
            single_ec_label = []
            single_go_label = []
            single_reaction_label = []
            if pdb_id in pp_info_dict:
                uniprots = pp_info_dict[pdb_id]
            elif pdb_id in pl_info_dict:
                uniprots = pl_info_dict[pdb_id]
            else:
                raise NotImplementedError
            for uniprot_id in uniprots:
                if uniprot_id in ec_uniprot_dict:
                    ec_label = ec_labels[ec_uniprot_dict[uniprot_id]]
                else:
                    ec_label = -1
                if uniprot_id in go_uniprot_dict:
                    go_label = go_labels[go_uniprot_dict[uniprot_id]]
                else:
                    go_label = -1
                if uniprot_id in reaction_uniprot_dict:
                    reaction_label = reaction_labels[reaction_uniprot_dict[uniprot_id]]
                else:
                    reaction_label = -1
                single_ec_label.append(ec_label)
                single_go_label.append(go_label)
                single_reaction_label.append(reaction_label)
        if pdb_id in ppi_labels:
            ppi_label = ppi_labels[pdb_id]
        else:
            ppi_label = -1
        if pdb_id in lba_labels:
            lba_label = lba_labels[pdb_id]
        else:
            lba_label = -1
        uniform_labels[pdb_id] = {"uniprots":uniprots, "ec": single_ec_label, "go": single_go_label, "reaction":single_reaction_label,"ppi": ppi_label, "lba": lba_label}
    return uniform_labels

def gen_label(pdb_ids, ec_labels, go_labels, ppi_labels, lba_labels, ec_uniprot_dict, go_uniprot_dict, ec_info_dict, go_info_dict, pp_info_dict, pl_info_dict):
    uniform_labels = {}
    for pdb_id in pdb_ids:
        if '-' in pdb_id:
            if pdb_id in ec_labels:
                uniprots = ec_info_dict[pdb_id]
                single_ec_label = [ec_labels[pdb_id]]
            else:
                single_ec_label = [-1]
            if pdb_id in go_labels:
                uniprots = go_info_dict[pdb_id]
                single_go_label = [go_labels[pdb_id]]
            else:
                single_go_label = [-1]
        else:
            single_ec_label = []
            single_go_label = []
            if pdb_id in pp_info_dict:
                uniprots = pp_info_dict[pdb_id]
            elif pdb_id in pl_info_dict:
                uniprots = pl_info_dict[pdb_id]
            else:
                raise NotImplementedError
            for uniprot_id in uniprots:
                if uniprot_id in ec_uniprot_dict:
                    ec_label = ec_labels[ec_uniprot_dict[uniprot_id]]
                else:
                    ec_label = -1
                if uniprot_id in go_uniprot_dict:
                    go_label = go_labels[go_uniprot_dict[uniprot_id]]
                else:
                    go_label = -1
                single_ec_label.append(ec_label)
                single_go_label.append(go_label)
        if pdb_id in ppi_labels:
            ppi_label = ppi_labels[pdb_id]
        else:
            ppi_label = -1
        if pdb_id in lba_labels:
            lba_label = lba_labels[pdb_id]
        else:
            lba_label = -1
        uniform_labels[pdb_id] = {"uniprots":uniprots, "ec": single_ec_label, "go": single_go_label, "ppi": ppi_label, "lba": lba_label}
    return uniform_labels
def process_ppi_data():
    print("Start processing original datasets to a ppi task dataset")
    json_dir = './output_info/protein_protein_uniprots.json'
    
    # pp_uniprot_dict, pp_info_dict = gen_protein_property_uniprots(json_dir, single=False)
    ppi_labels = gen_ppi_labels()
    
    pp_uniprot_dict, pp_info_dict = gen_protein_property_uniprots(json_dir, single=False)
    print(pp_info_dict)
    ppi_labels = {k:{"uniprots":pp_info_dict[k],"ec":[-1 for _ in range(len(pp_info_dict[k]))],"go":[-1 for _ in range(len(pp_info_dict[k]))],"ppi":v,"lba":-1} for k,v in ppi_labels.items() if k in pp_info_dict.keys()}
    whole_set = [k for k in ppi_labels.keys()]
    train_set = whole_set[:int(len(whole_set)*0.8)]
    test_set = whole_set[int(len(whole_set)*0.8):int(len(whole_set)*0.9)]
    valid_set = whole_set[int(len(whole_set)*0.9):]
    os.makedirs('./datasets/ppi', exist_ok=True)
    save_dataset_info('./datasets/ppi/train_all.txt', train_set)
    save_dataset_info('./datasets/ppi/train.txt', train_set)
    save_dataset_info('./datasets/ppi/val.txt', valid_set)
    save_dataset_info('./datasets/ppi/test.txt', test_set)

    with open('./datasets/ppi/uniformed_labels.json', 'w') as f:
            json.dump(ppi_labels, f)
def process_ec_data():
    json_dir = "./output_info/enzyme_commission_new_uniprots.json"
    ec_uniprot_dict, ec_info_dict = gen_protein_property_uniprots(json_dir)
    print(ec_info_dict)
    ec_labels = gen_ec_labels()
    ec_labels = { k:{"uniprots":ec_info_dict[k],"ec":[v],"go":[-1],"ppi":-1,"lba":-1} for k,v in ec_labels.items() if k in ec_info_dict.keys()}
    print("label:", len(ec_labels))
    whole_set = [k for k in ec_labels.keys()]
    train_set = whole_set[:int(len(whole_set)*0.8)]
    test_set = whole_set[int(len(whole_set)*0.8):int(len(whole_set)*0.9)]
    valid_set = whole_set[int(len(whole_set)*0.9):]
    os.makedirs('./datasets/ec', exist_ok=True)
    save_dataset_info('./datasets/EnzymeCommissionNew/train_all.txt', train_set)
    save_dataset_info('./datasets/EnzymeCommissionNew/train.txt', train_set)
    save_dataset_info('./datasets/EnzymeCommissionNew/val.txt', valid_set)
    save_dataset_info('./datasets/EnzymeCommissionNew/test.txt', test_set)

    with open('./datasets/EnzymeCommissionNew/uniformed_labels.json', 'w') as f:
            json.dump(ec_labels, f)

def process_multi_data_new_new():
    '''
    生成有reaction label和fold label的label.json和train_all.txt,train.txt,test,txt,val.txt
    '''    
    print("Start processing original datasets to a multitask dataset")
    root_dir = './output_info/'
    json_files = ['enzyme_commission_uniprots.json', 'gene_ontology_uniprots.json', 'protein_protein_uniprots.json', 'protein_ligand_uniprots.json','reaction_uniprots.json','Homology_uniprots.json']
    json_dirs = [os.path.join(root_dir, json_file) for json_file in json_files]
    # cal_complex_all(json_dirs)
    # uniprot_dict: {Uniprot id: [List of pdb ids]}
    # info_dict: {PDB id: [List of uniprot ids]}
    ec_uniprot_dict, ec_info_dict = gen_protein_property_uniprots(json_dirs[0])
    go_uniprot_dict, go_info_dict = gen_protein_property_uniprots(json_dirs[1])
    pp_uniprot_dict, pp_info_dict = gen_protein_property_uniprots(json_dirs[2], single=False)
    pl_uniprot_dict, pl_info_dict = gen_protein_property_uniprots(json_dirs[3], single=False)
    reaction_uniprot_dict,reaction_info_dict = gen_protein_property_uniprots(json_dirs[4])
    fold_uniprot_dict,fold_info_dict = gen_protein_property_uniprots(json_dirs[5],single=False)
    # print(fold_uniprot_dict)
   
    # exit(1)
    # ec_uniprot_dict: key:uniprot_id value:[pdb-chain_id,...]
    # ec_labels: {1ABE-A:128}
    # fold_labels: {d1rqwa:b.25}
    print("Number of samples pl, pp, ec, go,reaction:", len(pl_info_dict), len(pp_info_dict), len(ec_info_dict), len(go_info_dict),len(reaction_info_dict))
    ec_labels = gen_ec_labels()
    go_labels, go_full_uniprot_dict = gen_go_labels(go_uniprot_dict)
    ppi_labels = gen_ppi_labels()
    lba_labels = gen_lba_labels()
    reaction_labels = gen_reaction_labels()
    fold_labels = gen_fold_labels()
    
    train_list_pp, test_list_pp, test_uniprots_1 = gen_train_test_ids_new_new(pp_uniprot_dict, ec_uniprot_dict, go_full_uniprot_dict,reaction_dict=reaction_uniprot_dict,fold_dict=fold_uniprot_dict)
    train_list_pl, test_list_pl, test_uniprots_2 = gen_train_test_ids_new_new(pl_uniprot_dict, ec_uniprot_dict, go_full_uniprot_dict,reaction_dict=reaction_uniprot_dict,fold_dict=fold_uniprot_dict)
    test_list_all = list(set(test_list_pl + test_list_pp))
    test_uniprots_all = list(set(test_uniprots_1 + test_uniprots_2))
    
    random.shuffle(test_list_pl)
    random.shuffle(test_list_pp)
    
    full_test_list = test_list_pl[int(0.6*len(test_list_pl)):] + test_list_pp[int(0.6*len(test_list_pp)):]
    full_val_list = test_list_pl[int(0.2*len(test_list_pl)): int(0.6*(len(test_list_pl)))] + test_list_pp[int(0.2*len(test_list_pp)): int(0.6*len(test_list_pp))]
    full_train_list = test_list_pl[: int(0.2*len(test_list_pl))] + test_list_pp[: int(0.2*len(test_list_pp))]

    train_list_ec = [ec_uniprot_dict[i] for i in ec_uniprot_dict if i not in test_uniprots_all]
    train_list_go = [go_uniprot_dict[i] for i in go_uniprot_dict if i not in test_uniprots_all]
    train_list_reaction = [reaction_uniprot_dict[i] for i in reaction_uniprot_dict if i not in test_uniprots_all]
    # fold_items = [fold_uniprot_dict[i] for i in fold_uniprot_dict if i in test_uniprots_all]
    
    train_list_all = list(set(train_list_pl + train_list_pp + train_list_ec + train_list_go +train_list_reaction+ full_train_list))
    print("Train List:", len(train_list_ec), len(train_list_go),len(train_list_reaction),len(train_list_all))
    # train/val/test.txt contains samples of full labels, while train_all.txt contains labals with partial labels and samples in train.txt.
    print("Process finished, saving information into ./dataset/MultiTaskNewNew/")
    if os.path.exists('./datasets/MultiTaskNewNew/train_all.txt') and os.path.exists('./datasets/MultiTask/tmp/train.txt'):
        print("File already exists, skip saving split information...")
    else:
        print("Saving split information...")
        os.makedirs('./datasets/MultiTaskNewNew', exist_ok=True)
        save_dataset_info('./datasets/MultiTaskNewNew/train_all.txt', train_list_all)
        save_dataset_info('./datasets/MultiTaskNewNew/train.txt', full_train_list)
        save_dataset_info('./datasets/MultiTaskNewNew/val.txt', full_val_list)
        save_dataset_info('./datasets/MultiTaskNewNew/test.txt', full_test_list)
    
    uniformed_label_dict = gen_label_new_new(train_list_all+full_test_list+full_val_list,ec_labels, go_labels,reaction_labels,fold_labels,ppi_labels, lba_labels,ec_uniprot_dict, go_uniprot_dict,reaction_uniprot_dict,fold_uniprot_dict,ec_info_dict,go_info_dict,reaction_info_dict,fold_info_dict,pp_info_dict,pl_info_dict)
    print("An example of the processed unified label:\n", full_test_list[0], ": ", uniformed_label_dict[full_test_list[0]])
    if os.path.exists('./datasets/MultiTaskNewNew/uniformed_labels.json'):
        print("File already exists, skip saving uniformed labels...")
    else:
        print('Generating uniformed labels...')
        with open('./datasets/MultiTaskNewNew/uniformed_labels.json', 'w') as f:
            json.dump(uniformed_label_dict, f)
        
def process_multi_data():
    print("Start processing original datasets to a multitask dataset")
    root_dir = './output_info/'
    json_files = ['enzyme_commission_uniprots.json', 'gene_ontology_uniprots.json', 'protein_protein_uniprots.json', 'protein_ligand_uniprots.json']
    json_dirs = [os.path.join(root_dir, json_file) for json_file in json_files]
    # cal_complex_all(json_dirs)
    # uniprot_dict: {Uniprot id: [List of pdb ids]}
    # info_dict: {PDB id: [List of uniprot ids]}
    ec_uniprot_dict, ec_info_dict = gen_protein_property_uniprots(json_dirs[0])
    go_uniprot_dict, go_info_dict = gen_protein_property_uniprots(json_dirs[1])
    pp_uniprot_dict, pp_info_dict = gen_protein_property_uniprots(json_dirs[2], single=False)
    pl_uniprot_dict, pl_info_dict = gen_protein_property_uniprots(json_dirs[3], single=False)
    reaction_uniprot_dict,reaction_info_dict = gen_reaction_uniprots()
    # ec_uniprot_dict: key:uniprot_id value:[pdb-chain_id,...]
    # ec_labels: {1ABE-A:128}
    # fold_labels: {d1rqwa:b.25}
    print("Number of samples pl, pp, ec, go,reaction:", len(pl_info_dict), len(pp_info_dict), len(ec_info_dict), len(go_info_dict),len(reaction_info_dict))
    ec_labels = gen_ec_labels()
    go_labels, go_full_uniprot_dict = gen_go_labels(go_uniprot_dict)
    ppi_labels = gen_ppi_labels()
    lba_labels = gen_lba_labels()
    reaction_labels = gen_reaction_labels()
    # print(len(ec_uniprot_dict), len(go_uniprot_dict), len(pp_uniprot_dict), len(pl_uniprot_dict))
    train_list_pp, test_list_pp, test_uniprots_1 = gen_train_test_ids_new(pp_uniprot_dict, ec_uniprot_dict, go_full_uniprot_dict,reaction_dict=reaction_uniprot_dict)
    train_list_pl, test_list_pl, test_uniprots_2 = gen_train_test_ids_new(pl_uniprot_dict, ec_uniprot_dict, go_full_uniprot_dict,reaction_dict=reaction_uniprot_dict)
    test_list_all = list(set(test_list_pl + test_list_pp))
    test_uniprots_all = list(set(test_uniprots_1 + test_uniprots_2))
    exit(1)
    # full_train_ratio = 0.2
    # full_val_ratio = 0.4
    # full_test_ratio = 0.4
    random.shuffle(test_list_pl)
    random.shuffle(test_list_pp)
    
    full_test_list = test_list_pl[int(0.6*len(test_list_pl)):] + test_list_pp[int(0.6*len(test_list_pp)):]
    full_val_list = test_list_pl[int(0.2*len(test_list_pl)): int(0.6*(len(test_list_pl)))] + test_list_pp[int(0.2*len(test_list_pp)): int(0.6*len(test_list_pp))]
    full_train_list = test_list_pl[: int(0.2*len(test_list_pl))] + test_list_pp[: int(0.2*len(test_list_pp))]

    train_list_ec = [ec_uniprot_dict[i] for i in ec_uniprot_dict if i not in test_uniprots_all]
    train_list_go = [go_uniprot_dict[i] for i in go_uniprot_dict if i not in test_uniprots_all]
    train_list_reaction = [reaction_uniprot_dict[i] for i in reaction_uniprot_dict if i not in test_uniprots_all]
    
    
    # train_list_pl:有pl label但是ec或者go缺点啥的pdb id训练集
    # train_list_pp:有pp label但是ec或者go缺点啥的训练集
    # train_list_ec:有ec但是没有go的训练集
    # train_list_go:有go但是没有ec的训练集
    train_list_all = list(set(train_list_pl + train_list_pp + train_list_ec + train_list_go +train_list_reaction+ full_train_list))
    print("Train List:", len(train_list_ec), len(train_list_go),len(train_list_reaction),len(train_list_all))
    # train/val/test.txt contains samples of full labels, while train_all.txt contains labals with partial labels and samples in train.txt.
    print("Process finished, saving information into ./dataset/MultiTaskNew/")
    if os.path.exists('./datasets/MultiTaskNew/train_all.txt') and os.path.exists('./datasets/MultiTask/tmp/train.txt'):
        print("File already exists, skip saving split information...")
    else:
        print("Saving split information...")
        os.makedirs('./datasets/MultiTaskNew', exist_ok=True)
        save_dataset_info('./datasets/MultiTaskNew/train_all.txt', train_list_all)
        save_dataset_info('./datasets/MultiTaskNew/train.txt', full_train_list)
        save_dataset_info('./datasets/MultiTaskNew/val.txt', full_val_list)
        save_dataset_info('./datasets/MultiTaskNew/test.txt', full_test_list)

    # uniformed_label_dict = gen_label(train_list_all+full_test_list+full_val_list, ec_labels, go_labels, ppi_labels, lba_labels, ec_uniprot_dict, go_uniprot_dict, ec_info_dict, go_info_dict, pp_info_dict, pl_info_dict)    
    uniformed_label_dict = gen_label_new(train_list_all+full_test_list+full_val_list,ec_labels, go_labels,reaction_labels,ppi_labels, lba_labels,ec_uniprot_dict, go_uniprot_dict,reaction_uniprot_dict,ec_info_dict,go_info_dict,reaction_info_dict,pp_info_dict,pl_info_dict)
    print("An example of the processed unified label:\n", full_test_list[0], ": ", uniformed_label_dict[full_test_list[0]])
    if os.path.exists('./datasets/MultiTaskNew/uniformed_labels.json'):
        print("File already exists, skip saving uniformed labels...")
    else:
        print('Generating uniformed labels...')
        with open('./datasets/MultiTaskNew/uniformed_labels.json', 'w') as f:
            json.dump(uniformed_label_dict, f)
        
if __name__ == '__main__':
    
    # process_multi_data_new_new()
    # process_ppi_data()
    # process_ec_data()
    # gen_reaction_labels()
    gen_domain_info()