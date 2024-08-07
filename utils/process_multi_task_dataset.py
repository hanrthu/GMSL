# This script is for processing the dataset to form a new multi-task dataset with a unified labeling code
# Datasets PDBBind(Protein-Ligand, Protein-Protein), EnzymeCommission, GeneOntology
import os
import json
from typing import List
import random
import pandas as pd

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
        

def gen_train_test_ids(complex_dict, ec_dict, go_dict):
    train_list = []
    # print(complex_dict)
    all_list = []
    for k, v in complex_dict.items():
        all_list.extend(v)
    all_list = list(set(all_list))
    for k, v in complex_dict.items():
        # print(k, v)
        # 由于每个蛋白可能有多个Uniprot，可以根据某一个Uniprot不在的情况找出训练集的样本
        if k not in ec_dict or k not in go_dict:
            train_list.extend(v)
    #取反即可获得标注完整的数据集
    test_list = [i for i in all_list if i not in train_list]
    test_uniprots = []
    for k, v in complex_dict.items():
        for id in v:
            if id in test_list and k not in test_uniprots:
                test_uniprots.append(k)
    # print("Uniprots:", len(test_uniprots))
    print("Train:", len(train_list), len(list(set(train_list))))
    print("Test:", len(test_list), len(list(set(test_list))))
    print("All:", len(list(set(test_list + train_list))))
    # print(list(set(train_list)))
    return list(set(train_list)), list(set(test_list)), list(set(test_uniprots))
        
            
def gen_protein_property_uniprots(json_dir: str, single=True):
    with open(json_dir, 'r') as f:
        info_dict = json.load(f)
    uniprot_dict = {}
    # print(len(info_dict))
    info_dict_new = {}
    count = 0
    for k, v in info_dict.items():
        if single:
            if len(v) != 1:
                count += 1
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
    print("Abnormal:", count)
    return uniprot_dict, info_dict_new
    
def save_dataset_info(dir, complex_list):
    with open(dir, 'w') as f:
        for complex in complex_list:
            f.write(complex + '\n')
        f.close()

def gen_ec_labels():
    root_dir = '../datasets/EnzymeCommission/nrPDB-EC_annot.tsv'
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

# def get_full_annotation(go_uniprot_dict):
#     root_dir = './datasets/GeneOntology/nrPDB-GO_annot.tsv'
#     with open(root_dir, 'r') as f:
#         lines = f.readlines()
#     go_classes_molecular_functions = lines[1].strip().split('\t')
#     go_classes_biological_process = lines[5].strip().split('\t')
#     go_classes_cellular_component = lines[9].strip().split('\t')
#     for k, v in go_uniprot_dict.items():
        

def gen_go_labels(go_uniprot_dict):
    root_dir = '../datasets/GeneOntology/nrPDB-GO_annot.tsv'
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
    root_dir = '../datasets/PDBbind/refined-set/index/INDEX_general_PL_data.2020'
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
    root_dir = '../datasets/protein_protein/pp_affinity.xlsx'
    pp_info = pd.read_excel(root_dir, header=1)
    pdb_codes = pp_info['PDB code']
    res = {}
    for i, code in enumerate(pdb_codes):
        res[code] = pp_info['pKd pKi pIC50'][i]
    # print("PPI:", len(res))
    return res

def gen_label(pdb_ids, ec_labels, go_labels, ppi_labels, lba_labels, ec_uniprot_dict, go_uniprot_dict, ec_info_dict, go_info_dict, pp_info_dict, pl_info_dict):
    uniform_labels = {}
    for pdb_id in pdb_ids:
        if '-' in pdb_id:
            if pdb_id in ec_labels:
                if pdb_id in ec_info_dict:
                    uniprots = ec_info_dict[pdb_id]
                else:
                    uniprots = []
                single_ec_label = [ec_labels[pdb_id]]
            else:
                single_ec_label = [-1]
            if pdb_id in go_labels:
                if pdb_id in go_info_dict:
                    uniprots = go_info_dict[pdb_id]
                else:
                    uniprots = []
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
        # uniform_labels[pdb_id] = {"uniprots":uniprots, "ec": single_ec_label, "go": single_go_label, "ppi": ppi_label, "lba": lba_label}
        uniform_labels[pdb_id] = {"uniprots":uniprots, "ec": single_ec_label, "go": single_go_label, "ppi": ppi_label, "lba": lba_label}
    return uniform_labels
    # print(uniform_labels)
        
if __name__ == '__main__':
    print("Start processing original datasets to a multitask dataset")
    root_dir = '../output_info/'
    json_files = ['enzyme_commission_uniprots.json', 'gene_ontology_uniprots.json', 'protein_protein_uniprots.json', 'protein_ligand_uniprots.json']
    json_dirs = [os.path.join(root_dir, json_file) for json_file in json_files]
    # cal_complex_all(json_dirs)
    # uniprot_dict: {Uniprot id: [List of pdb ids]}
    # info_dict: {PDB id: [List of uniprot ids]}
    ec_uniprot_dict, ec_info_dict = gen_protein_property_uniprots(json_dirs[0])
    go_uniprot_dict, go_info_dict = gen_protein_property_uniprots(json_dirs[1])
    pp_uniprot_dict, pp_info_dict = gen_protein_property_uniprots(json_dirs[2], single=False)
    pl_uniprot_dict, pl_info_dict = gen_protein_property_uniprots(json_dirs[3], single=False)
    print("Number of samples pl, pp, ec, go:", len(pl_info_dict), len(pp_info_dict), len(ec_info_dict), len(go_info_dict))
    ec_labels = gen_ec_labels()
    go_labels, go_full_uniprot_dict = gen_go_labels(go_uniprot_dict)
    ppi_labels = gen_ppi_labels()
    lba_labels = gen_lba_labels()
    # print(len(ec_uniprot_dict), len(go_uniprot_dict), len(pp_uniprot_dict), len(pl_uniprot_dict))
    train_list_pp, test_list_pp, test_uniprots_1 = gen_train_test_ids(pp_uniprot_dict, ec_uniprot_dict, go_full_uniprot_dict)
    train_list_pl, test_list_pl, test_uniprots_2 = gen_train_test_ids(pl_uniprot_dict, ec_uniprot_dict, go_full_uniprot_dict)
    test_list_all = list(set(test_list_pl + test_list_pp))
    test_uniprots_all = list(set(test_uniprots_1 + test_uniprots_2))
    
    # full_train_ratio = 0.2
    # full_val_ratio = 0.4
    # full_test_ratio = 0.4
    random.shuffle(test_list_pl)
    random.shuffle(test_list_pp)
    
    full_test_list = test_list_pl[int(0.6*len(test_list_pl)):] + test_list_pp[int(0.6*len(test_list_pp)):]
    full_val_list = test_list_pl[int(0.2*len(test_list_pl)): int(0.6*(len(test_list_pl)))] + test_list_pp[int(0.2*len(test_list_pp)): int(0.6*len(test_list_pp))]
    full_train_list = test_list_pl[: int(0.2*len(test_list_pl))] + test_list_pp[: int(0.2*len(test_list_pp))]

    # 以前使用这种划分时，会去重，其实没必要
    train_list_ec = [ec_uniprot_dict[i] for i in ec_uniprot_dict if i not in test_uniprots_all]
    train_list_go = [go_uniprot_dict[i] for i in go_uniprot_dict if i not in test_uniprots_all]
    # train_list_ec = [ec_uniprot_dict[i] for i in ec_uniprot_dict]
    # train_list_go = [go_uniprot_dict[i] for i in go_uniprot_dict]
    print("Train List:", len(train_list_ec), len(train_list_go))
    train_list_all = list(set(train_list_pl + train_list_pp + train_list_ec + train_list_go + full_train_list))

    # train/val/test.txt contains samples of full labels, while train_all.txt contains labals with partial labels and samples in train.txt.
    print("Process finished, saving information into ../datasets/MultiTask/")
    if os.path.exists('../datasets/MultiTask/train_all.txt') and os.path.exists('../datasets/MultiTask/tmp/train.txt'):
        print("File already exists, skip saving split information...")
    else:
        print("Saving split information...")
        os.makedirs('../datasets/Multitask', exist_ok=True)
        save_dataset_info('../datasets/MultiTask/train_all.txt', train_list_all)
        save_dataset_info('../datasets/MultiTask/train.txt', full_train_list)
        save_dataset_info('../datasets/MultiTask/val.txt', full_val_list)
        save_dataset_info('../datasets/MultiTask/test.txt', full_test_list)
    # with open('../datasets/PropertyPrediction/train.txt', 'r') as f:   
    #     train_list_all = f.readlines()
    #     train_list_all = [i.strip() for i in train_list_all]
    # with open('../datasets/PropertyPrediction/val.txt', 'r') as f:
    #     full_val_list = f.readlines()
    #     full_val_list = [i.strip() for i in full_val_list]
    # with open('../datasets/PropertyPrediction/test.txt', 'r') as f:
    #     full_test_list = f.readlines()
    #     full_test_list = [i.strip() for i in full_test_list]
    # print(len(train_list_all), len(full_val_list), len(full_test_list))
    uniformed_label_dict = gen_label(train_list_all+full_test_list+full_val_list, ec_labels, go_labels, ppi_labels, lba_labels, ec_uniprot_dict, go_uniprot_dict, ec_info_dict, go_info_dict, pp_info_dict, pl_info_dict)    
    print("An example of the processed unified label:\n", full_test_list[0], ": ", uniformed_label_dict[full_test_list[0]])
    if os.path.exists('../datasets/MultiTask/uniformed_labels.json'):
        print("File already exists, skip saving uniformed labels...")
    else:
        print('Generating uniformed labels...')
        with open('../datasets/MultiTask/uniformed_labels.json', 'w') as f:
            json.dump(uniformed_label_dict, f)