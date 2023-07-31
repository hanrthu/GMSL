## Process the original datasets

To start with, you can download out processed [PDBBind](http://www.pdbbind.org.cn), [EnzymeCommission](https://github.com/flatironinstitute/DeepFRI) and [GeneOntology](https://github.com/flatironinstitute/DeepFRI) datasets.

After placing the data into ./datasets, your folder structure should be consistent with the following.

```
├── datasets
│   ├── EnzymeCommission
│   │   ├── all
│   │   ├── ec_cache_train.pkl
│   │   ├── nrPDB-EC_annot.tsv
│   │   ├── nrPDB-EC_sequences.fasta
│   │   ├── nrPDB-EC_test.csv
│   │   ├── nrPDB-EC_test_sequences.fasta
│   │   ├── nrPDB-EC_test.txt
│   │   ├── nrPDB-EC_train.txt
│   │   └── nrPDB-EC_valid.txt
│   ├── GeneOntology
│   │   ├── all
│   │   ├── go_mf_cache_train.pkl
│   │   ├── nrPDB-GO_annot.tsv
│   │   ├── nrPDB-GO_sequences.fasta
│   │   ├── nrPDB-GO_test.csv
│   │   ├── nrPDB-GO_test_sequences.fasta
│   │   ├── nrPDB-GO_test.txt
│   │   ├── nrPDB-GO_train.txt
│   │   └── nrPDB-GO_valid.txt
│   ├── PDBbind
│   │   ├── mol2
│   │   ├── PP
│   │   ├── refined-set
│   │   ├── sdf
│   │   ├── protein_protein
│   │   └── v2020-other-PL
```

Then, parse the Uniprot ids from PDBBank based on the PDB ids of each dataset, and generate the necessary information to ./output_info/. After this, we can generate information for each single dataset.

```
python utils/PDBWebparser.py
```

After this, we can generate the Multitask dataset with above information and divide the samples into train/train_all/val/test splits.
```
python process_multi_task_datset.py
```

(这里目前稍繁琐，之后会把PDBChainUniprotparser整合到PDBWebParser中)
We also need to align each chain with their Uniprot id for further training, and run the following codes.
```
python utils/PDBChainUniprotparser.py
```


After all these steps, we can successfully generate the necessary labeling information for different tasks.