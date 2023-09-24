from pathlib import Path

import pandas as pd

from gmsl.data.table import lba_table_path, ppi_table_path

def gen_lba_labels():
    ret = {}
    for line in Path('datasets/PDBbind/refined-set/index/INDEX_general_PL_data.2020').read_text().splitlines():
        if line.startswith('#'):
            continue
        parts = line.strip().split()
        ret[parts[0]] = float(parts[3])
    pd.Series(ret, name='lba').to_csv(lba_table_path, index_label='pdb_id')

def gen_ppi_labels():
    root_dir = 'datasets/PDBbind/pp_affinity.xlsx'
    pp_info = pd.read_excel(root_dir, header=1, index_col='PDB code')
    pp_info['pKd pKi pIC50'].to_csv(ppi_table_path, index_label='pdb_id', header=['ppi'])

def main():
    gen_lba_labels()
    gen_ppi_labels()

if __name__ == '__main__':
    main()
