from Bio.PDB import PDBParser

def main():
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('1a0a', 'datasets/pdb/1a0a.pdb')
    compound = structure.header['compound']
    print(compound)

if __name__ == '__main__':
    main()
