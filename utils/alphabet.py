NUM_ATOM_TYPES = 10
# Common Atoms
element_mapping = lambda x: {
    'Super': 0,
    'H' : 1,
    'C' : 2,
    'N' : 3,
    'O' : 4,
    'S' : 5,
    'P' : 6,
    'Metal': 7,
    'Halogen':8,
}.get(x, 9)

# The weight mapping for atoms
weight_mapping = lambda x: {
    'H' : 1,    
    'C' : 12,
    'N' : 14,
    'O' : 16,
    'S' : 32,
    'P' : 31,
    'Li': 3,  'LI': 3,
    'Mn': 55, 'MN': 55,
    'Cl': 35.5,
    'K' : 39,
    'Fe': 56, 'FE': 56,
    'Zn': 65, 'ZN': 65,
    'Mg': 24, 'MG': 24,
    'Br': 80, 'BR': 80,
    'I' : 127,
}.get(x, 0)

# The order is the same as TorchDrug.Protein
amino_acids = lambda x: {
    "GLY": 0,
    "ALA": 1, 
    "SER": 2, 
    "PRO": 3, 
    "VAL": 4, 
    "THR": 5, 
    "CYS": 6, 
    "ILE": 7, 
    "LEU": 8,
    "ASN": 9, 
    "ASP": 10, 
    "GLN": 11, 
    "LYS": 12, 
    "GLU": 13, 
    "MET": 14, 
    "HIS": 15, 
    "PHE": 16,
    "ARG": 17, 
    "TYR": 18,
    "TRP": 19,
    "SEC": 20,
    "PYL": 21}.get(x, 22)

idx_to_residue = lambda x: {
    0: ["GLY", "G"], 
    1: ["ALA", "A"],
    2: ["SER", "S"],
    3: ["PRO", "P"],
    4: ["VAL", "V"],
    5: ["THR", "T"],
    6: ["CYS", "C"],
    7: ["ILE", "I"],
    8: ["LEU", "L"],
    9: ["ASN", "N"],
    10:["ASP", "D"],
    11:["GLN", "Q"],
    12:["LYS", "K"],
    13:["GLU", "E"],
    14:["MET", "M"],
    15:["HIS", "H"],
    16:["PHE", "F"],
    17:["ARG", "R"],
    18:["TYR", "Y"],
    19:["TRP", "W"],
    20:["SEC", "U"],
    21:["PYL", "O"]
    }.get(x, ["UNK", "X"])

# The common bonds in protein
bond_dict = lambda x: {
    'C-C': 0,
    'C-O': 1, 'O-C': 1,
    'C-N': 2, 'N-C': 2,
    'C-H': 3, 'H-C': 3,
    'C-S': 4, 'S-C': 4,
    'N-H': 5, 'H-N': 5,
    'N-O': 6, 'O-N': 6,
    'N-S': 7, 'S-N': 7,
    'N-N': 8, 
    'O-H': 9, 'H-O': 9,
    'O-S': 10,'S-O': 10,
    'O-O': 11,
    'S-H': 12, 'H-S': 12,
    'H-H': 13,
    # 's1': 14, 
    # 's2': 15,
    # 'ss': 16
}.get(x, 14)

affinity_num_dict = lambda x :{
    'lba': [1],
    'ppi': [1],
    'multi': [1, 1] # 这里把LBA先保留着，主要是为了能够测试一下和之前性能比起来怎么样
}.get(x, [])

class_num_dict = lambda x : {
    'ec': [538],
    'mf': [490],
    'bp': [1944],
    'cc': [321],
    'go': [490, 1944, 321],
    'multi': [538, 490, 1944, 321]
}.get(x, [])