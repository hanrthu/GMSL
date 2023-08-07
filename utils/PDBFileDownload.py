from urllib.request import urlretrieve
import tqdm
import os
root_dir = "./datasets/ProtFunc/"
failed_times = 0
def get_pdb_file(root_dir,pdbid):
    url = 'https://ftp.wwpdb.org/pub/pdb/data/structures/divided/pdb/'+pdbid[1:3]+'/pdb' + pdbid + '.ent.gz'
    try:
        urlretrieve(url, root_dir +'all/' + pdbid.upper() + '.pdb.gz')
    except:
        print("cannot download:", pdbid)
        global failed_times
        failed_times += 1
        return
        

def main():
    pdbid_chainid_list = []
    with open(root_dir + "chain_functions.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.split(",")[0]
            pdbid_chainid_list.append(line.split("."))
    # pdbid_chainid_list = list(set(pdbid_chainid_list))
    for [pdbid,chainid] in tqdm.tqdm(pdbid_chainid_list):
        if(not os.path.exists(root_dir + "all/" + pdbid.upper() + ".pdb.gz")):
            get_pdb_file(root_dir,pdbid)
    print("Done")

if __name__ == "__main__":
    main()
    print("failed_times:", failed_times)