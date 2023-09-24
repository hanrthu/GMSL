from pathlib import Path

import httpx
from tqdm.contrib.concurrent import process_map

from gmsl.data import get_pdb_ids

save_dir = Path('datasets') / 'fasta'

def download(pdb_id: str):
    r = httpx.get(f'https://www.rcsb.org/fasta/entry/{pdb_id}')
    (save_dir / f'{pdb_id}.fasta').write_text(r.content.decode())

def main():
    save_dir.mkdir(exist_ok=True)
    process_map(download, get_pdb_ids(), ncols=80, max_workers=32, chunksize=1)

if __name__ == '__main__':
    main()
