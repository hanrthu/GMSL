import gzip
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import cytoolz
import httpx
from tqdm.contrib.concurrent import process_map

from gmsl.datamodule import get_pdb_ids

save_dir = Path('datasets') / 'pdb'

def download_batch(pdb_ids: list[str]):
    while True:
        r = httpx.get('https://download.rcsb.org/batch/structures/' + ':'.join(map(lambda x: f'{x}.pdb', pdb_ids)))
        with ZipFile(BytesIO(r.content)) as zipf:
            try:
                for filename in zipf.namelist():
                    (save_dir / filename).with_suffix('').write_bytes(gzip.decompress(zipf.read(filename)))
            except EOFError:
                continue
        break

def main():
    batch_size = 10
    save_dir.mkdir(exist_ok=True)
    process_map(
        download_batch,
        list(cytoolz.partition_all(batch_size, get_pdb_ids(save_path=Path('datasets') / 'pdb_ids.txt'))),
        ncols=80, max_workers=8, chunksize=1,
    )

if __name__ == '__main__':
    main()
