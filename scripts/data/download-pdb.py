import gzip
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile, BadZipFile

import cytoolz
import httpx
from tqdm.contrib.concurrent import process_map

from gmsl.data import get_pdb_ids

save_dir = Path('datasets') / 'pdbx-mmcif'

def download_batch(pdb_ids: list[str]):
    while True:
        try:
            r = httpx.get('https://download.rcsb.org/batch/structures/' + ':'.join(map(lambda x: f'{x}.cif', pdb_ids)))
        except httpx.TransportError:
            continue
        try:
            zipf = ZipFile(BytesIO(r.content))
        except BadZipFile:
            continue
        try:
            for filename in zipf.namelist():
                (save_dir / filename).with_suffix('').write_bytes(gzip.decompress(zipf.read(filename)))
        except EOFError:
            continue
        zipf.close()
        break

def main():
    batch_size = 32
    save_dir.mkdir(exist_ok=True)
    pdb_ids = set(get_pdb_ids(save_path=Path('datasets') / 'pdb_ids.txt'))
    pdb_ids -= set(path.stem.upper() for path in save_dir.iterdir())
    process_map(
        download_batch,
        list(cytoolz.partition_all(batch_size, pdb_ids)),
        ncols=80, max_workers=8, chunksize=1,
    )

if __name__ == '__main__':
    main()
