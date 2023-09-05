from time import sleep

import torch

from tqdm.contrib.concurrent import process_map

def func(x: int):
    sleep(1)

def main():
    process_map(func, range(1000), max_workers=4, chunksize=5)

if __name__ == '__main__':
    main()
