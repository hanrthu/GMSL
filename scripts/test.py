import torch

from tqdm.contrib.concurrent import process_map

def func(x: int):
    print(x)

def main():
    process_map(func, range(10))

if __name__ == '__main__':
    main()
