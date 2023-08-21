import torch 
from torch import cuda 
import time
from utils.multitask_data import CustomMultiTaskDataset

train_dataset = CustomMultiTaskDataset(split='train_ec', task='go', gearnet=True, alpha_only=False, root_dir = './datasets/MultiTask_go', label_dir = './datasets/MultiTask_go/uniformed_labels.json')
# x = torch.zeros([1,1024,1024,128*2], requires_grad=True, device='cuda:0') 
# print(x.dtype)
# print("1", cuda.memory_allocated()/1024**2)  

# y = 5 * x 
# # y.retain_grad()
# print("2", cuda.memory_allocated()/1024**2)  


# torch.mean(y).backward()     
# print("3", cuda.memory_allocated()/1024**2)    
# print(cuda.memory_summary())


# time.sleep(60)