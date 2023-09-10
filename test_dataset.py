from utils.multitask_data import CustomMultiTaskDataset

val_dataset = CustomMultiTaskDataset(split='val', task='ec', gearnet=True, alpha_only=False, root_dir = './datasets/MultiTask_fold')