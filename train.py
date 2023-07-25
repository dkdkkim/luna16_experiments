import os
import wandb
import torch
import builtins
import pathlib
import json
from tqdm import trange
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
import pandas as pd
from setproctitle import setproctitle


from models.resnest_3d import generate_model as gen_resnest
from train_option import get_option, save_option
from loss_function import generate_loss_function
from optimizer import generate_optimizer
from lr_scheduler import generate_lr_scheduler
from dataset import ClassDataset, BalancedDistributedSampler, DatasetDataframe
from transforms import CenterCrop, RandomCrop, Flip, Normalize, Rotate, Unsqueeze, Compose
from trainer import Trainer

def _generate_model(option, crop_size):
    if option.model == 'resnest':
        model = gen_resnest(50)
    else:
        raise Exception(f"Wrong model name: {option.model}")
    return model

def main():
    
    option = get_option()
    
    if option.sweep:
        mp.set_start_method('spawn', force=True)  # set start method to 'spawn' BEFORE instantiating the queue and the event
        result_queue = mp.Queue()
        run = wandb.init(project=option.exp_name, resume='auto')
        print(f"run name: {run.name}")
        config = wandb.config
        option.loss_gamma = config.loss_fn_gamma
        option.loss_alpha = config.loss_fn_alpha
        option.exp_name = f"{option.exp_name}_{(run.name).split('-')[-1].zfill(3)}"
        result_queue = mp.Queue()
    else:
        result_queue=None
    
    os.environ["CUDA_VISIBLE_DEVICES"] = option.CUDA_VISIBLE_DEVICES
    option.world_size = torch.cuda.device_count()
    ngpus_per_node = option.world_size ### Use all GPUs
    print(f"ngpu per node: {ngpus_per_node}, world size: {option.world_size}")
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, option, result_queue))

def main_worker(rank, ngpus_per_node, option, queue):
    setproctitle(f"luna16_train")
    option.gpu = rank
    option.rank = rank
    print(f"Use GPU: {option.gpu} for training")
    
    if option.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass
    
    dist.init_process_group(backend=option.dist_backend, 
                            init_method=option.dist_url,
                            world_size=option.world_size,
                            rank=option.rank)
    
    save_path = option.save_path
    save_dir = pathlib.Path(os.path.join(save_path, 'fp_reduction')) / option.exp_name
    model_dir = save_dir / 'model'
    
    ''' model setting '''
    crop_size = [int(x) for x in option.crop_size.split(',')]
    if option.rank == 0:
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
            model_dir.mkdir(parents=True)
        save_option(save_dir, option)
    
    model = _generate_model(option, crop_size)
    torch.cuda.set_device(option.gpu)
    model.cuda(option.gpu)
    option.batch = int(option.batch / ngpus_per_node)
    option.num_workers = int((option.num_workers + ngpus_per_node - 1) / ngpus_per_node)

    if option.checkpoint:
        load_path = option.checkpoint
        loc = 'cuda:{}'.format(option.gpu)
        model_weights = torch.load(load_path, map_location=loc)
        try:
            model.load_state_dict(model_weights, strict=True)
            print(f"Weights loaded from {load_path}")
        except:
            model.load_state_dict(model_weights, strict=False)
            print(f"Some weights are not matched to the architecture. \nMatched weights loaded from {load_path}")
        
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[option.gpu])

    print(f"Number of classes: {option.num_classes}")
    
    ''' loss function, optimizer and learning rate scheduler'''
    criterion = generate_loss_function(option)
    lr_init = option.lr_min if option.lr_scheduler == 'cosine' else option.lr_init
    optimizer = generate_optimizer(option, model.parameters(), lr_init)
    lr_scheduler = generate_lr_scheduler(option, optimizer)
    
    ''' data setting '''    
    train_transforms = Compose([RandomCrop(crop_size=crop_size, shift_ratio=option.shift_ratio), Rotate(), Flip(), Normalize(), Unsqueeze()])
    valid_transforms = Compose([CenterCrop(crop_size=crop_size), Normalize(), Unsqueeze()])

    dataset_train = DatasetDataframe('train.csv', option.data_path, train_transforms, only_path=True)
    dataset_valid_pos = DatasetDataframe('valid_pos.csv', option.data_path, valid_transforms, scales=[1.0])
    dataset_valid_neg = DatasetDataframe('valid_neg.csv', option.data_path, valid_transforms, scales=[1.0])
    train_sampler = BalancedDistributedSampler(dataset_train, num_replicas=option.world_size, rank=option.rank)
    dataset_train.only_path=False
    print(f"Dataset set-up complete")
    
    dataloader_train = DataLoader(dataset=dataset_train,
                                  sampler=train_sampler,
                                  batch_size=option.batch,
                                  num_workers=option.num_workers,
                                  pin_memory=option.pin_memory,
                                  persistent_workers=True,
                                  prefetch_factor=option.prefetch_factor)
    dataloader_valid_pos = DataLoader(dataset=dataset_valid_pos,
                                  batch_size=option.batch,
                                  shuffle=True,
                                  num_workers=option.num_workers,
                                  pin_memory=option.pin_memory)
    dataloader_valid_neg = DataLoader(dataset=dataset_valid_neg,
                                  batch_size=option.batch,
                                  shuffle=True,
                                  num_workers=option.num_workers,
                                  pin_memory=option.pin_memory)

    print(f"Dataloader set-up complete")
    
    trainer = Trainer(model=model,
                        device=torch.device('cuda'),
                        option=option,
                        criterion=criterion,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        save_dir=save_dir,
                        training_dataloader=dataloader_train,
                        validation_pos_dataloader=dataloader_valid_pos,
                        validation_neg_dataloader=dataloader_valid_neg,
                        epoch=0,
                        distributed=True,
                        train_sampler=train_sampler,
                        )
    print('model setting comlete')
    
    exp_result = trainer.run_trainer()
    
    print('*'*10, 'Training Complete', '*'*10)
    
    if option.rank == 0 and option.sweep:
        print(option.rank, exp_result)
        queue.put(exp_result)

if __name__ == '__main__':
    option = get_option()
    if option.sweep:
        sweep_configuration = {
            'method': 'bayes',
            'metric': 
            {
                'goal': 'maximize', 
                'name': 'max_acc'
                },
            'parameters': 
            {
                'loss_fn_alpha': {'max': 0.99, 'min': 0.01},
                'loss_fn_gamma': {'values': [0.5, 1., 2., 5.]},
            }
        }
        
        sweep_id = wandb.sweep(
        sweep=sweep_configuration, 
        project='luna16_fp_reduction'
        )

        wandb.agent(sweep_id, function=main, count=20)
    else:
        main()