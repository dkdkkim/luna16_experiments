import torch
import pathlib
import json
import os
import wandb
import warnings
warnings.simplefilter('ignore')

import numpy as np
import torch.distributed as dist
import datetime as dt
from matplotlib.pyplot import axis
from sklearn import metrics
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from scipy.special import softmax

def _find_optimal_threshold(fpr, tpr, threshold):
    mul = tpr * (1 - fpr)
    max_idx = mul.argsort()[-1]
    opt_th = threshold[max_idx]
    return  opt_th

class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 option: dict,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 save_dir: pathlib.Path,
                 training_dataloader: torch.utils.data.Dataset,
                 validation_pos_dataloader: torch.utils.data.Dataset = None,
                 validation_neg_dataloader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epoch: int = 0,
                 distributed: bool = False,
                 train_sampler=None,
                 avu_criterion=None
                 ):

        self.net = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.training_dataloader = training_dataloader
        self.validation_pos_dataloader = validation_pos_dataloader
        self.validation_neg_dataloader = validation_neg_dataloader
        self.lr_scheduler = lr_scheduler
        self.epochs = option.epochs
        self.epoch = epoch
        self.exp_name = option.exp_name
        self.batch = option.batch
        self.distributed = distributed
        self.rank = option.rank
        self.world_size = option.world_size
        self.train_sampler = train_sampler
        self.num_classes = option.num_classes
        # self.auc_all=(option.cls_type=='mass')
        self.avu = avu_criterion
        self.sweep = option.sweep
        if self.avu is not None:
            self.beta = 3.
            self.opt_th = 1.
        
        self.training_loss, self.validation_pos_loss, self.validation_neg_loss = [], [], []
        self.training_acc, self.validation_pos_acc, self.validation_neg_acc, self.validation_acc = [], [], [], []
        self.validation_auc, self.th_opt, self.th_sens_max, self.th_malig_max = [], [], [], []
        self.learning_rate = []
        self.valid_pos_label, self.valid_pos_pred, self.valid_neg_label, self.valid_neg_pred = None, None, None, None
        if not self.distributed or (self.distributed and self.rank == 0):
            self.writer = SummaryWriter(os.path.join(self.save_dir, 'progress_log'))
        if self.distributed:
            self.group = dist.new_group(list(range(self.world_size)))

    def load_input(self, inputs):
        arrays = []
        for inp in inputs:
            arrays.append(np.expand_dims(np.load(inp),axis=0))
        output = np.concatenate(arrays)
        output = output / 255.
        output = np.expand_dims(output, axis=1)
        return torch.from_numpy(output).type(torch.float32)
    
    def load_mask(self, masks, labels):
        arrays = []
        for lbl, mask in zip(labels,masks):
            if lbl ==1:
                arrays.append(np.expand_dims(np.load(mask), axis=0))
            elif lbl == 0:
                arrays.append(np.zeros((1, 48, 48, 48)))
        for o in arrays:
            print(o.shape)
        output = np.concatenate(arrays)
        output = output / 255.
        output = np.expand_dims(output, axis = 1)
        return torch.from_numpy(output).type(torch.float32)

    def train(self):
        self.net.train()
        train_losses, train_acc = [], []
        if self.distributed:
            self.train_sampler.set_epoch(self.epoch)
        batch_iter = tqdm(enumerate(self.training_dataloader), 'Training',
                          total=len(self.training_dataloader),
                          leave=True, ncols=150)
        cur_lr = self.optimizer.param_groups[0]['lr']

        if self.avu is not None:
            preds_list, labels_list, unc_list = [], [], []
        
        for i, (x, y_label) in batch_iter:
            if type(x) == list:
                inp, target = [xx.cuda(non_blocking=True) for xx in x], y_label.cuda(non_blocking=True)
            else:
                inp, target = x.cuda(non_blocking=True), y_label.cuda(non_blocking=True)
            self.optimizer.zero_grad()
            y_pred = self.net(inp)
            loss = self.criterion(y_pred, target)
            y_pred_argmax = torch.argmax(y_pred, axis=-1)
            acc = ((target.cpu().numpy()==y_pred_argmax.cpu().detach().numpy()).sum()) / float(y_pred_argmax.size()[0])
            
            loss_value = loss.item()
            train_losses.append(loss.item())
            
            if self.avu is not None:
                avu_loss = self.beta * self.avu(y_pred, target, self.opt_th, type=0)
                loss += avu_loss

            # loss_value = loss.item()
            # train_losses.append(loss_value)
            train_acc.append(acc)
            
            if self.distributed:
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)

            loss.backward()
            self.optimizer.step()
            
            batch_iter.set_description(f'[{self.exp_name}]Training: {self.exp_name}(epoch {self.epoch}/loss {loss_value:.3f}/lr {cur_lr:.6f}/rank {self.rank})')

            if self.distributed:
                loss_value = torch.Tensor([loss_value]).cuda()
                acc = torch.Tensor([acc]).cuda()
                dist.all_reduce(loss_value, op=dist.ReduceOp.SUM)
                dist.all_reduce(acc, op=dist.ReduceOp.SUM)
                loss_value /= self.world_size
                acc /= self.world_size
                loss_value = loss_value.item()
                acc = acc.item()

            if not self.distributed or (self.distributed and self.rank == 0):
                self.writer.add_scalar('Loss/train_step',
                                    loss_value, (self.epoch-1)*len(batch_iter)+i)
                self.writer.add_scalar('Accuracy/train_step',
                                    acc, (self.epoch-1)*len(batch_iter)+i)
            
            if self.avu is not None:
                preds_list.append(y_pred_argmax.cpu().numpy())
                labels_list.append(target.cpu().numpy())
                pred = y_pred.cpu().detach().numpy()
                pred_entropy = -1 * np.sum(pred * np.log(pred + 1e-15), axis=-1)
                unc_list.append(pred_entropy)

        if self.avu is not None:
            preds = np.concatenate(preds_list)
            labels = np.concatenate(labels_list)
            unc_ = np.concatenate(unc_list)
            unc_correct = np.take(unc_, np.where(preds == labels))
            unc_incorrect = np.take(unc_, np.where(preds != labels))
        
            if self.epoch <= 1:
                self.opt_th = (np.mean(unc_correct, axis=1) + np.mean(unc_incorrect, axis=1)) / 2
            print(f"opt_th: {self.opt_th}")
            

        print(f"\ntraining loss: {np.mean(train_losses):.3f}, training acc: {np.mean(train_acc):.3f}, learning rate: {cur_lr}")
        self.training_loss.append(np.mean(train_losses))
        self.training_acc.append(np.mean(train_acc))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])

        batch_iter.close()

    def validate(self, data_loader, valid_type):
        self.net.eval()
        valid_losses, valid_acc, cnt = 0, 0, 0
        label_arr, pred_arr = np.zeros((1,1)), np.zeros((1,self.num_classes))
        batch_iter = tqdm(enumerate(data_loader), 'Validation', total=len(data_loader),
                            leave=False, ncols=150)
        print(f"\nvalidation start: {dt.datetime.now()}")
        for i, (x, y_label) in batch_iter:
            if not torch.is_tensor(y_label):
                y_label = torch.Tensor(y_label)
            if type(x) == list:
                inp, target = [xx.cuda() for xx in x], y_label.cuda()
            else:
                inp, target = x.cuda(), y_label.cuda()

            with torch.no_grad():
                y_pred = self.net.module(inp)
                y_pred_argmax = torch.argmax(y_pred, axis=-1)
                loss = self.criterion(y_pred, target)
                acc = ((target.cpu().numpy()==y_pred_argmax.cpu().detach().numpy()).sum()) 
                loss_value = loss.item()
                cnt += y_pred.size()[0]
                valid_losses += loss_value * y_pred.size()[0]
                valid_acc += acc
                
                batch_iter.set_description(f'Validation: (loss {loss_value:.4f}/valid type: {valid_type})')
                label_arr = np.vstack((label_arr,np.expand_dims(target.cpu().numpy(), axis=1)))
                pred_arr = np.vstack((pred_arr,y_pred.cpu().numpy()))
        
        print(f"\n[valid type: {valid_type}] validation loss: {float(valid_losses)/cnt}, "
              f"validation acc: {float(valid_acc)/cnt}")

        cur_valid_loss = float(valid_losses)/cnt
        cur_valid_acc = float(valid_acc)/cnt

        if valid_type == 'valid_pos':
            self.validation_pos_loss.append(cur_valid_loss)
            self.validation_pos_acc.append(cur_valid_acc)
            self.valid_pos_label = label_arr[1:]
            self.valid_pos_pred = pred_arr[1:]
        elif valid_type == 'valid_neg':
            self.validation_neg_loss.append(cur_valid_loss)
            self.validation_neg_acc.append(cur_valid_acc)
            self.valid_neg_label = label_arr[1:]
            self.valid_neg_pred = pred_arr[1:]

    def cal_accuracy(self, pred, label, th):
        th_pred = np.where(pred[:,1] >= th, 1., 0.)
        acc = np.where(th_pred == label, 1., 0.)
        acc = np.mean(acc)
        return acc

    def AUC(self):
        label_arr = np.vstack((self.valid_pos_label, self.valid_neg_label))
        pred_arr = np.vstack((self.valid_pos_pred, self.valid_neg_pred))
        
        pred_arr = softmax(pred_arr, axis=1)
        fpr, tpr, thresholds = metrics.roc_curve(label_arr, pred_arr[:, -1])
        auc_value = metrics.auc(fpr, tpr)
        opt_th = _find_optimal_threshold(fpr, tpr, thresholds)
        self.validation_auc.append(auc_value)
        print(f"[{self.exp_name}]AUC: {auc_value:.4f} / opt_th = {opt_th:.3f}")

        self.th_opt.append(opt_th)

        th_name_list = ['0.5', 'opt']
        th_list = [0.5, opt_th]
        acc_dict = {}
        for th_name, th in zip(th_name_list, th_list):
            acc_dict[th_name] = {
                'th': th,
                'pos': self.cal_accuracy(softmax(self.valid_pos_pred, axis=1), self.valid_pos_label, th),
                'neg': self.cal_accuracy(softmax(self.valid_neg_pred, axis=1), self.valid_neg_label, th),
                'acc': self.cal_accuracy(pred_arr, label_arr, th)
            }
            print(f"[threshold {th_name} accuracy({th:.4f})]  "
                  f"acc {acc_dict[th_name]['acc']:.4f}\n"
                  f"pos {acc_dict[th_name]['pos']:.4f}\n"
                  f"neg {acc_dict[th_name]['neg']:.4f}")
        self.validation_acc.append(acc_dict['0.5']['acc'])
        self.valid_label, self.valid_pred = None, None
        
    def run_trainer(self):
        if self.rank == 0:
            
            if self.sweep:
                run = wandb.init()
                print(f"Resume wandb: {run.name}")
            else:
                run = wandb.init(project='luna16_experiments',
                                 name=self.exp_name)
                print(f"Wandb initialization: {run.name}")
            
        progressbar = trange(self.epochs, desc='Progress', leave=False)
        min_loss = 100.
        max_auc = 0.
        max_acc = 0.
        for i in progressbar:
            init_time = dt.datetime.now()
            ''' epoch counter '''
            self.epoch += 1
            ''' Training block '''
            self.train()
            print(f"train end: {dt.datetime.now()}")
            ''' Validation block '''
            if not self.distributed or (self.distributed and self.rank == 0):

                self.validate(self.validation_pos_dataloader, 'valid_pos')
                self.validate(self.validation_neg_dataloader, 'valid_neg')

                if self.num_classes == 2:
                    self.AUC()
                
                self.writer.add_scalars('Loss/epoch', {'train': self.training_loss[-1]}, self.epoch)
                self.writer.add_scalars('Accuracy/epoch', {'train': self.training_acc[-1]}, self.epoch)
                self.writer.add_scalar('Learning rate', self.optimizer.param_groups[0]['lr'], self.epoch)
                self.writer.add_scalars('Loss/epoch', {'valid_pos': self.validation_pos_loss[-1]}, self.epoch)
                self.writer.add_scalars('Loss/epoch', {'valid_neg': self.validation_neg_loss[-1]}, self.epoch)
                self.writer.add_scalars('Accuracy/epoch', {'valid_pos': self.validation_pos_acc[-1]}, self.epoch)
                self.writer.add_scalars('Accuracy/epoch', {'valid_neg': self.validation_neg_acc[-1]}, self.epoch)
                self.writer.add_scalars('Accuracy/epoch', {'valid': self.validation_acc[-1]}, self.epoch)

                log_dict = {
                    'train_loss': self.training_loss[-1],
                    'train_acc': self.training_acc[-1],
                    'learning_rate': self.optimizer.param_groups[0]['lr'],
                    'valid_pos_loss': self.validation_pos_loss[-1],
                    'valid_pos_acc': self.validation_pos_acc[-1],
                    'valid_neg_loss': self.validation_neg_loss[-1],
                    'valid_neg_acc': self.validation_neg_acc[-1],
                    'valid_acc': self.validation_acc[-1]
                }

                if self.num_classes == 2:
                    self.writer.add_scalars('AUC/epoch', {'valid': self.validation_auc[-1]}, self.epoch)
                    self.writer.add_scalars('threshold/epoch', {'valid_opt': self.th_opt[-1]}, self.epoch)

                    log_dict['auc'] = self.validation_auc[-1]
                    
                if self.rank == 0:
                    wandb.log(log_dict)
                ''' model save block'''
                cur_loss = (self.validation_pos_loss[-1] + self.validation_neg_loss[-1]) / 2
                cur_acc = self.validation_acc[-1]
                cur_auc = self.validation_auc[-1]
                
                if cur_loss < min_loss:
                    min_loss = cur_loss
                if cur_acc > max_acc:
                    print(f"max_acc changed: {cur_acc}")
                    max_acc = cur_acc
                if cur_auc > max_auc:
                    print(f"max_auc changed: {cur_auc}")
                    max_auc = cur_auc
                    
                # if cur_loss < min_loss or self.validation_auc[-1] > max_auc or i == self.epochs-1:
                if torch.cuda.device_count() > 1 or self.distributed:
                    torch.save(self.net.module.state_dict(), self.save_dir / f"model_{i}.pt")
                else:
                    torch.save(self.net.state_dict(), self.save_dir / f"model_{i}.pt")
                print('model saved')

            ''' Learning rate scheduler block'''
            if self.lr_scheduler is not None:
                if self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.step(cur_loss)
                else:
                    self.lr_scheduler.step()
            
            print(f"epoch processing time: {dt.datetime.now() - init_time}")
        
        if not self.distributed or (self.distributed and self.rank == 0):    
            wandb.log({
                    'max_auc': max_auc,
                    'max_acc': max_acc,
                    'min_loss': min_loss
                })
            return {
                    'max_auc': max_auc,
                    'max_acc': max_acc,
                    'min_loss': min_loss
                }