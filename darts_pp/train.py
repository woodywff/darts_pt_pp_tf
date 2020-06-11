import pdb
import os
import torch
import torch.nn as nn
from helper import calc_param_size
from .utils import accuracy
from .searched import SearchedNet
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import defaultdict, OrderedDict
import pickle
from genotype import Genotype
import shutil
from .search import Base

DEBUG_FLAG = True

    
class Training(Base):
    '''
    Training the searched network
    cf: config.yml path
    cv_i: Which fold in the cross validation. If cv_i >= n_fold: use all the training dataset.
    for_train: If True, for training process, otherwise for searching.
    new_lr: if True, check_resume() will not load the saved states of optimizers and lr_schedulers.
    '''
    def __init__(self, cf='config.yml', cv_i=0, for_train=True, new_lr=False):
        super().__init__(cf=cf, cv_i=cv_i, for_train=for_train)
        self._init_model()
        self.check_resume(new_lr=new_lr)
    
    def _init_model(self):
        geno_file = os.path.join(self.log_path, self.config['search']['geno_file'])
        with open(geno_file, 'rb') as f:
            gene = eval(pickle.load(f)[0])
        self.model = SearchedNet(gene=gene, 
                                 in_channels=self.config['data']['in_channels'], 
                                 init_node_c=self.config['search']['init_node_c'], 
                                 out_channels=self.config['data']['out_channels'], 
                                 depth=self.config['search']['depth'], 
                                 n_nodes=self.config['search']['n_nodes'], 
                                 drop_rate=self.config['train']['drop_rate']).to(self.device)
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model)))
        self.loss = nn.CrossEntropyLoss().to(self.device)

        self.optim = Adam(self.model.parameters())
        self.scheduler = ReduceLROnPlateau(self.optim,verbose=True,factor=0.5)
        

    def check_resume(self, new_lr=False):
        self.last_save = os.path.join(self.log_path, self.config['train']['last_save'])
        self.best_shot = os.path.join(self.log_path, self.config['train']['best_shot'])
        if os.path.exists(self.last_save):
            state_dicts = torch.load(self.last_save, map_location=self.device)
            self.epoch = state_dicts['epoch'] + 1
            self.history = state_dicts['history']
            self.model.load_state_dict(state_dicts['model_param'])
            if not new_lr:
                self.optim.load_state_dict(state_dicts['optim'])
                self.scheduler.load_state_dict(state_dicts['scheduler'])
            self.best_val_loss = state_dicts['best_loss']
        else:
            self.epoch = 0
            self.history = defaultdict(list)
            self.best_val_loss = float('inf')

    def main_run(self):
        pdb.set_trace()
        n_epochs = self.config['train']['epochs']
        
        for epoch in range(n_epochs):
            is_best = False
            loss, acc1, acc5 = self.train()
            val_loss, val_acc1, val_acc5 = self.validate()
            self.scheduler.step(val_loss)
            self.history['loss'].append(loss)
            self.history['acc1'].append(acc1)
            self.history['acc5'].append(acc5)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc1'].append(val_acc1)
            self.history['val_acc5'].append(val_acc5)
            if val_loss < self.best_val_loss:
                is_best = True
                self.best_val_loss = val_loss
            
            # Save what the current epoch ends up with.
            state_dicts = {
                'epoch': self.epoch,
                'history': self.history,
                'model_param': self.model.state_dict(),
                'optim': self.optim.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'best_loss': self.best_val_loss
            }
            torch.save(state_dicts, self.last_save)
            
            if is_best:
                shutil.copy(self.last_save, self.best_shot)
            
            self.epoch += 1
            if self.epoch > n_epochs:
                break
            
            if DEBUG_FLAG and epoch >= 1:
                break
        print('Training Finished.')
        return 
        
    
    def train(self):
        '''
        Training | Training process
        '''
        self.model.train()
        n_steps = self.train_generator.steps_per_epoch
        sum_loss = 0
        sum_acc1 = 0
        sum_acc5 = 0
        with tqdm(self.train_generator.epoch(), total = n_steps,
                  desc = 'Training | Epoch {} | Training'.format(self.epoch)) as pbar:
            for step, (x, y_truth) in enumerate(pbar):
                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.long)

                self.optim.zero_grad()
                y_pred = self.model(x)
                loss = self.loss(y_pred, y_truth)
                sum_loss += loss.item()
                acc1, acc5 = accuracy(y_pred, y_truth, topk=(1,5))
                sum_acc1 += acc1
                sum_acc5 += acc5
                loss.backward()
                self.optim.step()
                
                postfix = OrderedDict()
                postfix['Loss'] = round(sum_loss/(step+1), 3)
                postfix['Top-1-Acc'] = round(sum_acc1/(step+1), 3)
                postfix['Top-5-Acc'] = round(sum_acc5/(step+1), 3)
                pbar.set_postfix(postfix)
                
                if DEBUG_FLAG and step >= 1:
                    break
                
        return [round(i/n_steps,3) for i in [sum_loss, sum_acc1, sum_acc5]]
    
    
    def validate(self):
        '''
        Training | Validation process
        '''
        self.model.eval()
        n_steps = self.val_generator.steps_per_epoch
        sum_loss = 0
        sum_acc1 = 0
        sum_acc5 = 0
        with tqdm(self.val_generator.epoch(), total = n_steps,
                  desc = 'Training | Epoch {} | Val'.format(self.epoch)) as pbar:
            for step, (x, y_truth) in enumerate(pbar):
                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.long)
                y_pred = self.model(x)
                loss = self.loss(y_pred, y_truth)
                sum_loss += loss.item()
                acc1, acc5 = accuracy(y_pred, y_truth, topk=(1,5))
                sum_acc1 += acc1
                sum_acc5 += acc5
                
                postfix = OrderedDict()
                postfix['Loss'] = round(sum_loss/(step+1), 3)
                postfix['Top-1-Acc'] = round(sum_acc1/(step+1), 3)
                postfix['Top-5-Acc'] = round(sum_acc5/(step+1), 3)
                pbar.set_postfix(postfix)
                
                if DEBUG_FLAG and step >= 1:
                    break
        return [round(i/n_steps,3) for i in [sum_loss, sum_acc1, sum_acc5]]

    
if __name__ == '__main__':
    training = Training()
    training.main_run()