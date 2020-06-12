import pdb
import argparse
import yaml
import os
import time
import sys
import numpy as np
import pipeline
from helper import calc_param_size, print_red
from .nas import ShellNet
# from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import defaultdict, Counter, OrderedDict
import pickle
from paddle.fluid import core
import paddle.fluid as fluid
from paddle.fluid.optimizer import Adam
from paddle.fluid.layers import accuracy


DEBUG_FLAG = True

class Base:
    '''
    Base class for Searching and Training
    cf: config.yml path
    cv_i: Which fold in the cross validation. If cv_i >= n_fold: use all the training dataset.
    for_train: If True, for training process, otherwise for searching.
    '''
    def __init__(self, cf='config.yml', cv_i=0, for_train=False):
        self.cf = cf
        self.cv_i = cv_i
        self.for_train = for_train
        self._init_config()
        self._init_log()
        self._init_device()
        self._init_dataset()
        
    def _init_config(self):
        with open(self.cf) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        return
    
    def _init_log(self):
        self.log_path = self.config['data']['log_path']['pp']
        try:
            os.mkdir(self.log_path)
        except FileExistsError:
            pass

    def _init_device(self):
        seed = self.config['data']['seed']
        np.random.seed(seed)
        if not core.is_compiled_with_cuda():
            print_red('PaddlePaddle for CPU!')
        return
    
    def _init_dataset(self):
        dataset = pipeline.Dataset(cf=self.cf, cv_i=self.cv_i, for_train=self.for_train)
        self.train_generator = dataset.train_generator
        self.val_generator = dataset.val_generator
        return

class Searching(Base):
    '''
    Searching process
    cf: config.yml path
    cv_i: Which fold in the cross validation. If cv_i >= n_fold: use all the training dataset.
    new_lr: if True, check_resume() will not load the saved states of optimizers and lr_schedulers.
    '''
    def __init__(self, cf='config.yml', cv_i=0, new_lr=False):
        super().__init__(cf=cf, cv_i=cv_i)
        self._init_model()
        self.check_resume(new_lr=new_lr)
    
    def _init_model(self):
        pdb.set_trace()
        self.model = ShellNet(in_channels=self.config['data']['in_channels'], 
                              init_node_c=self.config['search']['init_node_c'], 
                              out_channels=self.config['data']['out_channels'], 
                              depth=self.config['search']['depth'], 
                              n_nodes=self.config['search']['n_nodes'])
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model)))
        self.loss = lambda props, y_truth: fluid.layers.reduce_mean(fluid.layers.softmax_with_cross_entropy(props, y_truth))
        self.optim_shell = Adam(parameter_list=self.model.alphas()) 
        self.optim_kernel = Adam(parameter_list=self.model.kernel.parameters())

    def check_resume(self, new_lr=False):
        self.last_save = os.path.join(self.log_path, self.config['search']['last_save'])
        self.last_aux = os.path.join(self.log_path, self.config['search']['last_aux'])
        if os.path.exists(self.last_aux):
            model_params,_ = fluid.dygraph.load_dygraph(self.last_save)
            with open(self.last_aux, 'rb') as f:
                state_dicts = pickle.load(f)
            self.epoch = state_dicts['epoch'] + 1
            self.geno_count = state_dicts['geno_count']
            self.history = state_dicts['history']
            self.model.set_dict(model_params)
            if not new_lr:
                self.optim_shell._learning_rate = state_dicts['lr_shell']
                self.optim_kernel._learning_rate = state_dicts['lr_kernel']
        else:
            self.epoch = 0
            self.geno_count = Counter()
            self.history = defaultdict(list)

    def search(self):
        '''
        Return the best genotype in tuple:
        (best_gene: str(Genotype), geno_count: int)
        '''
#         pdb.set_trace()
        geno_file = os.path.join(self.log_path, self.config['search']['geno_file'])
        if os.path.exists(geno_file):
            print('{} exists.'.format(geno_file))
            with open(geno_file, 'rb') as f:
                return pickle.load(f)

        best_gene = None
        best_geno_count = self.config['search']['best_geno_count']
        n_epochs = self.config['search']['epochs']
        lr_count_shell = 0
        lr_count_kernel = 0
        lr_record_shell = float('inf')
        lr_record_shell = float('inf')
        lr_patience = self.config['search']['lr_patience']
        for epoch in range(n_epochs):
            gene = str(self.model.get_gene())
            self.geno_count[gene] += 1
            if self.geno_count[gene] >= best_geno_count:
                print('>= best_geno_count: ({})'.format(best_geno_count))
                best_gene = (gene, best_geno_count)
                break

            shell_loss, kernel_loss, shell_acc, kernel_acc = self.train()
            
            if shell_loss >= lr_record_shell:
                lr_count_shell += 1
                if lr_count_shell == lr_patience:
                    self.optim_shell._learning_rate *= 0.5
                    lr_count_shell = 0
                    lr_record_shell = shell_loss
            else:
                lr_count_shell = 0
                lr_record_shell = shell_loss
                
            self.shell_scheduler.step(shell_loss)
            self.kernel_scheduler.step(kernel_loss)
            self.history['shell_loss'].append(shell_loss)
            self.history['kernel_loss'].append(kernel_loss)
            self.history['shell_acc'].append(shell_acc)
            self.history['kernel_acc'].append(kernel_acc)
            
            
            # Save what the current epoch ends up with.
            state_dicts = {
                'epoch': self.epoch,
                'geno_count': self.geno_count,
                'history': self.history,
                'model_param': self.model.state_dict(),
                'optim_shell': self.optim_shell.state_dict(),
                'optim_kernel': self.optim_kernel.state_dict(),
                'kernel_scheduler': self.kernel_scheduler.state_dict(),
                'shell_scheduler': self.kernel_scheduler.state_dict(),
            }
            torch.save(state_dicts, self.last_save)
            
            self.epoch += 1
            if self.epoch > n_epochs:
                break
            
            if DEBUG_FLAG and epoch >= 1:
                break
                
        if best_gene is None:
            best_gene = self.geno_count.most_common(1)[0]
        with open(geno_file, 'wb') as f:
            pickle.dump(best_gene, f)
        return best_gene
        
    
    def train(self):
        '''
        Searching | Training process
        To do optim_shell.step() and optim_kernel.step() alternately.
        '''
        self.model.train()
        train_epoch = self.train_generator.epoch()
        val_epoch = self.val_generator.epoch()
        n_steps = self.train_generator.steps_per_epoch
        sum_loss = 0
        sum_val_loss = 0
        sum_acc = 0
        sum_val_acc = 0
        with tqdm(train_epoch, total = n_steps,
                  desc = 'Searching | Epoch {}'.format(self.epoch)) as pbar:
            for step, (x, y_truth) in enumerate(pbar):
                x = torch.as_tensor(x, device=self.device, dtype=torch.float)
                y_truth = torch.as_tensor(y_truth, device=self.device, dtype=torch.long)
                try:
                    val_x, val_y_truth = next(val_epoch)
                except StopIteration:
                    val_epoch = self.val_generator.epoch()
                    val_x, val_y_truth = next(val_epoch)
                val_x = torch.as_tensor(val_x, device=self.device, dtype=torch.float)
                val_y_truth = torch.as_tensor(val_y_truth, device=self.device, dtype=torch.long)
                
                # optim_shell
                self.optim_shell.zero_grad()
                val_y_pred = self.model(val_x)
                val_loss = self.loss(val_y_pred, val_y_truth)
                sum_val_loss += val_loss.item()
                val_acc = accuracy(val_y_pred, val_y_truth)[0]
                sum_val_acc += val_acc
                val_loss.backward()
                self.optim_shell.step()
                
                # optim_kernel
                self.optim_kernel.zero_grad()
                y_pred = self.model(x)
                loss = self.loss(y_pred, y_truth)
                sum_loss += loss.item()
                acc = accuracy(y_pred, y_truth)[0]
                sum_acc += acc
                loss.backward()
                self.optim_kernel.step()
                
                # postfix for progress bar
                postfix = OrderedDict()
                postfix['Loss(optim_shell)'] = round(sum_val_loss/(step+1), 3)
                postfix['Acc(optim_shell)'] = round(sum_val_acc/(step+1), 3)
                postfix['Loss(optim_kernel)'] = round(sum_loss/(step+1), 3)
                postfix['Acc(optim_kernel)'] = round(sum_acc/(step+1), 3)
                pbar.set_postfix(postfix)
                
                if DEBUG_FLAG and step > 1:
                    break
                
        return [round(i/n_steps,3) for i in [sum_val_loss, sum_loss, sum_val_acc, sum_acc]]
    
    
    
if __name__ == '__main__':
    with fluid.dygraph.guard():
        searching = Searching()
        gene = searching.search()