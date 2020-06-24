import pdb
import argparse
import yaml
import os
import time
import sys
import numpy as np
import pipeline
from helper import print_red, calc_param_size
from .nas import ShellNet
# from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import defaultdict, Counter, OrderedDict
import pickle
from .utils import accuracy
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from helper import ReduceLROnPlateau


DEBUG_FLAG = True

class Base:
    '''
    Base class for Searching and Training
    cf: config.yml path
    cv_i: Which fold in the cross validation. If cv_i >= n_fold: use all the training dataset.
    for_train: If True, for training process, otherwise for searching.
    channel_last: if True, corresponds to inputs with shape (batch, height, width, channels),
                  otherwise, (batch, channels, height, width).
    '''
    def __init__(self, cf='config.yml', cv_i=0, for_train=False, channel_last=True):
        self.cf = cf
        self.cv_i = cv_i
        self.for_train = for_train
        self.channel_last = channel_last
        self._init_config()
        self._init_log()
        self._init_device()
        self._init_dataset()
        
    def _init_config(self):
        with open(self.cf) as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)
        return
    
    def _init_log(self):
        self.log_path = self.config['data']['log_path']['tf']
        try:
            os.mkdir(self.log_path)
        except FileExistsError:
            pass

    def _init_device(self):
        '''
        So far we only consider single GPU.
        '''
        seed = self.config['data']['seed']
        np.random.seed(seed)
        tf.random.set_seed(seed)

        try: 
            gpu_check = tf.config.list_physical_devices('GPU')
            tf.config.experimental.set_memory_growth(gpu_check[0], True)
        except AttributeError: 
            gpu_check = tf.test.is_gpu_available()
        except IndexError:
            gpu_check = False
        if not (gpu_check):
            print_red('We are running on CPU!')
        return
    
    def _init_dataset(self):
        dataset = pipeline.Dataset(cf=self.cf, cv_i=self.cv_i, 
                                   for_train=self.for_train, channel_last=self.channel_last)
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
        self.model = ShellNet(img_size=self.config['data']['img_size'],
                              in_channels=self.config['data']['in_channels'], 
                              init_node_c=self.config['search']['init_node_c'], 
                              out_channels=self.config['data']['out_channels'], 
                              depth=self.config['search']['depth'], 
                              n_nodes=self.config['search']['n_nodes'])
        self.model(np.random.rand(1,28,28,1).astype('float32'),training=True)
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model.trainable_variables)))
        self.loss = lambda props, y_truth: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_truth, props))
        self.optim_shell = Adam() 
        self.optim_kernel = Adam()
        self.shell_scheduler = ReduceLROnPlateau(self.optim_shell, framework='tf')
        self.kernel_scheduler = ReduceLROnPlateau(self.optim_kernel, framework='tf')

    def check_resume(self, new_lr=False):
        checkpoint = tf.train.Checkpoint(model=self.model,
                                         optim_shell=self.optim_shell,
                                         optim_kernel=self.optim_kernel)
        self.manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                  directory=self.log_path,
                                                  checkpoint_name=self.config['search']['last_save'],
                                                  max_to_keep=1)
        self.last_aux = os.path.join(self.log_path, self.config['search']['last_aux'])
        if self.manager.latest_checkpoint:
            checkpoint.restore(self.manager.latest_checkpoint)
            with open(self.last_aux, 'rb') as f:
                state_dicts = pickle.load(f)
            self.epoch = state_dicts['epoch'] + 1
            self.geno_count = state_dicts['geno_count']
            self.history = state_dicts['history']
            if new_lr:
                del(self.optim_shell) # I'm not sure if this is necessary
                del(self.optim_kernel)
                self.optim_shell = Adam() 
                self.optim_kernel = Adam()
            else:
                self.shell_scheduler.load_state_dict(state_dicts['shell_scheduler'])
                self.kernel_scheduler.load_state_dict(state_dicts['kernel_scheduler'])
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
        for epoch in range(n_epochs):
            gene = str(self.model.get_gene())
            self.geno_count[gene] += 1
            if self.geno_count[gene] >= best_geno_count:
                print('>= best_geno_count: ({})'.format(best_geno_count))
                best_gene = (gene, best_geno_count)
                break

            shell_loss, kernel_loss, shell_acc, kernel_acc = self.train()
            self.shell_scheduler.step(shell_loss)
            self.kernel_scheduler.step(kernel_loss)
            self.history['shell_loss'].append(shell_loss)
            self.history['kernel_loss'].append(kernel_loss)
            self.history['shell_acc'].append(shell_acc)
            self.history['kernel_acc'].append(kernel_acc)
            
            
            # Save what the current epoch ends up with.
            self.manager.save()
            state_dicts = {
                'epoch': self.epoch,
                'geno_count': self.geno_count,
                'history': self.history,
                'kernel_scheduler': self.kernel_scheduler.state_dict(),
                'shell_scheduler': self.kernel_scheduler.state_dict(),
            }
            with open(self.last_aux, 'wb') as f:
                pickle.dump(state_dicts, f)
            
            self.epoch += 1
            if self.epoch > n_epochs:
                break
            
            if DEBUG_FLAG and epoch >= 3:
                break
                
        if best_gene is None:
            best_gene = self.geno_count.most_common(1)[0]
        with open(geno_file, 'wb') as f:
            pickle.dump(best_gene, f)
        return best_gene
    
    @tf.function
    def shell_step(self, x, y_truth):
        with tf.GradientTape() as tape:
            props = self.model(x, training=True)
            loss = self.loss(props, y_truth)
        grads = tape.gradient(loss, self.model.alphas)
        self.optim_kernel.apply_gradients(zip(grads, self.model.alphas))
        return props, loss
    
    @tf.function
    def kernel_step(self, x, y_truth):
        with tf.GradientTape() as tape:
            props = self.model(x, training=True)
            loss = self.loss(props, y_truth)
            loss += tf.add_n(self.model.losses) # l2 regularization
        grads = tape.gradient(loss, self.model.kernel.trainable_variables)
        self.optim_kernel.apply_gradients(zip(grads, self.model.kernel.trainable_variables))
        return props, loss
            
    def train(self):
        '''
        Searching | Training process
        To do optim_shell.step() and optim_kernel.step() alternately.
        '''
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
                x = tf.constant(x.astype('float32'))
                y_truth = tf.constant(y_truth.astype('int32'))
                try:
                    val_x, val_y_truth = next(val_epoch)
                except StopIteration:
                    val_epoch = self.val_generator.epoch()
                    val_x, val_y_truth = next(val_epoch)
                val_x = tf.constant(val_x.astype('float32'))
                val_y_truth = tf.constant(val_y_truth.astype('int32'))
                
                # optim_shell
# #                 pdb.set_trace()
#                 print(self.model.trainable_variables[0][0,0,0])
#                 print(self.model.alphas[0][:,0])
                val_props, val_loss = self.shell_step(val_x, val_y_truth)
                sum_val_loss += val_loss.numpy()
                val_acc = accuracy(val_props.numpy(), val_y_truth.numpy())
                sum_val_acc += val_acc
                
                # optim_kernel
# #                 pdb.set_trace()
#                 print(self.model.trainable_variables[0][0,0,0])
#                 print(self.model.alphas[0][:,0])
                y_props, loss = self.kernel_step(x, y_truth)
                sum_loss += loss.numpy()
                acc = accuracy(y_props.numpy(), y_truth.numpy())
                sum_acc += acc
#                 print(self.model.trainable_variables[0][0,0,0])
#                 print(self.model.alphas[0][:,0])
#                 pdb.set_trace()
                
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
    s = Searching()
    gene = s.search()