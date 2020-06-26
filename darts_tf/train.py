import pdb
import os
from helper import calc_param_size, ReduceLROnPlateau
from .searched import SearchedNet
# from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import defaultdict, OrderedDict
from genotype import Genotype
import shutil
from .search import Base
import pickle
import numpy as np
from .utils import accuracy
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

DEBUG_FLAG = False

    
class Training(Base):
    '''
    Training the searched network
    cf: config.yml path
    cv_i: Which fold in the cross validation. If cv_i >= n_fold: use all the training dataset.
    for_train: If True, for training process, otherwise for searching. It affects the Dataset.
    new_lr: if True, check_resume() will not load the saved states of optimizers and lr_schedulers.
    '''
    def __init__(self, cf='config.yml', cv_i=0, for_train=True, new_lr=False):
        super().__init__(cf=cf, cv_i=cv_i, for_train=for_train)
        self._init_model()
        self.check_resume(new_lr=new_lr)
    
    def _init_model(self):
        geno_file = os.path.join(self.search_log, self.config['search']['geno_file'])
        with open(geno_file, 'rb') as f:
            gene = eval(pickle.load(f)[0])
        self.model = SearchedNet(gene=gene, 
                                 img_size=self.config['data']['img_size'],
                                 in_channels=self.config['data']['in_channels'], 
                                 init_node_c=self.config['search']['init_node_c'], 
                                 out_channels=self.config['data']['out_channels'], 
                                 depth=self.config['search']['depth'], 
                                 n_nodes=self.config['search']['n_nodes'], 
                                 drop_rate=self.config['train']['drop_rate'])
        self.model(np.random.rand(1,28,28,1).astype('float32'),training=True)
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model.trainable_variables)))
        self.loss = lambda props, y_truth: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_truth, props))
        
        self.optim = Adam()
        self.scheduler = ReduceLROnPlateau(self.optim, framework='tf')

    def check_resume(self, new_lr=False):
        checkpoint = tf.train.Checkpoint(model=self.model,
                                         optim=self.optim)
        self.manager = tf.train.CheckpointManager(checkpoint=checkpoint,
                                                  directory=self.train_log,
                                                  checkpoint_name=self.config['train']['last_save'],
                                                  max_to_keep=1)
        self.last_save = os.path.join(self.train_log, self.config['train']['last_save'])
        self.last_aux = os.path.join(self.train_log, self.config['train']['last_aux'])
        self.best_shot = os.path.join(self.train_log, self.config['train']['best_shot'])
        self.best_aux = os.path.join(self.train_log, self.config['train']['best_aux'])
#         if self.manager.latest_checkpoint:
        if os.path.exists(self.last_aux):
            checkpoint.restore(self.manager.latest_checkpoint)
            with open(self.last_aux, 'rb') as f:
                state_dicts = pickle.load(f)
            self.epoch = state_dicts['epoch'] + 1
            self.history = state_dicts['history']
            if new_lr:
                del(self.optim)
                self.optim = Adam()
            else:
                self.scheduler.load_state_dict(state_dicts['scheduler'])
            self.best_val_loss = state_dicts['best_loss']
        else:
            self.epoch = 0
            self.history = defaultdict(list)
            self.best_val_loss = float('inf')

    def main_run(self):
#         pdb.set_trace()
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
            self.manager.save()
            state_dicts = {
                'epoch': self.epoch,
                'history': self.history,
                'scheduler': self.scheduler.state_dict(),
                'best_loss': self.best_val_loss
            }
            with open(self.last_aux, 'wb') as f:
                pickle.dump(state_dicts, f)
            
            if is_best:
                self.model.save_weights(self.best_shot)
                shutil.copy(self.last_aux, self.best_aux)
            
            self.epoch += 1
            if self.epoch > n_epochs:
                break
            
            if DEBUG_FLAG and epoch >= 1:
                break
        print('Training Finished.')
        return 
        
    @tf.function
    def train_step(self, x, y_truth):
        with tf.GradientTape() as tape:
            props = self.model(x, training=True)
            loss = self.loss(props, y_truth)
#             loss += tf.add_n(self.model.losses) # l2 regularization
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optim.apply_gradients(zip(grads, self.model.trainable_variables))
        return props, loss
    
    def train(self):
        '''
        Training | Training process
        '''
        n_steps = self.train_generator.steps_per_epoch
        sum_loss = 0
        sum_acc1 = 0
        sum_acc5 = 0
        with tqdm(self.train_generator.epoch(), total = n_steps,
                  desc = 'Training | Epoch {} | Training'.format(self.epoch)) as pbar:
            for step, (x, y_truth) in enumerate(pbar):
                x = tf.constant(x.astype('float32'))
                y_truth = tf.constant(y_truth.astype('int32'))
                props, loss = self.train_step(x, y_truth)
                sum_loss += loss.numpy()
                acc1, acc5 = accuracy(props.numpy(), y_truth.numpy(), topk=(1,5))
                sum_acc1 += acc1
                sum_acc5 += acc5
                
                # postfix for progress bar
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
        Loss values in validation have not taken the l2 regularization into account.
        '''
        n_steps = self.val_generator.steps_per_epoch
        sum_loss = 0
        sum_acc1 = 0
        sum_acc5 = 0
        with tqdm(self.val_generator.epoch(), total = n_steps,
                  desc = 'Training | Epoch {} | Val'.format(self.epoch)) as pbar:
            for step, (x, y_truth) in enumerate(pbar):
                x = tf.constant(x.astype('float32'))
                y_truth = tf.constant(y_truth.astype('int32'))
                props = self.model(x, training=False)
                loss = self.loss(props, y_truth)
                sum_loss += loss.numpy()
                acc1, acc5 = accuracy(props.numpy(), y_truth.numpy(), topk=(1,5))
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
    t = Training()
    t.main_run()