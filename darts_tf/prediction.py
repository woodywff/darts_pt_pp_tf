import pdb
import os
from helper import calc_param_size
from .searched import SearchedNet
# from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import OrderedDict
import pickle
from genotype import Genotype
from .search import Base
import pipeline
from .utils import accuracy
import tensorflow as tf
import numpy as np

DEBUG_FLAG = False

    
class Prediction(Base):
    '''
    cf: config.yml path
    '''
    def __init__(self, cf='config.yml'):
        super().__init__(cf=cf)
        self._init_model()
        
    def _init_dataset(self):
        dataset = pipeline.Dataset(cf=self.cf, test_only=True,
                                   channel_last=True)
        self.test_generator = dataset.test_generator
        return
    
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
                                 n_nodes=self.config['search']['n_nodes'])
#         self.model(np.random.rand(1,28,28,1).astype('float32'),training=True)
#         print('Param size = {:.3f} MB'.format(calc_param_size(self.model.trainable_variables)))
        self.loss = lambda props, y_truth: tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(y_truth, props))

        save_path = os.path.join(self.train_log, self.config['train']['best_shot'])
        self.model.load_weights(save_path)

    def predict(self):
        n_steps = self.test_generator.steps_per_epoch
        sum_loss = 0
        sum_acc1 = 0
        sum_acc5 = 0
        with tqdm(self.test_generator.epoch(), total = n_steps,
                  desc = 'Prediction') as pbar:
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
    p = Prediction()
    loss, acc1, acc5 = p.predict()