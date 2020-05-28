import h5py
import numpy as np
from random import shuffle
import pdb
import yaml
import pickle
'''
The pipeline provides ndarray rather than framework-specific formats.
'''
class Generator():
    '''
    ids: id list
    h5f: .h5 file path
    bs: batch size
    '''
    def __init__(self, ids, h5f, bs):
        self.ids = ids
        self.h5f = h5f
        self.bs = bs
        # steps per epoch:
        self.spe = int(np.ceil(len(self.ids)/self.bs)) 
        
    def epoch(self):
        x = []
        y = []
        ids = self.ids.copy()
        shuffle(ids)
        while ids:
            i = ids.pop()
            self.append(x, y, i)
            if len(x) == self.bs or not ids:
                yield self.feed(x, y)
                x = []
                y = []
        return
    
    def append(self, x, y, i):
        with h5py.File(self.h5f, 'r') as f:
            x.append(f['train']['x'][i])
            y.append(f['train']['y'][i])
        return
    
    def feed(self, x, y):
        return np.asarray(x), np.asarray(y)
        
class Dataset():
    def __init__(self, cf='config.yml', cv_id=0, for_train=True):
        '''
        cf: config.yml path
        cv_id: which fold in the cross validation.
        if cv_id >= n_fold: use all the training dataset.
        for_train: if True, for training process, otherwise for searching.
        '''
        with open(cf) as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)
        with open(self.config['data']['cv_file'],'rb') as f:
            self.cv_dict = pickle.load(f)
        self.cv_id = cv_id
        self.n_fold = self.config['data']['n_fold']
        self.for_train = for_train
    
    @property
    def _train_ids(self):
        if self.cv_id >= self.n_fold:
            return self.cv_dict['train_0'] + self.cv_dict['val_0'] 
        else:
            return self.cv_dict['train_{}'.format(self.cv_id)]
        
    @property
    def _val_ids(self):
        if self.cv_id >= self.n_fold:
            return self.cv_dict['train_0'] + self.cv_dict['val_0'] 
        else:
            return self.cv_dict['val_{}'.format(self.cv_id)]
        
    @property
    def train_generator(self):
        return Generator(ids = self._train_ids, 
                         h5f = self.config['data']['preprocessed'], 
                         bs = self.config['train' if self.for_train else 'search']['batchsize'])
    @property
    def val_generator(self):
        return Generator(ids = self._val_ids, 
                         h5f = self.config['data']['preprocessed'], 
                         bs = self.config['train' if self.for_train else 'search']['batchsize'])
    
    
    