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
    is_test: if True, it is the test dataset, otherwise training dataset.
    channel_last: if True, corresponds to inputs with shape [batch, height, width, channels] (for tensorflow),
                  otherwise, [batch, channels, height, width] (for pytorch and paddlepaddle).
    '''
    def __init__(self, ids, h5f, bs, is_test=False, channel_last=False):
        self.ids = ids
        self.h5f = h5f
        self.bs = bs
        self.is_test = is_test
        self.channel_last = channel_last
        # steps per epoch:
        self.spe = int(np.ceil(len(self.ids)/self.bs)) 
        
    @property
    def steps_per_epoch(self):
        return self.spe
        
    def epoch(self):
        x = []
        y = []
        ids = self.ids.copy()
        if not self.is_test:
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
        '''Dataset specific. This is for fashion-mnist'''
        with h5py.File(self.h5f, 'r') as f:
            train_or_test = 'train' if not self.is_test else 'test'
            if self.channel_last:
                x.append(f[train_or_test]['x'][i][:,:,np.newaxis])
            else:
                x.append(f[train_or_test]['x'][i][np.newaxis])
            y.append(f[train_or_test]['y'][i])
        return
    
    def feed(self, x, y):
        return np.asarray(x), np.asarray(y)
        
class Dataset():
    def __init__(self, cf='config.yml', cv_i=0, for_train=False, test_only=False, channel_last=False):
        '''
        cf: config.yml path
        cv_i: which fold in the cross validation.
        if cv_i >= n_fold: use all the training dataset.
        for_train: if True, for training process, otherwise for searching.
        test_only: if True, only for test dataset.
        channel_last: if True, corresponds to inputs with shape (batch, height, width, channels) (for tensorflow),
                  otherwise, (batch, channels, height, width) (for pytorch and paddlepaddle).
        '''
        with open(cf) as f:
            self.config = yaml.load(f,Loader=yaml.FullLoader)
        self.channel_last = channel_last
        if test_only:
            return
        self.search_or_train = 'train' if for_train else 'search'
        cv_file = self.config[self.search_or_train]['cv_file']
        self.n_fold = self.config[self.search_or_train]['n_fold']
        with open(cv_file,'rb') as f:
            self.cv_dict = pickle.load(f)
        self.cv_i = cv_i
    
    @property
    def _train_ids(self):
        if self.cv_i >= self.n_fold:
            return self.cv_dict['train_0'] + self.cv_dict['val_0'] 
        else:
            return self.cv_dict['train_{}'.format(self.cv_i)]
        
    @property
    def _val_ids(self):
        if self.cv_i >= self.n_fold:
            return self.cv_dict['train_0'] + self.cv_dict['val_0'] 
        else:
            return self.cv_dict['val_{}'.format(self.cv_i)]
        
    @property
    def _test_ids(self):
        return list(range(self.config['data']['len_test']))
        
    @property
    def train_generator(self):
        return Generator(ids = self._train_ids, 
                         h5f = self.config['data']['preprocessed'], 
                         bs = self.config[self.search_or_train]['batchsize'],
                         channel_last = self.channel_last)
    @property
    def val_generator(self):
        return Generator(ids = self._val_ids, 
                         h5f = self.config['data']['preprocessed'], 
                         bs = self.config[self.search_or_train]['batchsize'],
                         channel_last = self.channel_last)
    @property
    def test_generator(self):
        return Generator(ids = self._test_ids,
                         h5f = self.config['data']['preprocessed'],
                         bs = self.config['test']['batchsize'],
                         is_test = True,
                         channel_last = self.channel_last)
    
    
    