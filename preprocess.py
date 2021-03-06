import pickle
import yaml
from helper import load_fmnist
from random import shuffle
import numpy as np
import os
import pdb
import h5py

def cv_split(n_ids, saved_path, n_fold=5, overwrite=False):
    '''
    To generate k-fold-cross-validation indices.
    {'train_0':[],'val_0':[],'train_1':[],'val_1':[],...} is saved as .pkl 
    '''
    if os.path.exists(saved_path) and not overwrite:
        print('{:s} exists already.'.format(saved_path))
        return
    ids = list(range(n_ids))
    shuffle(ids)
    res = {}
    for i in range(n_fold):
        left = int(i/n_fold * n_ids)
        right = int((i+1)/n_fold * n_ids)
        res['train_{}'.format(i)] = ids[:left] + ids[right:]
        res['val_{}'.format(i)] = ids[left : right]
    for i in res.values():
        shuffle(i)
    with open(saved_path,'wb') as f:
        pickle.dump(res,f)
    return


def preprocess(config_yml='config.yml', overwrite=False):
    '''
    Load fashion-mnist dataset.
    Normalizations.
    Cross-validation preparation for searching and training processes.
    '''
#     pdb.set_trace()
    with open(config_yml) as f:
        config = yaml.load(f,Loader=yaml.FullLoader)
    # get .h5 file prepared
    pre_path = config['data']['preprocessed']
    if os.path.exists(pre_path) and not overwrite:
        print('{} exists already!'.format(pre_path))
    else:
        x_train, y_train = load_fmnist(config['data']['dataset_path'])
        x_test, y_test = load_fmnist(config['data']['dataset_path'], is_train=False)
        with open(config_yml,'w') as f:
            config['data'].update({'len_train':len(y_train),
                                   'len_test':len(y_test)})
            yaml.dump(config,f)

        # z-score normalization
        mean = np.mean(x_train)
        std = np.std(x_train) # zero-devide checked
        train_x = (x_train-mean)/std # x_train is not writable
        test_x = (x_test-mean)/std

        # min-max normalization
        minmax = lambda x: (x - np.min(x, axis=-1).reshape(-1,1))/(np.max(x, axis=-1)-np.min(x, axis=-1)).reshape(-1,1)
        train_x = minmax(train_x)
        test_x = minmax(test_x)

        train_x = train_x.reshape(-1,28,28)
        test_x = test_x.reshape(-1,28,28)

        with h5py.File(config['data']['preprocessed'],'w') as h5f:
            g = h5f.create_group('train')
            g.create_dataset('x', data = train_x)
            g.create_dataset('y', data = y_train)
            g = h5f.create_group('test')
            g.create_dataset('x', data = test_x)
            g.create_dataset('y', data = y_test)

    # split for cross validation
    for i in ['search','train']:
        cv_file = config[i]['cv_file']
        if os.path.exists(cv_file) and not overwrite:
            print('{} exists already!'.format(cv_file))
        else:
            cv_split(n_ids=config['data']['len_train'], 
                     saved_path=cv_file, 
                     n_fold=config[i]['n_fold'], 
                     overwrite=overwrite)
    
    return

if __name__ == '__main__':
    preprocess()
