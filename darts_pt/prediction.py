import pdb
import os
import torch
import torch.nn as nn
from helper import calc_param_size
from .utils import accuracy
from .searched import SearchedNet
# from tqdm import tqdm
from tqdm.notebook import tqdm
from collections import OrderedDict
import pickle
from genotype import Genotype
from .search import Base
import pipeline

DEBUG_FLAG = False

    
class Prediction(Base):
    '''
    cf: config.yml path
    '''
    def __init__(self, cf='config.yml'):
        super().__init__(cf=cf)
        self._init_model()
        
    def _init_dataset(self):
        dataset = pipeline.Dataset(cf=self.cf, test_only=True)
        self.test_generator = dataset.test_generator
        return
    
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
        print('Param size = {:.3f} MB'.format(calc_param_size(self.model.parameters())))
        self.loss = nn.CrossEntropyLoss().to(self.device)

        state_dicts = torch.load(os.path.join(self.log_path, self.config['train']['best_shot']), 
                                 map_location=self.device)
        self.model.load_state_dict(state_dicts['model_param'])
        self.model.eval()

    def predict(self):
        n_steps = self.test_generator.steps_per_epoch
        sum_loss = 0
        sum_acc1 = 0
        sum_acc5 = 0
        with tqdm(self.test_generator.epoch(), total = n_steps,
                  desc = 'Prediction') as pbar:
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
    p = Prediction()
    loss, acc1, acc5 = p.predict()