import pickle


def drop_path(x, drop_rate):
    '''
    dropout on the channel dimension.
    drop_rate: probability of an element to be zero-ed.
    This is a backup for torch.nn.Dropout2d.
    '''
    if drop_rate > 0.:
        keep_prob = 1.-drop_rate
        mask = torch.zeros(1,x.size(1)).bernoulli_(keep_prob)
        x /= keep_prob
        x *= mask
    return x

class ReduceLROnPlateau():
    '''
    opt: optimizer
    factor (float): Factor by which the learning rate will be
        reduced. new_lr = lr * factor.
    patience (int): Number of epochs with no improvement after
        which learning rate will be reduced. For example, if
        patience = 2, then we will ignore the first 2 epochs
        with no improvement, and will only decrease the LR after the
        3rd epoch if the loss still hasn't improved then.
    verbose (bool): If True, prints a message to stdout for
        each update.
    '''
    def __init__(self, opt, patience=10, factor=0.5, verbose=True):
        self.opt = opt
        self.patience = patience
        self.factor = factor
        self.verbose = verbose
        self.record = float('inf')
        self.count = 0
        
    def step(self, loss):
        if loss >= self.record:
            self.count += 1
            if self.count == self.patience:
                self.opt._learning_rate *= self.factor
                self.count = 0
                self.record = loss
        else:
            self.count = 0
            self.record = loss
        return
    
    def state_dict(self):
        return {'record':self.record,
                'count': self.count,
                'lr': self.opt._learning_rate,
                'patience': self.patience,
                'factor': self.factor}
    def load_state_dict(self, state_dict, full_load=False):
        '''
        state_dict (dict): State_dict to be recovered.
        full_load (bool): If True, recovers the self.patience and self.factor.
        '''
        self.record = state_dict['record']
        self.count = state_dict['count']
        self.opt._learning_rate = state_dict['lr']
        if full_load:
            self.patience = state_dict['patience']
            self.factor = state_dict['factor']
        return
    
def load_opt(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)