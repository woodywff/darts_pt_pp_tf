import torch

def accuracy(output, target, topk=(1,)):
    '''
    output: probability output from the network.
    target: class labels.
    topk: for example, topk=(1,2,5) would return top-1, top-2, and top-5 accuracies.
    '''
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(k=maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = torch.sum(correct[:k].view(-1).float())
        res.append(round((correct_k/batch_size).item(),3))
    return res   

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