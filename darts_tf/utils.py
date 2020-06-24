import numpy as np

def accuracy(output, y_truth, topk=(1,)):
    '''
    output: probability output from the network.
    y_truth: ground truth labels.
    topk: for example, topk=(1,2,5) would return top-1, top-2, and top-5 accuracies.
    '''
    maxk = max(topk)
    batch_size = y_truth.shape[0]

    pred = np.argsort(output)[:, ::-1][:, :maxk]
    correct = pred == np.reshape(y_truth, (batch_size, 1))

    res = []
    for k in topk:
        correct_k = np.sum(correct[:, :k])
        res.append(correct_k / batch_size)
    return res[0] if len(res)==1 else res

