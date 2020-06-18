import pickle

def load_opt(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

    