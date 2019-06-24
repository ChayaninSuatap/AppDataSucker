import pickle
def save_pickle(obj, fn):
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)

def load_picle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)

def load_pickle(fn):
    with open(fn, 'rb') as f:
        return pickle.load(f)