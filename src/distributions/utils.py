import pickle

def load_distribution(path):
    f = open(path, "rb")
    distribution = pickle.load(f)
    f.close()
    return distribution

def store_distribution(distribution, path):
    f = open(path, "wb")
    pickle.dump(distribution, f)
    f.close()
